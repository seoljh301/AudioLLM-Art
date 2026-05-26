"""MVP-B entry point: audio -> caption -> mutate text -> TTA -> audio (recursive).

Two modes:
  --mode render --input seed.wav --output out_dir/          # offline batch loop
  --mode serve                                               # OSC server
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from src.core.osc_bridge import OSCBridge, OSCBridgeConfig
from src.modules.mvp_b.caption_loop import LoopParams, run_loop
from src.modules.mvp_b.models import (
    AudioLDM2Config,
    Qwen2AudioConfig,
    make_audioldm2_synth,
    make_qwen2_audio_caption,
    make_stub_caption,
    make_stub_synth,
)


def _build_caption(cfg: dict, sample_rate: int):
    backend = cfg["caption"].get("backend", "stub")
    if backend == "stub":
        return make_stub_caption(sample_rate)
    if backend == "qwen2_audio":
        qcfg = Qwen2AudioConfig(
            model_id=cfg["caption"].get("model", "Qwen/Qwen2-Audio-7B-Instruct"),
            prompt=cfg["caption"].get("prompt", "Describe the sound in one short sentence."),
            max_new_tokens=int(cfg["caption"].get("max_new_tokens", 48)),
            device=cfg["caption"].get("device", "cpu"),
        )
        return make_qwen2_audio_caption(qcfg, sample_rate)
    raise ValueError(f"unknown caption backend: {backend}")


def _build_synth(cfg: dict, sample_rate: int):
    backend = cfg["tta"].get("backend", "stub")
    if backend == "stub":
        return make_stub_synth(sample_rate, float(cfg["tta"].get("duration_s", 6.0)))
    if backend == "audioldm2":
        acfg = AudioLDM2Config(
            model_id=cfg["tta"].get("model", "cvssp/audioldm2"),
            duration_s=float(cfg["tta"].get("duration_s", 6.0)),
            guidance_scale=float(cfg["tta"].get("guidance_scale", 3.5)),
            num_inference_steps=int(cfg["tta"].get("num_inference_steps", 50)),
            device=cfg["tta"].get("device", "cpu"),
        )
        return make_audioldm2_synth(acfg)
    raise ValueError(f"unknown tta backend: {backend}")


def _load_seed(path: Path, sample_rate: int) -> np.ndarray:
    import librosa  # noqa: WPS433
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate).astype(np.float32)
    return audio


def _run_loop_and_save(
    seed_audio: np.ndarray,
    cfg: dict,
    out_dir: Path,
    caption_fn,
    synth_fn,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = LoopParams(
        depth=int(cfg["loop"]["depth"]),
        text_mutation_prob=float(cfg["loop"]["text_mutation_prob"]),
        mutation_pool=list(cfg["loop"]["mutation_pool"]),
        seed=int(cfg.get("seed", 0)),
    )
    sr = int(cfg["sample_rate"])

    sf.write(out_dir / "step_00_seed.wav", seed_audio, sr)
    history = run_loop(seed_audio, caption_fn, synth_fn, params)

    transcript = []
    for i, (text, audio) in enumerate(history, start=1):
        path = out_dir / f"step_{i:02d}.wav"
        sf.write(path, audio, sr)
        transcript.append({"step": i, "text": text, "audio": path.name})
        logging.info("saved step %d -> %s : %r", i, path, text)

    transcript_path = out_dir / "transcript.json"
    transcript_path.write_text(json.dumps({
        "depth": params.depth,
        "text_mutation_prob": params.text_mutation_prob,
        "seed": params.seed,
        "caption_backend": cfg["caption"].get("backend", "stub"),
        "tta_backend": cfg["tta"].get("backend", "stub"),
        "history": transcript,
    }, indent=2))
    logging.info("transcript -> %s", transcript_path)
    return transcript_path


def _run_render(cfg: dict, args: argparse.Namespace) -> None:
    sr = int(cfg["sample_rate"])
    caption_fn = _build_caption(cfg, sr)
    synth_fn = _build_synth(cfg, sr)
    seed = _load_seed(Path(args.input), sr)
    out_dir = Path(args.output)
    _run_loop_and_save(seed, cfg, out_dir, caption_fn, synth_fn)


def _run_serve(cfg: dict) -> None:
    sr = int(cfg["sample_rate"])
    caption_fn = _build_caption(cfg, sr)
    synth_fn = _build_synth(cfg, sr)
    bridge = OSCBridge(OSCBridgeConfig(**cfg["osc"]))

    def on_start(_addr: str, in_path: str, out_dir: str) -> None:
        logging.info("OSC start: seed=%s out_dir=%s", in_path, out_dir)
        try:
            seed = _load_seed(Path(in_path), sr)
            transcript = _run_loop_and_save(seed, cfg, Path(out_dir), caption_fn, synth_fn)
            bridge.send("/mvp_b/done", str(transcript))
        except Exception as e:  # noqa: BLE001
            logging.exception("loop failed")
            bridge.send("/mvp_b/error", repr(e))

    bridge.register("/mvp_b/start", on_start)
    logging.info("MVP-B serve ready on %s:%d",
                 cfg["osc"]["listen_host"], cfg["osc"]["listen_port"])
    bridge.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-B: caption->TTA recursive loop")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--mode", choices=["render", "serve"], default="render")
    parser.add_argument("--input", type=str, default=None, help="seed wav (render mode)")
    parser.add_argument("--output", type=str, default=None, help="output directory (render mode)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "render":
        if not args.input or not args.output:
            parser.error("--mode render requires --input and --output")
        _run_render(cfg, args)
    else:
        _run_serve(cfg)


if __name__ == "__main__":
    main()
