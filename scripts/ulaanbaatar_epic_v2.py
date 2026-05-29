import os
import subprocess
import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

# V1/V2 Imports
from src.modules.mvp_a.rave_io import load_rave, RaveHandle
from src.modules.mvp_c.codec_io import load_codec, CodecHandle
from src.modules.mvp_d.morph_io import load_morph
from src.modules.mvp_d.ckpt_morph import MorphParams
from src.modules.mvp_j.mimi_bend import MimiBender, MimiBendParams
from src.core.v2.llm_helmsman import LLMHelmsman
from src.core.v2.disagreement_noise import MultiAgentNoise, apply_disagreement_noise
from src.core.v2.hallucinatory_midi import HallucinatoryMIDI

# Stage runners from multinet (re-used for deep pipes)
from scripts.multinet import (
    stage_A, stage_B, stage_C, stage_D, stage_E, stage_F, stage_G, stage_H, stage_I, 
    stage_J, stage_K, stage_L, _resample_to
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("ulaanbaatar_epic_v2")

INPUT_WAV = "data/Ulaanbaatar.wav"
OUT_DIR = Path("runs/ulaanbaatar_epic_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAVE_GUITAR = "checkpoints/rave/guitar_iil_b2048_r48000_z16.ts"
RAVE_ORGAN = "checkpoints/rave/organ_archive_b2048_r48000_z16.ts"

def run_shell(cmd):
    log.info(f"Exec: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def build_epic():
    log.info("Starting Ulaanbaatar Epic V2: The Neural Mountain Range (Fractal Architecture)")
    
    # --- STEP 0: Split Input into Frequency Bands (WIDE) ---
    sub_raw = OUT_DIR / "split_sub.wav"
    mid_raw = OUT_DIR / "split_mid.wav"
    high_raw = OUT_DIR / "split_high.wav"
    
    if not sub_raw.exists():
        log.info("Splitting frequency bands via SoX...")
        run_shell(f"sox {INPUT_WAV} {sub_raw} sinc -80") # Sub < 80Hz
        run_shell(f"sox {INPUT_WAV} {mid_raw} sinc 80-4000") # Mid 80-4k
        run_shell(f"sox {INPUT_WAV} {high_raw} sinc 4000") # High > 4k

    # Load Models
    rave_g = load_rave(RAVE_GUITAR, device=DEVICE)
    morph = load_morph([RAVE_GUITAR, RAVE_ORGAN], MorphParams(mode="linear", t=0.005), device=DEVICE)
    codec = load_codec("encodec_24khz", device=DEVICE)
    
    # -------------------------------------------------------------------------
    # [WIDE 1] ALPHA ABYSS (Deep Pipe: EQ -> I -> G)
    # -------------------------------------------------------------------------
    # Renamed to stem_alpha_nowub.wav to force regeneration without the LFO wobble
    if not (OUT_DIR / "stem_alpha_nowub.wav").exists():
        log.info("--- Processing Alpha Abyss (Sub-Bass, NO WUB) ---")
        alpha_eq = OUT_DIR / "alpha_0_eq.wav"
        if not alpha_eq.exists():
            run_shell(f"sox {sub_raw} {alpha_eq} bass +20 40 0.5q equalizer 150 1.0q -6 norm -1")
        
        audio, sr = sf.read(alpha_eq)
        audio = audio.astype(np.float32)
        if audio.ndim > 1: audio = np.mean(audio, axis=1)
        
        # I (Bass Massive) -> G (Feedback)
        log.info("  Alpha Stage I (Smear x3)...")
        x = stage_I(_resample_to(audio, int(len(audio)*24000/sr)), codec, smear=30, jitter=0.1, fold=0.02, dw=0.7)
        log.info("  Alpha Stage G (Deep Echo)...")
        alpha_final = stage_G(_resample_to(x, len(audio)), rave_g, delay=128, feedback=0.65, dw=0.6)
        # Stage A (Sub wobble) removed
        sf.write(OUT_DIR / "stem_alpha_nowub.wav", alpha_final, sr)
        del audio, x, alpha_final; torch.cuda.empty_cache()
    else:
        log.info("Skipping Alpha Abyss (stem_alpha_nowub.wav exists)")

    # -------------------------------------------------------------------------
    # [WIDE 2] BETA SENTIENCE (Deep Pipe: J -> B -> D -> E)
    # -------------------------------------------------------------------------
    if not (OUT_DIR / "stem_beta.wav").exists():
        log.info("--- Processing Beta Sentience (Mids) ---")
        audio, sr = sf.read(mid_raw)
        audio = audio.astype(np.float32)
        if audio.ndim > 1: audio = np.mean(audio, axis=1)
        
        # J (Mimi Bender) -> B (Caption Loop) -> D (Morphing) -> E (Granular)
        log.info("  Beta Stage J (Mimi Bender)...")
        x = stage_J(audio, semantic_rate=0.03, acoustic_rate=0.15)
        log.info("  Beta Stage B (Caption Loop)...")
        x = stage_B(x, rave_g, sr, depth=2)
        log.info("  Beta Stage D (Morphing)...")
        x = stage_D(x, morph, dw=0.6)
        log.info("  Beta Stage E (Latent Granular)...")
        beta_final = stage_E(x, rave_g, grain=48, mem=8192, num=8, dw=0.55)
        sf.write(OUT_DIR / "stem_beta.wav", beta_final, sr)
        del audio, x, beta_final; torch.cuda.empty_cache()
    else:
        log.info("Skipping Beta Sentience (stem_beta.wav exists)")

    # -------------------------------------------------------------------------
    # [WIDE 3] GAMMA DECAY (Deep Pipe: K -> C -> F)
    # -------------------------------------------------------------------------
    if not (OUT_DIR / "stem_gamma.wav").exists():
        log.info("--- Processing Gamma Decay (Highs) ---")
        audio, sr = sf.read(high_raw)
        audio = audio.astype(np.float32)
        if audio.ndim > 1: audio = np.mean(audio, axis=1)
        
        # Save duration for Delta Ether before we delete audio
        dur_s = len(audio) / sr

        # K (Temporal Warp) -> C (Token Bend) -> F (Freeze)
        log.info("  Gamma Stage C (Heavy Token Bend)...")
        x_24k = stage_C(_resample_to(audio, int(len(audio)*24000/sr)), codec, mode="bit_flip", rate=0.15, q_range=(-4, 0), dw=0.65)
        log.info("  Gamma Stage F (Spectral Freeze)...")
        gamma_final = stage_F(_resample_to(x_24k, len(audio)), rave_g, auto_upper=0.75, update_interval=48, dw=0.55)
        sf.write(OUT_DIR / "stem_gamma.wav", gamma_final, sr)
        del audio, x_24k, gamma_final; torch.cuda.empty_cache()
    else:
        log.info("Skipping Gamma Decay (stem_gamma.wav exists)")
        # We still need dur_s and sr for Delta Ether
        audio, sr = sf.read(high_raw)
        dur_s = len(audio) / sr
        del audio

    # -------------------------------------------------------------------------
    # [WIDE 4] DELTA ETHER (Deep Pipe: L -> H)
    # -------------------------------------------------------------------------
    if not (OUT_DIR / "stem_delta.wav").exists():
        log.info("--- Generating Delta Ether (Air) ---")
        # L (Restoration from Noise) -> H (Generative)
        # sr is 48000
        log.info("  Delta Stage L (Dream from White Noise)...")
        x_noise = stage_L(dur_s, sr)
        log.info("  Delta Stage H (Codebook Organ - Prime)...")
        frames = int(len(x_noise) * 75 / sr)
        h_24k = stage_H(codec, mode="prime", stride=11, frames=frames)
        delta_final = 0.5 * x_noise + 0.5 * _resample_to(h_24k, len(x_noise))
        sf.write(OUT_DIR / "stem_delta.wav", delta_final, sr)
        del x_noise, delta_final
    else:
        log.info("Skipping Delta Ether (stem_delta.wav exists)")

    # -------------------------------------------------------------------------
    # MASTER INTEGRATION (V2 CENTRAL NERVE)
    # -------------------------------------------------------------------------
    log.info("--- Final Integration: PURE NEURAL BASS (NO ORIGINAL) ---")
    
    final_out = OUT_DIR / "ULAANBAATAR_PURE_NEURAL_BASS.wav"
    
    # 1. Dual-Band Filtering using gentle Butterworth filters
    log.info("  Filtering neural stems into Subwoofer (<80Hz) and Woofer (60-250Hz) bands...")
    
    # Force ALL bass stems to pure Mono (`remix -`)
    run_shell(f"sox {OUT_DIR}/stem_alpha_nowub.wav {OUT_DIR}/sub_alpha.wav remix - lowpass -2 80")
    run_shell(f"sox {OUT_DIR}/stem_delta.wav {OUT_DIR}/sub_delta.wav remix - lowpass -2 80")
    
    run_shell(f"sox {OUT_DIR}/stem_beta.wav {OUT_DIR}/midbass_beta.wav remix - highpass -2 60 lowpass -2 250")
    run_shell(f"sox {OUT_DIR}/stem_gamma.wav {OUT_DIR}/midbass_gamma.wav remix - highpass -2 60 lowpass -2 250")
    
    # 2. Mix the mono neural bass layers
    log.info("  Mixing Neural Bass (No Original)...")
    run_shell(f"sox -m -v 1.0 {OUT_DIR}/sub_alpha.wav -v 0.5 {OUT_DIR}/sub_delta.wav "
              f"-v 0.8 {OUT_DIR}/midbass_beta.wav -v 0.6 {OUT_DIR}/midbass_gamma.wav "
              f"{OUT_DIR}/neural_bass_raw.wav")
              
    # 3. Clean Mastering (No Pumping)
    # Replaced aggressive compander with a smooth soft-knee limiter and volume normalization
    # Output to dual-mono (-c 2 ... remix 1 1)
    run_shell(f"sox {OUT_DIR}/neural_bass_raw.wav -c 2 {final_out} norm -0.5 compand 0.05,0.2 6:-40,-40,0,-1 -1 -3 remix 1 1")
    
    log.info(f"SUCCESS: Pure Neural Bass completed at {final_out}")

if __name__ == "__main__":
    build_epic()
