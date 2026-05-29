import os
import subprocess
import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import scipy.signal as signal

# V1/V2 Imports
from src.modules.mvp_a.rave_io import load_rave
from src.modules.mvp_c.codec_io import load_codec
from scripts.multinet import stage_A, stage_C, stage_E, stage_F, stage_G, stage_I, stage_H, _resample_to

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("v3_opus_idm")

OUT_DIR = Path("runs/masterpiece/v3_opus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_shell(cmd):
    log.info(f"Exec: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def generate_base_stems(dur=300.0, sr=48000, bpm=110):
    log.info("Generating 5-minute algorithmic electronic stems...")
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    bps = bpm / 60.0
    beat_phase = (t * bps) % 1.0
    
    # 1. Foundation (Kick + Sub Bass in C Minor)
    # Kick: pitch drop
    kick_env = np.exp(-15 * beat_phase)
    kick_pitch = 150 * np.exp(-30 * beat_phase) + 40
    kick = np.sin(2 * np.pi * kick_pitch * t) * kick_env
    
    # Sub Bass (C1 = 32.7Hz, Eb1 = 38.89Hz, G1 = 49.0Hz)
    bar_phase = (t * (bps / 4)) % 1.0
    sub_freqs = np.where(bar_phase < 0.5, 32.7, np.where(bar_phase < 0.75, 38.89, 49.0))
    sub = np.sin(2 * np.pi * sub_freqs * t) * 0.8
    foundation = (kick + sub).astype(np.float32)
    sf.write(OUT_DIR / "raw_foundation.wav", foundation, sr)
    
    # 2. Harmonic (Chords: Cm9 -> Abmaj7 -> Fm7 -> G7)
    pad = np.zeros_like(t)
    chord_prog = [
        [130.81, 155.56, 196.00, 233.08, 293.66], # Cm9
        [103.83, 155.56, 196.00, 261.63, 311.13], # Abmaj7
        [87.31, 130.81, 174.61, 233.08, 261.63],  # Fm7
        [98.00, 146.83, 196.00, 246.94, 293.66]   # G7
    ]
    
    # Vectorized chord generation
    bar_duration_samples = int(sr * 4 * (60.0 / bpm)) # 4 beats per bar
    for i, freqs in enumerate(chord_prog):
        # We loop this 4-bar progression over the 5 minutes
        start_idx = 0
        while start_idx < len(t):
            # The current chord plays for 1 bar, every 4 bars
            bar_start = start_idx + (i * bar_duration_samples)
            bar_end = min(bar_start + bar_duration_samples, len(t))
            if bar_start < len(t):
                time_slice = t[bar_start:bar_end]
                chord_val = sum(signal.sawtooth(2 * np.pi * f * time_slice) for f in freqs)
                pad[bar_start:bar_end] += chord_val
            start_idx += 4 * bar_duration_samples
    
    # Lowpass filter the pad for lushness
    b, a = signal.butter(2, 800, fs=sr, btype='low')
    pad = signal.lfilter(b, a, pad)
    # Slow LFO tremolo
    pad = pad * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
    pad = (pad / np.max(np.abs(pad)) * 0.7).astype(np.float32)
    sf.write(OUT_DIR / "raw_harmonic.wav", pad, sr)
    
    # 3. Percussive (Hats/Glitch base)
    sixteenth_phase = (t * bps * 4) % 1.0
    hat_env = np.exp(-40 * sixteenth_phase) * (np.sin(2 * np.pi * (bps/2) * t) > 0)
    noise = np.random.standard_normal(len(t))
    b_h, a_h = signal.butter(4, 4000, fs=sr, btype='high')
    perc = signal.lfilter(b_h, a_h, noise * hat_env).astype(np.float32) * 0.5
    sf.write(OUT_DIR / "raw_perc.wav", perc, sr)
    
    return sr

def apply_neural_physics(sr):
    log.info("Loading Neural Models...")
    rave_g = load_rave("checkpoints/rave/guitar_iil_b2048_r48000_z16.ts", device=DEVICE)
    codec = load_codec("encodec_24khz", device=DEVICE)
    
    log.info("--- Applying Neural Physics to Stems ---")
    
    # 1. Foundation -> MVP-I (Bass Massive)
    if not (OUT_DIR / "neural_foundation.wav").exists():
        log.info("Processing Foundation...")
        fnd, _ = sf.read(OUT_DIR / "raw_foundation.wav")
        fnd_i = stage_I(_resample_to(fnd, int(len(fnd)*24000/sr)), codec, smear=20, jitter=0.05, dw=0.6)
        sf.write(OUT_DIR / "neural_foundation.wav", _resample_to(fnd_i, len(fnd)), sr)
        del fnd, fnd_i; torch.cuda.empty_cache()
    
    # 2. Harmonic -> MVP-E (Granular) -> MVP-F (Freeze)
    if not (OUT_DIR / "neural_harmonic.wav").exists():
        log.info("Processing Harmonics (Pad)...")
        pad, _ = sf.read(OUT_DIR / "raw_harmonic.wav")
        pad_e = stage_E(pad, rave_g, grain=32, mem=4096, num=4, mix_amt=0.6, dw=0.7)
        pad_f = stage_F(pad_e, rave_g, auto_upper=0.6, update_interval=96, cross=24, dw=0.6)
        sf.write(OUT_DIR / "neural_harmonic.wav", pad_f, sr)
        del pad, pad_e, pad_f; torch.cuda.empty_cache()
    
    # 3. Percussive -> MVP-C (Token Bend)
    if not (OUT_DIR / "neural_perc.wav").exists():
        log.info("Processing Percussion (Glitch)...")
        perc, _ = sf.read(OUT_DIR / "raw_perc.wav")
        perc_c = stage_C(_resample_to(perc, int(len(perc)*24000/sr)), codec, mode="bit_flip", rate=0.05, q_range=(-3, 0), dw=0.8)
        sf.write(OUT_DIR / "neural_perc.wav", _resample_to(perc_c, len(perc)), sr)
        del perc, perc_c; torch.cuda.empty_cache()
    
    # 4. Ether -> MVP-H (Generative Drone)
    if not (OUT_DIR / "neural_ether.wav").exists():
        log.info("Generating Ether...")
        frames = int((300.0) * 75) # 5 mins @ 75fps
        h_24k = stage_H(codec, mode="prime", stride=7, frames=frames)
        sf.write(OUT_DIR / "neural_ether.wav", _resample_to(h_24k, int(300*sr)), sr)
        torch.cuda.empty_cache()

def mix_and_master(sr, dur=300.0):
    log.info("--- Orchestrating the 5-Minute Arrangement ---")
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    
    # Envelopes for composition structure
    # 0-60s: Intro (Ether + Pad)
    # 60-150s: Build (Add Perc + Sub)
    # 150-240s: Climax (All)
    # 240-300s: Outro (Fade out Sub/Perc, Pad dominates)
    
    env_fnd = np.interp(t, [0, 60, 90, 240, 270, 300], [0, 0, 1, 1, 0, 0])
    env_pad = np.interp(t, [0, 30, 270, 300], [0, 1, 1, 0])
    env_prc = np.interp(t, [0, 90, 120, 240, 250, 300], [0, 0, 0.8, 0.8, 0, 0])
    env_eth = np.interp(t, [0, 60, 240, 300], [0.6, 0.3, 0.3, 0])
    
    fnd, _ = sf.read(OUT_DIR / "neural_foundation.wav")
    pad, _ = sf.read(OUT_DIR / "neural_harmonic.wav")
    prc, _ = sf.read(OUT_DIR / "neural_perc.wav")
    eth, _ = sf.read(OUT_DIR / "neural_ether.wav")
    
    # Ensure all stems are the exact same length before mixing
    min_len = min(len(fnd), len(pad), len(prc), len(eth), len(t))
    fnd = fnd[:min_len]
    pad = pad[:min_len]
    prc = prc[:min_len]
    eth = eth[:min_len]
    
    t = t[:min_len]
    env_fnd = env_fnd[:min_len]
    env_pad = env_pad[:min_len]
    env_prc = env_prc[:min_len]
    env_eth = env_eth[:min_len]
    
    # Apply spatial panning
    def stereo_pan(audio, pan_val): # -1 to 1
        l = audio * np.cos((pan_val + 1.0) * np.pi / 4.0)
        r = audio * np.sin((pan_val + 1.0) * np.pi / 4.0)
        return np.column_stack((l, r))
    
    mix = np.zeros((min_len, 2), dtype=np.float32)
    mix += stereo_pan(fnd * env_fnd, 0.0)             # Bass dead center
    mix += stereo_pan(pad * env_pad, 0.2 * np.sin(t)) # Pad slow drift right
    mix += stereo_pan(prc * env_prc, -0.3 * np.cos(t))# Perc drift left
    mix += stereo_pan(eth * env_eth, 0.0)             # Ether center surround
    
    sf.write(OUT_DIR / "V3_OPUS_UNMASTERED.wav", mix, sr)
    
    log.info("--- Final High-End Mastering ---")
    final_out = OUT_DIR / "V3_OPUS_NEURAL_IDM_MASTER.wav"
    
    # 1. Warm saturation + Soft limiting + -12 LUFS perceived loudness targeting
    run_shell(f"sox {OUT_DIR}/V3_OPUS_UNMASTERED.wav {final_out} norm -0.5 compand 0.02,0.1 6:-40,-40,0,-1 -1 -3")
    
    log.info(f"SUCCESS! 5-Minute Opus generated: {final_out}")

if __name__ == "__main__":
    sr = generate_base_stems()
    apply_neural_physics(sr)
    mix_and_master(sr)
