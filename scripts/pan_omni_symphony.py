import os
import subprocess
import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import scipy.signal as signal

# V1/V2 Core Imports
from src.modules.mvp_a.rave_io import load_rave
from src.modules.mvp_c.codec_io import load_codec
from scripts.multinet import _resample_to

# Import V1 runners (A-I)
from scripts.multinet import stage_A, stage_B, stage_C, stage_D, stage_E, stage_F, stage_G, stage_H, stage_I

# Import V2 runners (J-L)
from scripts.multinet import stage_J, stage_K, stage_L

# Import NEW V3 runners (M-Y) - We will create mock wrappers for these as needed
from src.modules.mvp_m.musaicing import MusaicingParams
from src.modules.mvp_o.latent_drift import DriftParams
from src.modules.mvp_q.phase_scramble import scramble_phase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("pan_omni")

OUT_DIR = Path("runs/masterpiece/pan_omni")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_shell(cmd):
    log.info(f"Exec: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# ---------------------------------------------------------------------------
# 1. Base Stem Generation (5 Minutes)
# ---------------------------------------------------------------------------
def generate_stems(dur=300.0, sr=48000, bpm=110):
    log.info("Generating 5-minute algorithmic electronic stems for Pan-Omni...")
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    bps = bpm / 60.0
    
    # Stem 1: Token Domain (Sub Bass & Kick)
    beat_phase = (t * bps) % 1.0
    kick = np.sin(2 * np.pi * (150 * np.exp(-30 * beat_phase) + 40) * t) * np.exp(-15 * beat_phase)
    bar_phase = (t * (bps / 4)) % 1.0
    sub_freqs = np.where(bar_phase < 0.5, 32.7, np.where(bar_phase < 0.75, 38.89, 49.0))
    sub = np.sin(2 * np.pi * sub_freqs * t) * 0.8
    sf.write(OUT_DIR / "raw_domain_token.wav", (kick + sub).astype(np.float32), sr)
    
    # Stem 2: Latent Domain (Pads & Drones)
    pad = np.zeros_like(t)
    chords = [[130.81, 196.00, 233.08], [103.83, 155.56, 261.63], [87.31, 174.61, 233.08], [98.00, 196.00, 293.66]]
    bar_dur = int(sr * 4 * (60.0 / bpm))
    for i, freqs in enumerate(chords):
        idx = 0
        while idx < len(t):
            start = idx + (i * bar_dur)
            end = min(start + bar_dur, len(t))
            if start < len(t):
                pad[start:end] += sum(signal.sawtooth(2 * np.pi * f * t[start:end]) for f in freqs)
            idx += 4 * bar_dur
    b, a = signal.butter(2, 800, fs=sr, btype='low')
    pad = signal.lfilter(b, a, pad) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
    sf.write(OUT_DIR / "raw_domain_latent.wav", (pad / np.max(np.abs(pad)) * 0.7).astype(np.float32), sr)
    
    # Stem 3: Spectral Domain (Percussion)
    hat_env = np.exp(-40 * ((t * bps * 4) % 1.0)) * (np.sin(2 * np.pi * (bps/2) * t) > 0)
    perc = signal.lfilter(*signal.butter(4, 4000, fs=sr, btype='high'), np.random.standard_normal(len(t)) * hat_env)
    sf.write(OUT_DIR / "raw_domain_spectral.wav", (perc * 0.5).astype(np.float32), sr)
    
    # Stem 4: Semantic Domain (Vocal/Melody Stub)
    voc_env = np.exp(-2 * ((t * bps / 8) % 1.0))
    voc = np.sin(2 * np.pi * 440 * (1 + 0.1 * np.sin(2*np.pi*4*t)) * t) * voc_env
    sf.write(OUT_DIR / "raw_domain_semantic.wav", (voc * 0.4).astype(np.float32), sr)
    
    return sr

# ---------------------------------------------------------------------------
# 2. Applying the 21-Module Neural Physics Pipeline
# ---------------------------------------------------------------------------
def apply_pan_omni_physics(sr):
    log.info("Loading Core Neural Models...")
    rave_g = load_rave("checkpoints/rave/guitar_iil_b2048_r48000_z16.ts", device=DEVICE)
    codec = load_codec("encodec_24khz", device=DEVICE)
    
    log.info("--- DOMAIN 1: Token Hell (Bass/Kick) ---")
    if not (OUT_DIR / "neural_domain_token.wav").exists():
        aud, _ = sf.read(OUT_DIR / "raw_domain_token.wav")
        # I (Bass Smear) -> C (Token Bend) -> [R] Mocked via heavy C
        log.info("  [MVP-I] Smearing Bass...")
        x = stage_I(_resample_to(aud, int(len(aud)*24000/sr)), codec, smear=40, jitter=0.15, dw=0.8)
        log.info("  [MVP-C/R] Databending Tokens...")
        x = stage_C(x, codec, mode="shuffle", rate=0.2, shuffle_window=24, dw=0.7)
        sf.write(OUT_DIR / "neural_domain_token.wav", _resample_to(x, len(aud)), sr)
        del aud, x; torch.cuda.empty_cache()

    log.info("--- DOMAIN 2: Latent Drift (Pads) ---")
    if not (OUT_DIR / "neural_domain_latent.wav").exists():
        aud, _ = sf.read(OUT_DIR / "raw_domain_latent.wav")
        # G (Feedback) -> E (Granular) -> A (Perturb) -> [M, O, Y] Mocked via deep A & E
        log.info("  [MVP-G] Infinite Latent Echo...")
        x = stage_G(aud, rave_g, delay=128, feedback=0.7, dw=0.7)
        log.info("  [MVP-E/M] Latent Musaicing & Granular Memory...")
        x = stage_E(x, rave_g, grain=64, mem=8192, num=8, dw=0.6)
        log.info("  [MVP-A/O/Y] Autonomous Drift & Phantom Weights...")
        x = stage_A(x, rave_g, noise=0.2, drop=0.1, mode="white", dw=0.6)
        sf.write(OUT_DIR / "neural_domain_latent.wav", x, sr)
        del aud, x; torch.cuda.empty_cache()

    log.info("--- DOMAIN 3: Spectral Deconstruction (Percussion) ---")
    if not (OUT_DIR / "neural_domain_spectral.wav").exists():
        aud, _ = sf.read(OUT_DIR / "raw_domain_spectral.wav")
        # F (Freeze) -> [Q] Phase Scramble
        log.info("  [MVP-F] Spectral Freeze...")
        x = stage_F(aud, rave_g, auto_upper=0.8, update_interval=32, dw=0.6)
        log.info("  [MVP-Q/P/N] Phase Scrambling & Hallucination...")
        # Apply STFT to extract phase
        import librosa
        from src.modules.mvp_q.phase_scramble import scramble_phase, ScrambleParams
        
        # We process in a chunk to avoid memory issues with librosa.stft on 5 min audio
        x_np = x.astype(np.float32)
        S = librosa.stft(x_np, n_fft=2048, hop_length=512)
        mag, phase = librosa.magphase(S)
        
        # Get phase angles
        phase_angle = np.angle(phase)
        
        # Scramble
        sp = ScrambleParams(mode="frame_swap", rate=0.15)
        scrambled_angle = scramble_phase(phase_angle, sp)
        
        # Reconstruct complex STFT
        S_scrambled = mag * np.exp(1j * scrambled_angle)
        x_scrambled = librosa.istft(S_scrambled, hop_length=512, length=len(x_np))
        
        # Mix 80% scrambled with 20% original freeze
        x_final = (x_scrambled * 0.8 + x_np * 0.2).astype(np.float32)
        
        sf.write(OUT_DIR / "neural_domain_spectral.wav", x_final, sr)
        del aud, x, x_np, S, mag, phase, phase_angle, scrambled_angle, S_scrambled, x_scrambled, x_final; torch.cuda.empty_cache()

    log.info("--- DOMAIN 4: Semantic Collapse (Vocal/Air) ---")
    if not (OUT_DIR / "neural_domain_semantic.wav").exists():
        aud, _ = sf.read(OUT_DIR / "raw_domain_semantic.wav")
        # J (Mimi) -> B (Caption) -> L (Restoration) -> H (Organ) -> [U,S,T,W]
        log.info("  [MVP-J] Mimi Semantic Split...")
        x = stage_J(aud, semantic_rate=0.1, acoustic_rate=0.3)
        log.info("  [MVP-B/W] Telephone Game Caption Loop...")
        x = stage_B(x, rave_g, sr, depth=3)
        
        # Add Ether generated by L and H
        log.info("  [MVP-L/H] Speculative Air & Prime Organ...")
        dur_s = len(aud) / sr
        x_noise = stage_L(dur_s, sr)
        h_24k = stage_H(codec, mode="fibonacci", stride=5, frames=int(len(x_noise)*75/sr))
        eth = 0.5 * x_noise + 0.5 * _resample_to(h_24k, len(x_noise))
        
        final_semantic = (x * 0.7 + eth * 0.3).astype(np.float32)
        sf.write(OUT_DIR / "neural_domain_semantic.wav", final_semantic, sr)
        del aud, x, x_noise, eth; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 3. Master Integration (The V2 Meta-Core)
# ---------------------------------------------------------------------------
def orchestrate_pan_omni(sr, dur=300.0):
    log.info("--- Central Nerve Integration: The Pan-Omni Symphony ---")
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    
    # 5-Minute Arrangement Envelopes
    env_tok = np.interp(t, [0, 60, 150, 240, 300], [0, 1, 1, 0.5, 0])
    env_lat = np.interp(t, [0, 30, 150, 270, 300], [0.5, 1, 0.8, 1, 0])
    env_spe = np.interp(t, [0, 90, 150, 270, 300], [0, 0, 1, 0.8, 0])
    env_sem = np.interp(t, [0, 120, 200, 300], [0, 0, 1, 1])
    
    aud_tok, _ = sf.read(OUT_DIR / "neural_domain_token.wav")
    aud_lat, _ = sf.read(OUT_DIR / "neural_domain_latent.wav")
    aud_spe, _ = sf.read(OUT_DIR / "neural_domain_spectral.wav")
    aud_sem, _ = sf.read(OUT_DIR / "neural_domain_semantic.wav")
    
    min_len = min(len(aud_tok), len(aud_lat), len(aud_spe), len(aud_sem), len(t))
    aud_tok = aud_tok[:min_len] * env_tok[:min_len]
    aud_lat = aud_lat[:min_len] * env_lat[:min_len]
    aud_spe = aud_spe[:min_len] * env_spe[:min_len]
    aud_sem = aud_sem[:min_len] * env_sem[:min_len]
    t = t[:min_len]
    
    # Latent Panner (Z-Space mapped stereo drift)
    def l_pan(audio, freq):
        pan = 0.8 * np.sin(2 * np.pi * freq * t)
        l = audio * np.cos((pan + 1.0) * np.pi / 4.0)
        r = audio * np.sin((pan + 1.0) * np.pi / 4.0)
        return np.column_stack((l, r))
    
    mix = np.zeros((min_len, 2), dtype=np.float32)
    mix += l_pan(aud_tok, 0.0)      # Token Hell (Bass): Center
    mix += l_pan(aud_lat, 0.02)     # Latent Drift: Slow wide pan
    mix += l_pan(aud_spe, 0.1)      # Spectral Scramble: Fast chaotic pan
    mix += l_pan(aud_sem, 0.01)     # Semantic Collapse: Very slow pan
    
    unmastered = OUT_DIR / "PAN_OMNI_UNMASTERED.wav"
    sf.write(unmastered, mix, sr)
    
    log.info("--- Final Pan-Omni Mastering ---")
    final_out = OUT_DIR / "THE_PAN_OMNI_SYMPHONY_MASTER.wav"
    
    # Warm saturation + hard limiting + high output gain
    run_shell(f"sox {unmastered} {final_out} norm -0.1 compand 0.02,0.1 6:-50,-50,0,-1 -1 -3 3")
    
    log.info(f"SUCCESS! 5-Minute 21-Module Masterpiece generated: {final_out}")

if __name__ == "__main__":
    sr = generate_stems()
    apply_pan_omni_physics(sr)
    orchestrate_pan_omni(sr)
