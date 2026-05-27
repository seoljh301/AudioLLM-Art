import numpy as np
import soundfile as sf
import torch
import logging
import time
from pathlib import Path

# Import the architectural macros and tools
from scripts.multinet import load_all, net1, net2, net3, net_max, net_dynamic, RAVE_GUITAR, RAVE_ORGAN
from src.core.mix import soft_limiter, match_rms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("meta_symphony")

def generate_seed(dur=180.0, sr=48000):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    # 1. Deep Sub-Bass Foundation (30Hz, 40Hz, 55Hz)
    sub = 0.6 * np.sin(2 * np.pi * 32.7 * t) + 0.3 * np.sin(2 * np.pi * 41.2 * t) + 0.2 * np.sin(2 * np.pi * 55.0 * t)
    
    # 2. Slow FM modulation for neural texture grip
    fm_mod = np.sin(2 * np.pi * 0.1 * t)
    fm = np.sin(2 * np.pi * (55.0 + 5.0 * fm_mod) * t)
    
    sig = sub + 0.15 * fm
    
    # 3. Macro breathing LFO (30s period)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (1/30) * t) 
    sig = sig * lfo
    
    sig = (sig / np.max(np.abs(sig)) * 0.8).astype(np.float32)
    return sig, sr

def run_meta():
    out_dir = Path("runs/masterpiece/meta_symphony")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Generating 3-min sub-bass seed...")
    seed, sr = generate_seed()
    sf.write(out_dir / "seed_sub_180s.wav", seed, sr)
    
    rave, morph, codec = load_all()
    
    # =========================================================================
    # Phase 1: Generating the "Stems" via Macro-Networks
    # =========================================================================
    
    log.info("--- Phase 1.1: Net 1 (Crystal Cathedral) ---")
    stem_n1 = net1(seed, sr, rave, morph, codec)
    
    log.info("--- Phase 1.2: Net 3 (Decoding Chamber) ---")
    # Feed Net 1's output into Net 3 for compounded structural destruction
    in_n3 = match_rms(seed, stem_n1)
    stem_n3 = net3(in_n3, sr, rave, morph, codec)
    
    log.info("--- Phase 1.3: Net 2 (Recursive Organ) ---")
    # Use 2 passes for speed, feeding original seed
    stem_n2 = net2(seed, sr, rave, morph, codec, passes=2)
    
    log.info("--- Phase 1.4: Net Dynamic (Tempest) ---")
    # Feed Net 2's output into the time-varying Tempest
    # Note: net_dynamic expects exactly a 60s composition arc, but if we feed 180s,
    # the envelope will just hold its final value for the last 120s. 
    # To fix this elegantly without rewriting multinet, we will compress the 180s into 
    # the envelope function by scaling time, OR we just let the storm settle and drone for 2 mins.
    # We will let it settle into a massive drone.
    in_dyn = match_rms(seed, stem_n2)
    stem_dyn = net_dynamic(in_dyn, sr, rave, morph, codec, [RAVE_GUITAR, RAVE_ORGAN])
    
    # =========================================================================
    # Phase 2: Interweaving the Meta-Symphony
    # =========================================================================
    
    log.info("--- Phase 2: Interweaving Stems ---")
    
    # Ensure all stems are the exact same length before mixing
    min_len = min(len(stem_n1), len(stem_n3), len(stem_n2), len(stem_dyn), len(seed))
    stem_n1 = stem_n1[:min_len]
    stem_n3 = stem_n3[:min_len]
    stem_n2 = stem_n2[:min_len]
    stem_dyn = stem_dyn[:min_len]
    seed_trim = seed[:min_len]
    
    t = np.linspace(0, 180.0, min_len, dtype=np.float32)
    
    # Crossfading LFOs to create the "Intertwining" (얽힘) effect
    # Pair A: Spatial (Net 1) vs Destruction (Net 3) - 60s cycle
    lfo_A = 0.5 + 0.5 * np.sin(2 * np.pi * (1/60) * t) 
    
    # Pair B: Loops (Net 2) vs Tempest (Net Dynamic) - 45s cycle
    lfo_B = 0.5 + 0.5 * np.cos(2 * np.pi * (1/45) * t) 
    
    # Dynamic spatial panning
    pan_A = 0.7 * np.sin(2 * np.pi * (1/20) * t)
    pan_B = 0.7 * np.cos(2 * np.pi * (1/25) * t)
    
    def apply_pan(sig, pan):
        l = np.cos((pan + 1.0) * np.pi / 4.0)
        r = np.sin((pan + 1.0) * np.pi / 4.0)
        out = np.zeros((len(sig), 2), dtype=np.float32)
        out[:, 0] = sig * l
        out[:, 1] = sig * r
        return out

    # Mix the pairs
    mix_A = (lfo_A * stem_n1) + ((1 - lfo_A) * stem_n3)
    mix_B = (lfo_B * stem_n2) + ((1 - lfo_B) * stem_dyn)
    
    # Apply stereo drift
    stereo_A = apply_pan(mix_A, pan_A)
    stereo_B = apply_pan(mix_B, pan_B)
    
    # Sum the interwoven networks
    master = stereo_A + stereo_B
    
    # =========================================================================
    # Phase 3: Foundation Reinforcement & Mastering
    # =========================================================================
    
    log.info("--- Phase 3: Foundation Anchoring & Mastering ---")
    # Add the foundation sub-bass strictly to the center channel
    from scipy.signal import butter, lfilter
    b, a = butter(2, 100.0, fs=sr, btype='low')
    sub_only = lfilter(b, a, seed_trim).astype(np.float32) * (10**(8/20)) # +8dB sub
    master[:, 0] += sub_only
    master[:, 1] += sub_only
    
    # Final Soft Tanh Limiter
    master = np.tanh(master * 1.25) / np.tanh(1.25)
    peak = np.max(np.abs(master))
    if peak > 0:
        master = master / peak * 0.95
    
    sf.write(out_dir / "META_SYMPHONY_FINAL.wav", master, sr)
    log.info("META_SYMPHONY saved successfully!")

if __name__ == "__main__":
    run_meta()
