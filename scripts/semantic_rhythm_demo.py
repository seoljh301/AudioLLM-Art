import numpy as np
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from src.core.v2.llm_helmsman import LLMHelmsman, SteeringParams

def generate_semantic_rhythm(caption: str, dur_s: float = 10.0, base_bpm: float = 120.0, sr: int = 48000, out_path: str = "rhythm.wav"):
    """Generates an algorithmic drum loop controlled by LLM caption analysis."""
    
    # 1. Ask the Helmsman to interpret the caption
    helmsman = LLMHelmsman()
    steer = helmsman.analyze_caption(caption)
    
    # 2. Apply steering parameters to rhythm logic
    bpm = base_bpm * steer.bpm_multiplier
    bps = bpm / 60.0
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    
    # 16th note phase (for precise glitch/stutter timing)
    sixteenth_phase = (t * bps * 4) % 1.0
    beat_phase = (t * bps) % 1.0
    
    print(f"\n--- Generating Rhythm for '{caption}' ---")
    print(f"Target BPM: {bpm:.1f} (Multiplier: {steer.bpm_multiplier:.2f})")
    print(f"Kick Decay: {steer.kick_decay:.1f} | Glitch Density: {steer.glitch_density:.2f}")

    # --- KICK DRUM (Governed by kick_decay and filter_cutoff_bias) ---
    kick_env = np.exp(-steer.kick_decay * beat_phase)
    kick_pitch = 150 * np.exp(-30 * beat_phase) + 40
    kick = np.sin(2 * np.pi * kick_pitch * t) * kick_env
    
    # Add distortion if the scene is chaotic
    if steer.distortion_intensity > 0:
        kick = np.tanh(kick * (1.0 + steer.distortion_intensity * 5))
        
    # --- HI-HAT / GLITCH PERCUSSION (Governed by glitch_density) ---
    # Base hi-hat pattern (every 16th note)
    hat_env = np.exp(-40 * sixteenth_phase)
    
    # Introduce random ghost notes / silence based on glitch_density
    # We slice time into 16th note chunks and decide if they play, stutter, or mute
    chunk_samples = int(sr / (bps * 4))
    glitch_mask = np.ones_like(t)
    
    rng = np.random.default_rng(42) # fixed seed for reproducibility of this exact prompt
    for i in range(0, len(t), chunk_samples):
        if rng.random() < steer.glitch_density:
            # Glitch event!
            event_type = rng.random()
            if event_type < 0.4:
                # Silence
                glitch_mask[i:i+chunk_samples] = 0.0
            elif event_type < 0.8:
                # Stutter (32nd notes or 64th notes)
                stutter_rate = rng.choice([2, 4])
                stutter_phase = (t[i:i+chunk_samples] * bps * 4 * stutter_rate) % 1.0
                glitch_mask[i:i+chunk_samples] = np.exp(-20 * stutter_phase)
            else:
                # Reverse envelope (suck-in effect)
                glitch_mask[i:i+chunk_samples] = np.linspace(0.1, 1.5, min(chunk_samples, len(t)-i))
                
    noise = rng.standard_normal(len(t))
    # Filter hat based on brightness
    cutoff = 4000 if steer.filter_cutoff_bias <= 0 else 8000
    b_h, a_h = signal.butter(4, cutoff, fs=sr, btype='high')
    hats = signal.lfilter(b_h, a_h, noise * hat_env * glitch_mask).astype(np.float32)
    
    # Apply heavy distortion to hats if aggressive
    if steer.distortion_intensity > 0.3:
        hats = np.clip(hats * 5.0, -1.0, 1.0)
    
    # Mix
    mix = (kick + hats * 0.7).astype(np.float32)
    mix = np.clip(mix, -1.0, 1.0)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mix, sr)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    # Test 1: Calm Ambient
    generate_semantic_rhythm("A very calm ambient drone underwater", out_path="runs/semantic_rhythm/calm_beat.wav")
    
    # Test 2: Chaotic Industrial
    generate_semantic_rhythm("Aggressive screaming chaos with heavy glitch noise", out_path="runs/semantic_rhythm/chaos_beat.wav")
