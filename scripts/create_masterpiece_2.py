
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def create_masterpiece_2(input_paths, output_path, target_sr=48000):
    tracks = []
    min_len = float('inf')
    
    print(f"Phase 1: Harmonic Alignment & Analysis...")
    
    for i, p in enumerate(input_paths):
        audio, sr = sf.read(str(p))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Pitch Normalization to 'A'
        chroma = librosa.feature.chroma_stft(y=audio[::2], sr=target_sr)
        dom_pitch = np.argmax(np.mean(chroma, axis=1))
        shift_steps = (0 - dom_pitch) % 12
        if shift_steps > 6: shift_steps -= 12
        
        print(f"  Track {i+1}: {p.name} | Pitch: {dom_pitch} -> Root | Normalizing...")
        shifted = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=shift_steps)
        
        # RMS base normalization
        rms = np.sqrt(np.mean(shifted**2))
        tracks.append(shifted / (rms + 1e-6))
        min_len = min(min_len, len(shifted))

    tracks = [t[:min_len] for t in tracks]
    duration_s = min_len / target_sr
    t_arr = np.linspace(0, duration_s, min_len)
    
    # Create Stereo Master
    master_mix = np.zeros((min_len, 2), dtype=np.float32)
    
    print("Phase 2: Dynamic Ensemble Rendering (LFOs & Drifting)...")
    
    rng = np.random.default_rng(123)
    
    for i, t in enumerate(tracks):
        # 1. Organic Breathing (Volume LFO)
        # Each track gets a unique slow oscillation (0.01Hz to 0.05Hz)
        lfo_freq = rng.uniform(0.01, 0.04)
        lfo_phase = rng.uniform(0, 2 * np.pi)
        # Depth 0.3 to 0.7 (never fully silent)
        volume_env = 0.5 + 0.3 * np.sin(2 * np.pi * lfo_freq * t_arr + lfo_phase)
        
        # 2. Drifting Panning
        # Base pan spread like symphony_1, but with slow drift
        base_pan = (i / (len(tracks) - 1)) * 1.6 - 0.8
        pan_drift_freq = rng.uniform(0.02, 0.08)
        pan_env = base_pan + 0.1 * np.sin(2 * np.pi * pan_drift_freq * t_arr)
        pan_env = np.clip(pan_env, -1.0, 1.0)
        
        # Stereo gains
        left_gain = np.cos((pan_env + 1.0) * np.pi / 4.0)
        right_gain = np.sin((pan_env + 1.0) * np.pi / 4.0)
        
        # 3. Frequency-Selective Ducking (Simple)
        # Apply a bit more gain to the 'ghost' layers (E, D) and less to 'A'
        # based on track index or name
        track_multiplier = 0.18 # average
        if "mvp_e" in input_paths[i].name: track_multiplier = 0.22 # make granular more visible
        if "step_7" in input_paths[i].name: track_multiplier = 0.14 # step 7 is heavy, keep it subtle
        
        processed = t * volume_env * track_multiplier
        
        master_mix[:, 0] += processed * left_gain
        master_mix[:, 1] += processed * right_gain

    # 4. Master Polish
    print("Phase 3: Glue Compression & Final Polish...")
    # Soft Tanh with "Drive"
    master_mix = np.tanh(master_mix * 1.2) / np.tanh(1.2)
    
    # Final normalization to safety peak
    peak = np.max(np.abs(master_mix))
    master_mix = master_mix / peak * 0.98
    
    sf.write(output_path, master_mix, target_sr)
    print(f"Masterpiece 2 saved: {output_path}")

files = sorted(list(Path("runs/feedback_final5").glob("*.wav")))
create_masterpiece_2(files, "runs/masterpiece/masterpiece_2.wav")
