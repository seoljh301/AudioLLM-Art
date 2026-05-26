
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from scipy.signal.windows import tukey

def create_defined_galaxy(input_paths, output_path, target_sr=48000):
    tracks = []
    min_len = float('inf')
    
    # Frequency slots (spread across 4 octaves)
    semitone_offsets = [-24, -12, -5, 0, 7, 12, 19, 24]
    # Fixed pan positions for each slot to reduce "chaos"
    # L <-------------------- Center --------------------> R
    slot_pans = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 0.4] 
    
    print(f"Phase 1: High-Precision Alignment...")
    
    for i, p in enumerate(input_paths):
        audio, sr = sf.read(str(p))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Pre-normalize each source for definition
        audio = librosa.util.normalize(audio)
        
        chroma = librosa.feature.chroma_stft(y=audio[::2], sr=target_sr) # faster analysis
        dom_pitch = np.argmax(np.mean(chroma, axis=1))
        target_shift = semitone_offsets[i] - dom_pitch
        
        print(f"  Slot {i+1}: {p.name} | Shift: {target_shift}st | Pan: {slot_pans[i]}")
        shifted = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=target_shift)
        
        tracks.append(shifted)
        min_len = min(min_len, len(shifted))

    tracks = [t[:min_len] for t in tracks]
    num_tracks = len(tracks)
    
    # 2. Refined Granular Parameters for "Al-gaeng-i" (Definition)
    grain_size_ms = 120  # Shorter for crispness
    hop_size_ms = 60    # 50% overlap for clear grain boundaries
    grain_size = int(grain_size_ms * target_sr / 1000)
    hop_size = int(hop_size_ms * target_sr / 1000)
    
    master_mix = np.zeros((min_len, 2), dtype=np.float32)
    # Use Tukey window for sharper attacks (less smearing than Hanning)
    window = tukey(grain_size, alpha=0.3)
    rng = np.random.default_rng(888)
    
    print(f"Phase 2: Defined Granular Interweaving...")
    
    for start in range(0, min_len - grain_size, hop_size):
        end = start + grain_size
        
        # Pick only 2 layers at a time to keep it from getting muddy/busy
        active_indices = rng.choice(num_tracks, size=2, replace=False)
        
        for idx in active_indices:
            # Time jitter (reduced to 15ms to keep rhythmic alignment)
            jitter = rng.integers(-int(0.015 * target_sr), int(0.015 * target_sr))
            t_start = max(0, min(min_len - grain_size, start + jitter))
            
            grain = tracks[idx][t_start:t_start + grain_size]
            
            # Use fixed pan for the slot
            pan = slot_pans[idx]
            l_gain = np.cos((pan + 1.0) * np.pi / 4.0)
            r_gain = np.sin((pan + 1.0) * np.pi / 4.0)
            
            master_mix[start:end, 0] += grain * window * l_gain * 0.45
            master_mix[start:end, 1] += grain * window * r_gain * 0.45

    print("Phase 3: Mastering for Definition...")
    # Lighter limiting to preserve transients
    peak = np.max(np.abs(master_mix))
    if peak > 0:
        master_mix = master_mix / peak * 0.95
    
    sf.write(output_path, master_mix, target_sr)
    print(f"DONE: Defined neural galaxy saved to {output_path}")

files = sorted(list(Path("runs/feedback_final5").glob("*.wav")))
create_defined_galaxy(files, "runs/masterpiece/defined_neural_galaxy.wav")
