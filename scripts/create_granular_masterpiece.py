
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def multi_source_granular_master(input_paths, output_path, target_sr=48000):
    tracks = []
    min_len = float('inf')
    
    print(f"Phase 1: Analyzing and Aligning {len(input_paths)} sources...")
    
    for i, p in enumerate(input_paths):
        audio, sr = sf.read(str(p))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Dominant pitch alignment to 'A'
        chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
        avg_chroma = np.mean(chroma, axis=1)
        dom_pitch = np.argmax(avg_chroma)
        shift_steps = (0 - dom_pitch) % 12
        if shift_steps > 6: shift_steps -= 12
        
        print(f"  Track {i+1}: {p.name} (Pitch: {dom_pitch}, Shifting {shift_steps})")
        shifted = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=shift_steps)
        
        # RMS Normalization
        rms = np.sqrt(np.mean(shifted**2))
        normalized = shifted / (rms + 1e-6)
        
        tracks.append(normalized)
        min_len = min(min_len, len(normalized))

    tracks = [t[:min_len] for t in tracks]
    num_tracks = len(tracks)
    
    # Granular Parameters
    grain_size_ms = 250
    hop_size_ms = 80
    grain_size = int(grain_size_ms * target_sr / 1000)
    hop_size = int(hop_size_ms * target_sr / 1000)
    
    master_mix = np.zeros((min_len, 2), dtype=np.float32)
    window = np.hanning(grain_size)
    
    rng = np.random.default_rng(42)
    
    print(f"Phase 2: Generating Multi-Source Granular Interweaving...")
    
    # Progress tracking
    total_steps = (min_len - grain_size) // hop_size
    
    for step, start in enumerate(range(0, min_len - grain_size, hop_size)):
        if step % 500 == 0:
            print(f"  Progress: {100 * step / total_steps:.1f}%")
            
        end = start + grain_size
        
        # Pick 3 random sources for this specific grain position
        active_indices = rng.choice(num_tracks, size=3, replace=False)
        
        for idx in active_indices:
            # Time jitter: drift slightly within +/- 50ms of the current position
            jitter = rng.integers(-int(0.05 * target_sr), int(0.05 * target_sr))
            t_start = max(0, min(min_len - grain_size, start + jitter))
            
            grain = tracks[idx][t_start:t_start + grain_size]
            
            # Randomized Panning for this grain
            # This creates a "sparkling" stereo field
            pan = rng.uniform(0.1, 0.9)
            l_gain = np.sqrt(1.0 - pan)
            r_gain = np.sqrt(pan)
            
            # Apply grain to master with windowing
            # Gain factor 0.3 to account for 3 overlapping sources + hop overlap
            master_mix[start:end, 0] += grain * window * l_gain * 0.3
            master_mix[start:end, 1] += grain * window * r_gain * 0.3

    # Phase 3: Final Polishing
    print("Phase 3: Final Mastering (Limiter)...")
    # Soft tanh saturation for a warm, dense feel
    master_mix = np.tanh(master_mix * 1.2) / np.tanh(1.2)
    
    sf.write(output_path, master_mix, target_sr)
    print(f"SUCCESS: Granular Masterpiece saved to {output_path}")

files = sorted(list(Path("runs/feedback_final5").glob("*.wav")))
multi_source_granular_master(files, "runs/masterpiece/neural_granular_symphony.wav")
