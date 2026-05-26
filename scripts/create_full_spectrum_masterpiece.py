
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def create_full_spectrum_master(input_paths, output_path, target_sr=48000):
    tracks = []
    min_len = float('inf')
    
    # Harmonic spread in semitones:
    # 1: Sub (-2 oct), 2: Bass (-1 oct), 3: Low-Mid (-5 st), 4: Root (0), 
    # 5: High-Mid (+7 st), 6: High (+12 st), 7: Sparkle (+19 st), 8: Ultra (+24 st)
    semitone_offsets = [-24, -12, -5, 0, 7, 12, 19, 24]
    
    print(f"Phase 1: Harmonic Spreading of {len(input_paths)} sources...")
    
    for i, p in enumerate(input_paths):
        audio, sr = sf.read(str(p))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Detect current pitch to normalize it to Root first
        chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
        avg_chroma = np.mean(chroma, axis=1)
        dom_pitch = np.argmax(avg_chroma)
        
        # Calculate shift to reach the assigned harmonic slot
        target_shift = semitone_offsets[i] - dom_pitch
        
        print(f"  Slot {i+1}: {p.name} | Assigned Shift: {target_shift} semitones")
        
        # Use a high-quality pitch shifter
        shifted = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=target_shift)
        
        # RMS Normalization (balanced energy across spectrum)
        rms = np.sqrt(np.mean(shifted**2))
        normalized = shifted / (rms + 1e-6)
        
        tracks.append(normalized)
        min_len = min(min_len, len(normalized))

    tracks = [t[:min_len] for t in tracks]
    num_tracks = len(tracks)
    
    # Granular Parameters
    grain_size_ms = 300  # Slightly longer grains for better pitch perception
    hop_size_ms = 100
    grain_size = int(grain_size_ms * target_sr / 1000)
    hop_size = int(hop_size_ms * target_sr / 1000)
    
    master_mix = np.zeros((min_len, 2), dtype=np.float32)
    window = np.hanning(grain_size)
    rng = np.random.default_rng(777)
    
    print(f"Phase 2: Full-Spectrum Granular Synthesis...")
    
    for start in range(0, min_len - grain_size, hop_size):
        end = start + grain_size
        
        # Pick 4 random sources for a denser spectral profile
        active_indices = rng.choice(num_tracks, size=4, replace=False)
        
        for idx in active_indices:
            # Subtle time drift
            jitter = rng.integers(-int(0.04 * target_sr), int(0.04 * target_sr))
            t_start = max(0, min(min_len - grain_size, start + jitter))
            
            grain = tracks[idx][t_start:t_start + grain_size]
            
            # Wide stereo spreading
            pan = rng.uniform(0.0, 1.0)
            l_gain = np.sqrt(1.0 - pan)
            r_gain = np.sqrt(pan)
            
            # Layer into master
            master_mix[start:end, 0] += grain * window * l_gain * 0.25
            master_mix[start:end, 1] += grain * window * r_gain * 0.25

    print("Phase 3: Final Spectral Mastering...")
    # Soft tanh saturation
    master_mix = np.tanh(master_mix * 1.3) / np.tanh(1.3)
    
    sf.write(output_path, master_mix, target_sr)
    print(f"COMPLETE: Full-spectrum masterpiece saved to {output_path}")

files = sorted(list(Path("runs/feedback_final5").glob("*.wav")))
create_full_spectrum_master(files, "runs/masterpiece/full_spectrum_neural_galaxy.wav")
