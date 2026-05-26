
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def normalize_and_layer(input_paths, output_path, target_sr=48000):
    tracks = []
    min_len = float('inf')
    
    print(f"Loading and analyzing {len(input_paths)} tracks...")
    
    for i, p in enumerate(input_paths):
        audio, sr = sf.read(str(p))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # 1. Pitch-wise alignment (Normalization)
        # We detect the dominant pitch and shift it to the average or a target
        # Since these are complex textures, we'll use chroma-based shift 
        # to ensure they all 'feel' like they are in the same key.
        chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
        avg_chroma = np.mean(chroma, axis=1)
        dom_pitch = np.argmax(avg_chroma)
        
        # Shift to 'A' (root 0 in chroma)
        shift_steps = (0 - dom_pitch) % 12
        if shift_steps > 6: shift_steps -= 12 # find shortest path
        
        print(f"  Track {i+1}: {p.name} | Dom Pitch: {dom_pitch} | Shifting: {shift_steps} steps")
        # Shift audio (using pitch_shift for better quality)
        shifted = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=shift_steps)
        
        # 2. Level Normalization (RMS)
        rms = np.sqrt(np.mean(shifted**2))
        normalized = shifted / (rms + 1e-6) * 0.15 # lower gain per track to avoid clipping
        
        tracks.append(normalized)
        min_len = min(min_len, len(normalized))

    # Crop all to same length
    tracks = [t[:min_len] for t in tracks]
    
    # 3. Stereo Spatialization (Panning)
    # L <-------- Center --------> R
    # Create a stereo mix
    master_mix = np.zeros((min_len, 2), dtype=np.float32)
    
    for i, t in enumerate(tracks):
        # Pan positions from -0.8 (Left) to 0.8 (Right)
        pan = (i / (len(tracks) - 1)) * 1.6 - 0.8
        
        # Constant power panning
        left_gain = np.cos((pan + 1.0) * np.pi / 4.0)
        right_gain = np.sin((pan + 1.0) * np.pi / 4.0)
        
        master_mix[:, 0] += t * left_gain
        master_mix[:, 1] += t * right_gain

    # 4. Master Limiter (Tanh)
    print("Applying master limiter and saving...")
    master_mix = np.tanh(master_mix * 1.1) / np.tanh(1.1)
    
    sf.write(output_path, master_mix, target_sr)
    print(f"Masterpiece saved to: {output_path}")

files = sorted(list(Path("runs/feedback_final5").glob("*.wav")))
normalize_and_layer(files, "runs/masterpiece/neural_symphony_1.wav")
