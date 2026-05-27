import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter
from pathlib import Path
import gc
import os

def create_hfo_layered_master(input_paths, output_path, target_sr=48000):
    print("Initializing Sequential HFO Layering for Massive Files...", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Foundation: Load original and extract Low-End
    print(f"Loading Foundation: data/Ulaanbaatar.wav", flush=True)
    orig_audio, orig_sr = sf.read("data/Ulaanbaatar.wav")
    if orig_audio.ndim == 1:
        orig_audio = np.stack([orig_audio, orig_audio], axis=1)
        
    print(f"  Filtering Low-End (< 300Hz) at {orig_sr}Hz...", flush=True)
    b_low, a_low = butter(2, 300.0, fs=orig_sr, btype='low')
    foundation = lfilter(b_low, a_low, orig_audio, axis=0)
    
    # Free original audio to save RAM
    del orig_audio
    gc.collect()
    
    if orig_sr != target_sr:
        print(f"  Resampling Foundation to {target_sr}Hz...", flush=True)
        # Resample channel by channel to save memory
        f_resampled = np.zeros((int(len(foundation) / orig_sr * target_sr) + 1000, 2), dtype=np.float32)
        f_resampled[:, 0] = librosa.resample(foundation[:, 0], orig_sr=orig_sr, target_sr=target_sr, fix=False)
        f_resampled[:, 1] = librosa.resample(foundation[:, 1], orig_sr=orig_sr, target_sr=target_sr, fix=False)
        foundation = f_resampled
    
    foundation = foundation * (10**(6/20)) # +6dB boost
    min_len = len(foundation)
    
    # Initialize master mix
    master_mix = foundation
    del foundation
    gc.collect()

    print("Master Mix initialized.", flush=True)

    # 2. Add Neural Tracks sequentially
    b_high, a_high = butter(2, 800.0, fs=target_sr, btype='high')
    num_tracks = len(input_paths)
    
    for i, p in enumerate(input_paths):
        print(f"Processing Track {i+1}/{num_tracks}: {p.name}", flush=True)
        t_audio, t_sr = sf.read(str(p))
        if t_audio.ndim == 1:
            t_audio = np.stack([t_audio, t_audio], axis=1)
            
        if t_sr != target_sr:
            print(f"  Resampling {p.name}...", flush=True)
            t_resampled = np.zeros((int(len(t_audio) / t_sr * target_sr) + 1000, 2), dtype=np.float32)
            t_resampled[:, 0] = librosa.resample(t_audio[:, 0], orig_sr=t_sr, target_sr=target_sr, fix=False)
            t_resampled[:, 1] = librosa.resample(t_audio[:, 1], orig_sr=t_sr, target_sr=target_sr, fix=False)
            t_audio = t_resampled
            
        min_len = min(min_len, len(t_audio))
        
        print(f"  Filtering HFO (> 800Hz)...", flush=True)
        hfo_layer = lfilter(b_high, a_high, t_audio[:min_len], axis=0)
        del t_audio
        gc.collect()
        
        print(f"  Panning and Mixing...", flush=True)
        pan = (i / (max(1, num_tracks - 1))) * 1.6 - 0.8
        l_gain = np.cos((pan + 1.0) * np.pi / 4.0)
        r_gain = np.sin((pan + 1.0) * np.pi / 4.0)
        
        master_mix[:min_len, 0] += hfo_layer[:, 0] * l_gain * 0.25
        master_mix[:min_len, 1] += hfo_layer[:, 1] * r_gain * 0.25
        
        del hfo_layer
        gc.collect()

    master_mix = master_mix[:min_len]

    # 3. Final Mastering
    print("Applying Master Limiter...", flush=True)
    master_mix = np.tanh(master_mix * 1.1) / np.tanh(1.1)
    
    peak = np.max(np.abs(master_mix))
    if peak > 0:
        master_mix = master_mix / peak * 0.98

    print("Writing to disk...", flush=True)
    sf.write(output_path, master_mix.astype(np.float32), target_sr)
    print(f"SUCCESS: HFO Layered Master saved to {output_path}", flush=True)

if __name__ == "__main__":
    # We layer the 7 pure chain tracks 
    files = sorted(list(Path("runs/ulaanbaatar_pure_chain").glob("step_*.wav")))
    if len(files) > 0:
        create_hfo_layered_master(files, "runs/masterpiece/ulaanbaatar_hfo_layered.wav")
    else:
        print("Error: No step files found to layer.", flush=True)
