import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import time
import os

def create_1hr_masterpiece(input_paths, output_path, target_sr=48000):
    print(f"Preparing to layer {len(input_paths)} massive tracks...")
    
    # We will process in blocks of 30 seconds to avoid RAM OOM issues
    block_size_sec = 30.0
    block_size_samp = int(target_sr * block_size_sec)
    
    # 1. Analyze and find min length
    min_frames = float('inf')
    sf_objects = []
    
    # Open files and get info
    for p in input_paths:
        f = sf.SoundFile(str(p))
        sf_objects.append(f)
        min_frames = min(min_frames, f.frames)
        
    duration_s = min_frames / target_sr
    print(f"Total output duration will be: {duration_s/60:.2f} minutes")
    
    # Note: For memory efficiency, we are skipping the full-file Chroma pitch shift.
    # The pure chain (A-D-D-A-A-D-A) preserves the pitch structure relatively well.
    # Instead, we will focus on Dynamic Panning, Volume LFOs, and Soft Limiting.
    
    rng = np.random.default_rng(999)
    
    # Pre-calculate LFO parameters for each track
    # Freqs: 0.005Hz to 0.02Hz (Very slow changes for 1 hour duration)
    lfo_freqs = rng.uniform(0.005, 0.02, size=len(input_paths))
    lfo_phases = rng.uniform(0, 2 * np.pi, size=len(input_paths))
    
    # Panning params
    base_pans = np.linspace(-0.8, 0.8, len(input_paths))
    pan_drift_freqs = rng.uniform(0.01, 0.03, size=len(input_paths))
    
    # Create output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_f = sf.SoundFile(output_path, mode='w', samplerate=target_sr, channels=2, format='WAV', subtype='FLOAT')
    
    # 2. Block-by-block Processing
    current_frame = 0
    start_time = time.time()
    
    print("Starting block processing...")
    
    while current_frame < min_frames:
        read_frames = min(block_size_samp, min_frames - current_frame)
        master_block = np.zeros((read_frames, 2), dtype=np.float32)
        
        t_arr = np.linspace(current_frame / target_sr, (current_frame + read_frames) / target_sr, read_frames)
        
        for i, f in enumerate(sf_objects):
            # Read block
            data = f.read(read_frames)
            
            # Simple RMS normalization per block
            rms = np.sqrt(np.mean(data**2)) + 1e-6
            normalized = data / rms * 0.1 # Headroom for 7 tracks
            
            # Organic Breathing (Volume LFO)
            volume_env = 0.5 + 0.4 * np.sin(2 * np.pi * lfo_freqs[i] * t_arr + lfo_phases[i])
            
            # Drifting Panning
            pan_env = base_pans[i] + 0.15 * np.sin(2 * np.pi * pan_drift_freqs[i] * t_arr)
            pan_env = np.clip(pan_env, -1.0, 1.0)
            
            # Stereo gains
            left_gain = np.cos((pan_env + 1.0) * np.pi / 4.0)
            right_gain = np.sin((pan_env + 1.0) * np.pi / 4.0)
            
            processed = normalized * volume_env
            
            master_block[:, 0] += processed * left_gain
            master_block[:, 1] += processed * right_gain
            
        # Soft Tanh Limiter
        master_block = np.tanh(master_block * 1.2) / np.tanh(1.2)
        
        # Write to disk
        out_f.write(master_block)
        
        current_frame += read_frames
        
        # Progress output
        if (current_frame / min_frames) * 100 % 10 < (read_frames / min_frames) * 100:
            elapsed = time.time() - start_time
            print(f"  Progress: {current_frame / min_frames * 100:.1f}% ({current_frame/target_sr/60:.1f} min processed) - Elapsed: {elapsed:.1f}s")
            
    # Cleanup
    for f in sf_objects:
        f.close()
    out_f.close()
    
    print(f"SUCCESS: 1-Hour Neural Symphony saved to {output_path}")

if __name__ == "__main__":
    files = sorted(list(Path("runs/ulaanbaatar_pure_chain").glob("step_*.wav")))
    create_1hr_masterpiece(files, "runs/masterpiece/ulaanbaatar_neural_symphony.wav")
