
import numpy as np
import soundfile as sf
import librosa

def aggressive_mastering(input_path, output_path, target_sr=48000):
    print(f"Loading {input_path} for aggressive mastering...")
    audio, sr = sf.read(input_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
        
    # 1. Soft Glue Compression (RMS based)
    print("  Applying Heavy Glue Compression...")
    # Calculate RMS envelope
    frame_len = 1024
    hop_len = 256
    # Librosa expects (n_samples,) for envelope
    rms = librosa.feature.rms(y=audio[:, 0], frame_length=frame_len, hop_length=hop_len)[0]
    # Interpolate to original length
    rms_env = np.interp(np.arange(len(audio)), np.arange(len(rms)) * hop_len, rms)
    
    # Simple compression: gain = 1 / (rms + threshold)
    threshold = 0.1
    ratio = 4.0
    # Gain reduction logic
    gain = np.ones_like(rms_env)
    mask = rms_env > threshold
    gain[mask] = (rms_env[mask] / threshold) ** (1/ratio - 1)
    
    compressed = audio * gain[:, np.newaxis]
    
    # 2. Harmonic Saturation (Loudness Excitation)
    print("  Applying Harmonic Saturation for Perceived Loudness...")
    # Tanh with drive to bring up low-level details
    drive = 1.8 
    saturated = np.tanh(compressed * drive)
    
    # 3. Peak Normalization & Final Limiter
    print("  Maximizing Output...")
    # Final push
    final = saturated * 1.2
    # Hard clip safety
    final = np.clip(final, -0.99, 0.99)
    
    sf.write(output_path, final.astype(np.float32), sr)
    print(f"SUCCESS: Aggressive mastered version saved to {output_path}")

aggressive_mastering("runs/masterpiece/chained_symphony/symphony_step1_d.wav", "runs/masterpiece/symphony_step1_d_loud.wav")
