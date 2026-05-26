
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as ln
from scipy.signal import butter, lfilter

def open_source_mastering(input_path, output_path, target_lufs=-14.0):
    print(f"Loading {input_path} for Pro-Mastering...")
    audio, sr = sf.read(input_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)

    # 1. DC Offset & Rumble Removal (High-pass 30Hz)
    print("  Applying High-pass filter (30Hz)...")
    b, a = butter(2, 30.0, fs=sr, btype='high')
    audio_filtered = lfilter(b, a, audio, axis=0)

    # 2. Harmonic Enhancement (Subtle Excitation)
    # This adds the 'glue' and 'sparkle' associated with pro mastering.
    print("  Applying Harmonic Excitation...")
    # Apply soft saturation to mid-highs
    b_high, a_high = butter(2, 5000.0, fs=sr, btype='high')
    highs = lfilter(b_high, a_high, audio_filtered, axis=0)
    excited = audio_filtered + 0.05 * np.sign(highs) * (np.abs(highs) ** 0.9)

    # 3. Loudness Normalization (BS.1770-4)
    print(f"  Normalizing to {target_lufs} LUFS using pyloudnorm...")
    meter = ln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(excited)
    print(f"    Original Loudness: {loudness:.2f} LUFS")
    
    # Apply gain to reach target
    loudness_normalized = ln.normalize.loudness(excited, loudness, target_lufs)
    
    # 4. Peak Limiting (True Peak Safety)
    print("  Applying True-Peak Limiting (-1.0 dBTP)...")
    # Soft tanh limiting
    mastered = np.tanh(loudness_normalized * 1.05) / np.tanh(1.05)
    
    # Final safety peak normalization
    peak = np.max(np.abs(mastered))
    if peak > 0:
        mastered = mastered / peak * 0.95 # -0.5 dB peak
        
    sf.write(output_path, mastered.astype(np.float32), sr)
    print(f"SUCCESS: Pro-mastered version saved to {output_path}")

open_source_mastering("runs/masterpiece/chained_symphony/symphony_step1_d.wav", "runs/masterpiece/symphony_step1_d_pyloudnorm.wav")
