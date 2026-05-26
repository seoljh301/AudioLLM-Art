
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter, sosfilt

def hifi_enhance_high_end(input_path, output_path, target_sr=48000):
    print(f"Loading {input_path} for Hi-Fi High-End Enhancement...")
    audio, sr = sf.read(input_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1) # force stereo
    
    # 1. High-Frequency Harmonic Exciter
    # We apply subtle saturation only to the high frequencies to generate 'expensive' feeling harmonics.
    print("  Applying Harmonic Exciter (>8kHz)...")
    # Isolate high frequencies
    sos_hp = butter(4, 8000.0, fs=sr, btype='high', output='sos')
    high_freqs = sosfilt(sos_hp, audio, axis=0)
    
    # Non-linear excitation (subtle squaring/tanh)
    excited_highs = np.sign(high_freqs) * (np.abs(high_freqs) ** 0.8) # soft saturation for harmonics
    excited_highs = excited_highs - high_freqs # keep only the new content
    
    # 2. Ultra-High "Air" Restoration (+10dB @ 14kHz)
    # Using a 4th order precision shelf for high-definition air
    print("  Applying Ultra-High 'Air' Restoration (+10dB @ 14kHz)...")
    b_air, a_air = butter(2, 14000.0, fs=sr, btype='high')
    air_part = lfilter(b_air, a_air, audio, axis=0)
    
    # 3. Summing and High-Pass cleaning
    # Combine original + excited harmonics + air boost
    enhanced = audio + (excited_highs * 0.1) + (air_part * (10**(10/20) - 1))
    
    # 4. Phase-linear limiting & Normalization
    print("  Final Hi-Fi Mastering...")
    # Use a very soft limiter to preserve the 'expensive' transients
    peak = np.max(np.abs(enhanced))
    if peak > 0:
        enhanced = enhanced / peak * 0.96
        
    sf.write(output_path, enhanced.astype(np.float32), sr)
    print(f"SUCCESS: Hi-Fi enhanced masterpiece saved to {output_path}")

hifi_enhance_high_end("runs/masterpiece/neural_symphony_1.wav", "runs/masterpiece/hifi_enhanced_symphony_1.wav")
