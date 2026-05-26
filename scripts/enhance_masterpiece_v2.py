
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter

def enhance_audio_spectral_only(input_path, output_path, target_sr=48000):
    print(f"Loading {input_path} for spectral enhancement...")
    audio, sr = sf.read(input_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1) # force stereo
    
    # 1. Low-End Enhancement (Bass Boost)
    # Using a 2nd order low shelf at 100Hz
    print("  Applying Low-End Enhancement (+6dB @ 80Hz)...") # Slightly lower and stronger for impact
    b_low, a_low = butter(2, 80.0, fs=sr, btype='low')
    low_part = lfilter(b_low, a_low, audio, axis=0)
    # Boost by ~6dB
    audio_boosted = audio + low_part * (10**(6/20) - 1)
    
    # 2. Ultra-High Enhancement (Air Boost)
    # Using a 2nd order high shelf at 12kHz
    print("  Applying Ultra-High Enhancement (+8dB @ 12kHz)...") # Stronger sparkle
    b_high, a_high = butter(2, 12000.0, fs=sr, btype='high')
    high_part = lfilter(b_high, a_high, audio_boosted, axis=0)
    # Boost by ~8dB
    audio_boosted = audio_boosted + high_part * (10**(8/20) - 1)
    
    # 3. Final Glue & Limiter (No pulsing)
    print("  Final Mastering (Tanh Limiter)...")
    final = np.tanh(audio_boosted * 1.1) / np.tanh(1.1)
    
    # Normalize to -0.5dB peak
    peak = np.max(np.abs(final))
    if peak > 0:
        final = final / peak * 0.95
        
    sf.write(output_path, final.astype(np.float32), sr)
    print(f"SUCCESS: Spectral enhanced masterpiece saved to {output_path}")

enhance_audio_spectral_only("runs/masterpiece/neural_symphony_1.wav", "runs/masterpiece/spectral_enhanced_symphony_1.wav")
