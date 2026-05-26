
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter
from scipy.signal.windows import tukey

def enhance_audio(input_path, output_path, target_sr=48000):
    print(f"Loading {input_path} for enhancement...")
    audio, sr = sf.read(input_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1) # force stereo
    
    # 1. Low-End Enhancement (Bass Boost)
    # Using a 2nd order low shelf at 100Hz
    print("  Applying Low-End Enhancement (+4dB @ 100Hz)...")
    b_low, a_low = butter(2, 100.0, fs=sr, btype='low')
    low_part = lfilter(b_low, a_low, audio, axis=0)
    # Boost by ~4dB
    audio_boosted = audio + low_part * (10**(4/20) - 1)
    
    # 2. Ultra-High Enhancement (Air Boost)
    # Using a 2nd order high shelf at 12kHz
    print("  Applying Ultra-High Enhancement (+6dB @ 12kHz)...")
    b_high, a_high = butter(2, 12000.0, fs=sr, btype='high')
    high_part = lfilter(b_high, a_high, audio_boosted, axis=0)
    # Boost by ~6dB
    audio_boosted = audio_boosted + high_part * (10**(6/20) - 1)
    
    # 3. Tukey Window Rhythmic Shaper
    # This adds the "Al-gaeng-i" (grain definition) by pulsing the mix
    print("  Applying Rhythmic Tukey Shaper (250ms pulse)...")
    pulse_ms = 250
    pulse_samp = int(pulse_ms * sr / 1000)
    window = tukey(pulse_samp, alpha=0.5) # smooth but defined
    
    # Apply the window rhythmically
    shapen = np.zeros_like(audio_boosted)
    for i in range(0, len(audio_boosted) - pulse_samp, pulse_samp):
        # We multiply by the window to create a pulsing "grain" feel
        shapen[i:i+pulse_samp] = audio_boosted[i:i+pulse_samp] * window[:, np.newaxis]
    
    # 4. Final Glue & Limiter
    # Blend some of the original back to avoid 100% silence between pulses if needed
    # But user asked for Tukey window 적용, so we'll lean into the pulse.
    # Mix 70% pulsed, 30% static for a "Defined but Continuous" feel
    final = 0.7 * shapen + 0.3 * audio_boosted
    
    print("  Final Mastering (Tanh Limiter)...")
    final = np.tanh(final * 1.1) / np.tanh(1.1)
    
    # Normalize to -0.5dB peak
    peak = np.max(np.abs(final))
    if peak > 0:
        final = final / peak * 0.94
        
    sf.write(output_path, final.astype(np.float32), sr)
    print(f"SUCCESS: Enhanced masterpiece saved to {output_path}")

enhance_audio("runs/masterpiece/neural_symphony_1.wav", "runs/masterpiece/enhanced_symphony_1.wav")
