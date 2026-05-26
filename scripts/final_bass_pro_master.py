import numpy as np
import soundfile as sf
import pyloudnorm as ln
from scipy.signal import butter, lfilter

def final_master(input_path, output_path, target_lufs=-12.0):
    print(f"Loading {input_path} for Final Bass-Heavy Pro Mastering...")
    audio, sr = sf.read(input_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)

    # 1. NaN/Inf cleanup (Numerical stability)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Sub-Bass Supplementation (The "Good Bass")
    # Using 2nd order crossover to isolate and boost the 'core' bass
    print("  Reinforcing Sub-Bass (+8dB @ 80Hz)...")
    b_low, a_low = butter(2, 80.0, fs=sr, btype='low')
    sub_bass = lfilter(b_low, a_low, audio, axis=0)
    
    # Apply boost
    audio_with_bass = audio + sub_bass * (10**(8/20) - 1)

    # 3. Harmonic Excitation (Clarity)
    print("  Applying Hi-Fi Excitation...")
    b_high, a_high = butter(2, 6000.0, fs=sr, btype='high')
    highs = lfilter(b_high, a_high, audio_with_bass, axis=0)
    excited = audio_with_bass + 0.08 * np.sign(highs) * (np.abs(highs) ** 0.85)

    # 4. Pro Loudness Normalization
    print(f"  Normalizing to {target_lufs} LUFS...")
    meter = ln.Meter(sr)
    loudness = meter.integrated_loudness(excited)
    normalized = ln.normalize.loudness(excited, loudness, target_lufs)

    # 5. Peak Safety (Hard Limiter)
    print("  Applying Peak Safety...")
    # Tanh drive for density
    final = np.tanh(normalized * 1.15) / np.tanh(1.15)
    
    # Final normalization
    peak = np.max(np.abs(final))
    if peak > 0:
        final = final / peak * 0.98
        
    sf.write(output_path, final.astype(np.float32), sr)
    print(f"SUCCESS: Final Bass-Heavy Pro-Mastered version saved to {output_path}")

if __name__ == "__main__":
    final_master("runs/masterpiece/neural_symphony_1.wav", "runs/masterpiece/final_symphony_bass_heavy.wav")
