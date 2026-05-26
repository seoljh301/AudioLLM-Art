
import numpy as np
import soundfile as sf
from src.core.texture_metrics import compute_texture_metrics

def analyze(path, label):
    audio, sr = sf.read(path)
    metrics = compute_texture_metrics(audio, sr)
    print(f"[{label}]")
    print(f"  Path: {path}")
    print(f"  RMS: {metrics.rms:.4f}")
    print(f"  Spectral Flatness: {metrics.spectral_flatness:.4f}")
    print(f"  Spectral Centroid (Hz): {metrics.spectral_centroid_hz:.1f}")
    print(f"  ZCR: {metrics.zero_crossing_rate:.4f}")
    print("-" * 30)

analyze("runs/complex_seed_48k.wav", "COMPLEX SEED")
analyze("runs/feedback_high/mvp_a_heavy.wav", "MVP-A HEAVY")
analyze("runs/feedback_high/mvp_c_heavy.wav", "MVP-C HEAVY")
