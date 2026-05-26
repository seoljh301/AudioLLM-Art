
import numpy as np
import soundfile as sf
import librosa
from src.core.texture_metrics import compute_texture_metrics

def sampled_analysis(path, interval_sec=600):
    info = sf.info(path)
    sr = info.samplerate
    duration = info.duration
    
    print(f"Analyzing {path}...")
    print(f"Total Duration: {duration/60:.2f} minutes")
    print("-" * 50)
    print(f"{'Time (min)':<12} | {'RMS':<8} | {'Flatness':<10} | {'Centroid (Hz)':<12}")
    print("-" * 50)
    
    # Analyze 2-second chunks at each interval
    chunk_dur = 2.0
    for t in range(0, int(duration), interval_sec):
        # Read a small segment
        start_frame = int(t * sr)
        n_frames = int(chunk_dur * sr)
        
        # Read audio (handle multi-channel)
        audio, _ = sf.read(path, start=start_frame, frames=n_frames)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # to mono for metrics
            
        metrics = compute_texture_metrics(audio, sr)
        
        time_str = f"{t/60:02.0f}:00"
        print(f"{time_str:<12} | {metrics.rms:<8.4f} | {metrics.spectral_flatness:<10.4f} | {metrics.spectral_centroid_hz:<12.1f}")

if __name__ == "__main__":
    sampled_analysis("data/Ulaanbaatar.wav")
