import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import os

def analyze_pitch(path, target_sr=48000):
    audio, sr = sf.read(str(path), frames=target_sr*30) 
    if sr != target_sr:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1) # to mono for analysis
        
    chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
    avg_chroma = np.mean(chroma, axis=1)
    dom_pitch = np.argmax(avg_chroma)
    
    # Calculate base shift to 'A' (pitch class 0)
    base_shift = (0 - dom_pitch) % 12
    if base_shift > 6: base_shift -= 12
    return dom_pitch, base_shift

def generate_f0_octave_sox_script():
    input_paths = sorted(list(Path("runs/ulaanbaatar_pure_chain").glob("step_*.wav")))
    orig_path = "data/Ulaanbaatar.wav"
    
    if not input_paths:
        print("No input files found.")
        return

    print("Analyzing first 30 seconds of each track to determine pitch shifts...")
    base_shifts = []
    
    # 1. Analyze Original
    orig_pitch, orig_shift = analyze_pitch(orig_path, 44100)
    print(f"  {orig_path}: Dom Pitch = {orig_pitch}, Base Shift = {orig_shift} semitones")
    
    # 2. Analyze Neural Tracks
    for p in input_paths:
        p_pitch, p_shift = analyze_pitch(p)
        base_shifts.append(p_shift)
        print(f"  {p.name}: Dom Pitch = {p_pitch}, Base Shift = {p_shift} semitones")

    # Octave spreading strategy (in octaves)
    # 8 tracks total: Foundation + 7 Neural Steps
    # Let's spread them widely to create a massive wall of sound
    # Foundation: -2 octaves (Sub)
    # Step 1: -1 octave (Bass/Low-Mid)
    # Step 2: 0 octave (Mid)
    # Step 3: +1 octave (High-Mid)
    # Step 4: +2 octaves (High/Sparkle)
    # Step 5: -1 octave (Thicken Low-Mid)
    # Step 6: 0 octave (Thicken Mid)
    # Step 7: +1 octave (Thicken High-Mid)
    octave_spreads = [-2, -1, 0, 1, 2, -1, 0, 1]
    
    # Combine base shift and octave spread
    final_shifts = [orig_shift + octave_spreads[0] * 12]
    for bs, oct_s in zip(base_shifts, octave_spreads[1:]):
        final_shifts.append(bs + oct_s * 12)

    # Generate SoX command
    out_dir = "runs/ulaanbaatar_master"
    os.makedirs(out_dir, exist_ok=True)
    final_out = f"{out_dir}/f0_octave_symphony.wav"
    
    script_content = "#!/bin/bash\nset -euo pipefail\n\n"
    script_content += "echo 'Starting SoX f0-octave-spread mixing...'\n\n"
    
    processed_files = []
    
    # Process Foundation
    found_out = f"{out_dir}/tmp_foundation_oct.wav"
    cents = final_shifts[0] * 100
    script_content += f"echo 'Processing Foundation (Pitch shift: {cents} cents)...'\n"
    # Sinc -300 isolates lows. We boost it vol 1.5 since it's the anchor.
    script_content += f"sox {orig_path} {found_out} rate -v 48000 sinc -300 vol 1.5 pitch {cents}\n"
    processed_files.append(found_out)
    
    # Process Layers
    for i, (p, shift) in enumerate(zip(input_paths, final_shifts[1:])):
        tmp_out = f"{out_dir}/tmp_track_oct_{i}.wav"
        cents = shift * 100
        script_content += f"echo 'Processing neural track {i+1} (Pitch shift: {cents} cents)...'\n"
        
        # Adjust volume based on octave to prevent high-freq harshness and low-freq mud
        vol = 0.15
        if octave_spreads[i+1] >= 1:
            vol = 0.10  # High pitched tracks get slightly less volume
        elif octave_spreads[i+1] < 0:
            vol = 0.20  # Lower pitched tracks get slightly more volume
            
        if cents != 0:
            script_content += f"sox {str(p)} {tmp_out} vol {vol} pitch {cents}\n"
        else:
            script_content += f"sox {str(p)} {tmp_out} vol {vol}\n"
        processed_files.append(tmp_out)
        
    script_content += "\necho 'Summing all octave-aligned tracks...'\n"
    script_content += f"sox -m {' '.join(processed_files)} {final_out}\n"
    
    script_content += f"\necho 'Applying final limiter...'\n"
    final_limited = f"{out_dir}/f0_octave_symphony_limited.wav"
    script_content += f"sox {final_out} {final_limited} compand 0.01,0.1 -60,-60,0,-0.5 -1 -6\n"
    
    script_content += f"\necho 'Cleaning up temporary files...'\n"
    script_content += f"rm -f {' '.join(processed_files)} {final_out}\n"
    script_content += f"\necho 'SUCCESS: F0-Octave Master saved to {final_limited}'\n"
    
    script_path = "scripts/run_f0_octave_sox_master.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
        
    os.chmod(script_path, 0o755)
    print(f"\nGenerated SoX script at {script_path}")

if __name__ == "__main__":
    generate_f0_octave_sox_script()
