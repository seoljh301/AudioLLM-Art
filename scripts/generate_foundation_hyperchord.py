import numpy as np
import soundfile as sf
import librosa
import os

def generate_script():
    orig_path = "data/Ulaanbaatar.wav"
    target_sr = 48000
    
    print("Analyzing pitch of the Foundation track...")
    audio, sr = sf.read(orig_path, frames=target_sr*30) 
    if sr != target_sr:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
    avg_chroma = np.mean(chroma, axis=1)
    dom_pitch = np.argmax(avg_chroma)
    
    base_shift = (0 - dom_pitch) % 12
    if base_shift > 6: base_shift -= 12
    print(f"  Dom Pitch = {dom_pitch}, Base Shift to Root = {base_shift} semitones")
    
    octaves = range(-4, 5) # -4, -3, -2, -1, 0, 1, 2, 3, 4
    
    out_dir = "runs/ulaanbaatar_master"
    os.makedirs(out_dir, exist_ok=True)
    final_out = f"{out_dir}/foundation_hyperchord.wav"
    
    script_content = "#!/bin/bash\nset -euo pipefail\n\n"
    script_content += "echo 'Starting Foundation Hyper-Chord Generation (-4 to +4 Octaves)...'\n\n"
    
    processed_files = []
    
    for oct_shift in octaves:
        oct_label = f"m{abs(oct_shift)}" if oct_shift < 0 else f"{oct_shift}"
        tmp_out = f"{out_dir}/tmp_found_oct_{oct_label}.wav"
        
        total_shift_semitones = base_shift + (oct_shift * 12)
        cents = total_shift_semitones * 100
        
        vol = 0.2 / (abs(oct_shift) + 1.0)
        
        script_content += f"echo 'Processing Octave {oct_shift} (Pitch shift: {cents} cents, Vol: {vol:.3f})...'\n"
        
        if cents != 0:
            # SoX 'pitch' fails if the shift is too extreme (factor < 0.1 or > 100).
            # We break large shifts into smaller 2400-cent chunks.
            pitch_args = ""
            remaining_cents = cents
            while abs(remaining_cents) > 2400:
                chunk = 2400 if remaining_cents > 0 else -2400
                pitch_args += f" pitch {chunk}"
                remaining_cents -= chunk
            if remaining_cents != 0:
                pitch_args += f" pitch {remaining_cents}"
                
            script_content += f"sox {orig_path} {tmp_out} rate -v 48000 vol {vol:.3f} {pitch_args}\n"
        else:
            script_content += f"sox {orig_path} {tmp_out} rate -v 48000 vol {vol:.3f}\n"
        
        processed_files.append(tmp_out)
        
    script_content += "\necho 'Summing all 9 octave layers...'\n"
    script_content += f"sox -m {' '.join(processed_files)} {final_out}\n"
    
    script_content += "\necho 'Applying final limiter...'\n"
    final_limited = f"{out_dir}/foundation_hyperchord_limited.wav"
    script_content += f"sox {final_out} {final_limited} compand 0.01,0.1 -60,-60,0,-0.5 -1 -6\n"
    
    script_content += "\necho 'Cleaning up temporary files...'\n"
    script_content += f"rm -f {' '.join(processed_files)} {final_out}\n"
    script_content += f"\necho 'SUCCESS: Foundation Hyper-Chord saved to {final_limited}'\n"
    
    script_path = "scripts/run_foundation_hyperchord.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
        
    os.chmod(script_path, 0o755)
    print(f"\nGenerated SoX script at {script_path}")

if __name__ == "__main__":
    generate_script()
