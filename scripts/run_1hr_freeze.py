import subprocess
import os
import time

input_path = "data/Ulaanbaatar.wav"
out_dir = "runs/ulaanbaatar_freeze"
output_path = os.path.join(out_dir, "ulaanbaatar_frozen.wav")
log_path = os.path.join(out_dir, "freeze.log")

with open(log_path, "w") as f_log:
    f_log.write(f"Starting 1-hour freeze at {time.ctime()}\n")
    f_log.flush()

    cmd = [
        "bash", "experiments/mvp_f_neural_freeze/run.sh",
        "--mode", "render", "--input", input_path, "--output", output_path
    ]
    
    start_t = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_t
    
    if result.returncode == 0:
        f_log.write(f"SUCCESS in {elapsed:.2f}s\n")
    else:
        f_log.write(f"FAILED!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
    f_log.flush()

    f_log.write(f"DONE. Output: {output_path}\n")
