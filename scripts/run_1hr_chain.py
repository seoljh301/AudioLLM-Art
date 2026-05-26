import subprocess
import os
import time

sequence = "addaada"
current_input = "data/Ulaanbaatar.wav"
out_dir = "runs/ulaanbaatar_pure_chain"
log_path = os.path.join(out_dir, "pipeline.log")

with open(log_path, "w") as f_log:
    f_log.write(f"Starting 1-hour chain at {time.ctime()}\n")
    f_log.flush()

    for i, stage in enumerate(sequence):
        step_name = f"step_{i+1}_{stage}"
        output = os.path.join(out_dir, f"{step_name}.wav")
        f_log.write(f"--- Stage {i+1}/{len(sequence)}: {stage.upper()} ---\n")
        f_log.flush()
        
        if stage == 'a':
            cmd = [
                "bash", "experiments/mvp_a_rave_latent/run.sh",
                "--config", "experiments/mvp_a_rave_latent/config_chain_boost.yaml",
                "--mode", "render", "--input", current_input, "--output", output
            ]
        elif stage == 'd':
            cmd = [
                "bash", "experiments/mvp_d_ckpt_morph/run.sh",
                "--config", "experiments/mvp_d_ckpt_morph/config_chain_boost.yaml",
                "--mode", "render", "--input", current_input, "--output", output
            ]
        
        start_t = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_t
        
        if result.returncode == 0:
            f_log.write(f"  SUCCESS in {elapsed:.2f}s\n")
            current_input = output
        else:
            f_log.write(f"  FAILED!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
            break
        f_log.flush()

    f_log.write(f"DONE. Final output: {current_input}\n")
