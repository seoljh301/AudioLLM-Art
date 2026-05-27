import subprocess
import os
import time

def run_task(name, cmd, log_path):
    with open(log_path, "a") as f_log:
        f_log.write(f"\n\n--- [RESUME] Starting {name} at {time.ctime()} ---\n")
        f_log.flush()
        start_t = time.time()
        try:
            subprocess.run(cmd, stdout=f_log, stderr=f_log, check=True)
            elapsed = time.time() - start_t
            f_log.write(f"  {name} SUCCESS in {elapsed:.2f}s\n")
        except subprocess.CalledProcessError as e:
            f_log.write(f"  {name} FAILED with exit code {e.returncode}\n")
        f_log.flush()

def run_chain_resume():
    sequence = "da" # We are at step 6, so remaining sequence is Step 6 (d), Step 7 (a)
    current_input = "runs/ulaanbaatar_pure_chain/step_5_a.wav"
    out_dir = "runs/ulaanbaatar_pure_chain"
    log_path = os.path.join(out_dir, "pipeline.log")
    
    if not os.path.exists(current_input):
        with open(log_path, "a") as f:
            f.write(f"ERROR: Input file {current_input} not found!\n")
        return

    for i, stage in enumerate(sequence):
        real_step = i + 6
        step_name = f"step_{real_step}_{stage}"
        output = os.path.join(out_dir, f"{step_name}.wav")
        config = "experiments/mvp_a_rave_latent/config_chain_boost.yaml" if stage == 'a' else "experiments/mvp_d_ckpt_morph/config_chain_boost.yaml"
        script = "experiments/mvp_a_rave_latent/run.sh" if stage == 'a' else "experiments/mvp_d_ckpt_morph/run.sh"
        
        cmd = ["bash", script, "--config", config, "--mode", "render", "--input", current_input, "--output", output]
        run_task(f"Chain Stage {real_step} ({stage.upper()})", cmd, log_path)
        current_input = output

if __name__ == "__main__":
    run_chain_resume()
