import subprocess
import os
import time

def run_task(name, cmd, log_path):
    with open(log_path, "a") as f_log:
        f_log.write(f"\n\n--- Starting {name} at {time.ctime()} ---\n")
        f_log.flush()
        start_t = time.time()
        
        # Redirect stdout/stderr directly to the log file for real-time monitoring
        result = subprocess.run(cmd, stdout=f_log, stderr=f_log)
        
        elapsed = time.time() - start_t
        if result.returncode == 0:
            f_log.write(f"  {name} SUCCESS in {elapsed:.2f}s\n")
        else:
            f_log.write(f"  {name} FAILED with exit code {result.returncode}\n")
        f_log.flush()

def run_chain():
    sequence = "addaada"
    current_input = "data/Ulaanbaatar.wav"
    out_dir = "runs/ulaanbaatar_pure_chain"
    log_path = os.path.join(out_dir, "pipeline.log")
    
    for i, stage in enumerate(sequence):
        step_name = f"step_{i+1}_{stage}"
        output = os.path.join(out_dir, f"{step_name}.wav")
        config = "experiments/mvp_a_rave_latent/config_chain_boost.yaml" if stage == 'a' else "experiments/mvp_d_ckpt_morph/config_chain_boost.yaml"
        script = "experiments/mvp_a_rave_latent/run.sh" if stage == 'a' else "experiments/mvp_d_ckpt_morph/run.sh"
        
        cmd = ["bash", script, "--config", config, "--mode", "render", "--input", current_input, "--output", output]
        run_task(f"Chain Stage {i+1} ({stage.upper()})", cmd, log_path)
        current_input = output

def run_freeze():
    input_path = "data/Ulaanbaatar.wav"
    out_dir = "runs/ulaanbaatar_freeze"
    output_path = os.path.join(out_dir, "ulaanbaatar_frozen.wav")
    log_path = os.path.join(out_dir, "freeze.log")
    
    cmd = ["bash", "experiments/mvp_f_neural_freeze/run.sh", "--mode", "render", "--input", input_path, "--output", output_path]
    run_task("Spectral Freeze", cmd, log_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/gpu_task_manager.py [chain|freeze]")
        sys.exit(1)
        
    task = sys.argv[1]
    if task == "chain":
        run_chain()
    elif task == "freeze":
        run_freeze()
