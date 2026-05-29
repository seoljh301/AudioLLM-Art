import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from scripts.micro_semantic_sequencer_demo import NeuralTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("tracker_test")

OUT_DIR = Path("runs/micro_sequencer")

def run_advanced_test():
    log.info("--- Starting Advanced Neural Tracker Validation ---")
    tracker = NeuralTracker(bpm=140)
    
    # Simulating a 4-Bar continuous sequence from an LLM.
    # Bar 1: Intro/Sparse
    # Bar 2: Adding groove
    # Bar 3: Glitch introduction (Ratchets)
    # Bar 4: Complete IDM chaos (Extreme ratcheting, reverse logic simulation)
    
    multi_bar_json = """
    [
      {"step": 0, "sample": "K", "ratchet": 1},
      {"step": 8, "sample": "H", "ratchet": 1},
      {"step": 16, "sample": "S", "ratchet": 1},
      {"step": 24, "sample": "H", "ratchet": 2},
      
      {"step": 32, "sample": "K", "ratchet": 1},
      {"step": 36, "sample": "H", "ratchet": 1},
      {"step": 40, "sample": "H", "ratchet": 1},
      {"step": 44, "sample": "K", "ratchet": 2},
      {"step": 48, "sample": "S", "ratchet": 1},
      {"step": 56, "sample": "H", "ratchet": 3},
      
      {"step": 64, "sample": "K", "ratchet": 1},
      {"step": 68, "sample": "H", "ratchet": 4},
      {"step": 72, "sample": "H", "ratchet": 4},
      {"step": 76, "sample": "K", "ratchet": 1},
      {"step": 80, "sample": "S", "ratchet": 1},
      {"step": 88, "sample": "H", "ratchet": 8}, 
      {"step": 92, "sample": "S", "ratchet": 2},
      
      {"step": 96, "sample": "K", "ratchet": 2},
      {"step": 98, "sample": "K", "ratchet": 3},
      {"step": 100, "sample": "H", "ratchet": 8},
      {"step": 104, "sample": "H", "ratchet": 16}, 
      {"step": 108, "sample": "S", "ratchet": 1},
      {"step": 110, "sample": "S", "ratchet": 4},
      {"step": 112, "sample": "K", "ratchet": 1},
      {"step": 116, "sample": "H", "ratchet": 6},
      {"step": 120, "sample": "H", "ratchet": 8},
      {"step": 124, "sample": "S", "ratchet": 2}
    ]
    """
    
    log.info("Parsing 4-Bar JSON sequence (128 steps total)...")
    try:
        audio = tracker.sequence_from_json(multi_bar_json, bars=4)
        
        # Verify length
        expected_samples = int(tracker.samples_per_bar * 4)
        actual_samples = len(audio)
        log.info(f"Expected samples: {expected_samples}, Actual: {actual_samples}")
        
        if expected_samples != actual_samples:
            log.error("Length mismatch! Tracker logic has a bug.")
            return
            
        out_file = OUT_DIR / "advanced_4bar_idm_test.wav"
        
        # Add a light master limiter to glue the sequence
        audio = np.tanh(audio * 1.5) / np.tanh(1.5)
        
        sf.write(out_file, audio, tracker.sr)
        log.info(f"SUCCESS: Advanced IDM sequence written to {out_file}")
        
    except Exception as e:
        log.error(f"Test failed with exception: {e}")

if __name__ == "__main__":
    run_advanced_test()
