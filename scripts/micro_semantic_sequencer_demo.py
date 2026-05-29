import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("neural_tracker")

OUT_DIR = Path("runs/micro_sequencer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class NeuralTracker:
    """Micro-Semantic Sequencer that places one-shots based on LLM JSON grids."""
    
    def __init__(self, sr=48000, bpm=120):
        self.sr = sr
        self.bpm = bpm
        self.bps = bpm / 60.0
        # 1 bar = 4 beats. We divide 1 bar into 32 steps (1/32nd notes)
        self.steps_per_bar = 32 
        self.samples_per_bar = int(self.sr * 4 * (60.0 / bpm))
        self.samples_per_step = self.samples_per_bar // self.steps_per_bar
        
        # Load or generate the Palette (One-shots)
        self.palette = self._generate_palette()
        
    def _generate_palette(self):
        """Generates simple one-shots simulating AudioLDM/MVP outputs."""
        log.info("Generating Palette (One-shots)...")
        palette = {}
        
        # [K] Sub Kick (0.3s)
        t_k = np.linspace(0, 0.3, int(self.sr * 0.3), endpoint=False)
        kick = np.sin(2 * np.pi * (150 * np.exp(-30 * t_k) + 45) * t_k) * np.exp(-15 * t_k)
        palette['K'] = kick.astype(np.float32)
        
        # [H] Glitch Hat (0.05s)
        t_h = np.linspace(0, 0.05, int(self.sr * 0.05), endpoint=False)
        hat = np.random.standard_normal(len(t_h)) * np.exp(-80 * t_h)
        # highpass
        from scipy.signal import butter, lfilter
        b, a = butter(4, 8000, fs=self.sr, btype='high')
        palette['H'] = lfilter(b, a, hat).astype(np.float32) * 0.6
        
        # [S] Metallic Snare/Clang (0.2s)
        t_s = np.linspace(0, 0.2, int(self.sr * 0.2), endpoint=False)
        # FM synthesis for metallic sound
        mod = np.sin(2 * np.pi * 800 * t_s)
        car = np.sin(2 * np.pi * 200 * t_s + 2.0 * mod)
        snare = car * np.exp(-20 * t_s)
        palette['S'] = snare.astype(np.float32) * 0.5
        
        return palette

    def sequence_from_json(self, json_grid_str: str, bars: int = 1) -> np.ndarray:
        """Parses LLM JSON and renders the exact audio."""
        grid = json.loads(json_grid_str)
        total_samples = self.samples_per_bar * bars
        out_audio = np.zeros(total_samples, dtype=np.float32)
        
        for event in grid:
            step = event.get("step")
            sample_key = event.get("sample")
            ratchet = event.get("ratchet", 1) # How many times to repeat within this step
            
            if sample_key not in self.palette:
                continue
                
            sample_data = self.palette[sample_key]
            
            # Start position of the step
            step_start_idx = step * self.samples_per_step
            
            # Ratcheting logic (Micro-timing)
            # If ratchet is 4, we play the sample 4 times evenly spaced within the single 1/32nd note step
            # effectively creating 1/128th note stutter rolls.
            ratchet_spacing = self.samples_per_step // ratchet
            
            for r in range(ratchet):
                pos = step_start_idx + (r * ratchet_spacing)
                end_pos = min(pos + len(sample_data), total_samples)
                length_to_write = end_pos - pos
                if length_to_write > 0:
                    out_audio[pos:end_pos] += sample_data[:length_to_write]
                    
        return np.clip(out_audio, -1.0, 1.0)


def demo():
    print("--- Neural Tracker Demo ---")
    tracker = NeuralTracker(bpm=130)
    
    # 1. Draft 1.0 (Naive, human-like straight beat)
    # Kicks on 0, 8, 16, 24. Hats on every 2. Snare on 8, 24.
    draft_1_json = """
    [
      {"step": 0, "sample": "K", "ratchet": 1},
      {"step": 4, "sample": "H", "ratchet": 1},
      {"step": 8, "sample": "S", "ratchet": 1},
      {"step": 12, "sample": "H", "ratchet": 1},
      {"step": 16, "sample": "K", "ratchet": 1},
      {"step": 20, "sample": "H", "ratchet": 1},
      {"step": 24, "sample": "S", "ratchet": 1},
      {"step": 28, "sample": "H", "ratchet": 1}
    ]
    """
    audio_v1 = tracker.sequence_from_json(draft_1_json)
    sf.write(OUT_DIR / "draft_v1_straight.wav", audio_v1, tracker.sr)
    print("Saved Draft v1 (Straight Beat).")
    
    # [AudioLLM Critique Simulator]
    print("\n[Judge Critique on v1]: '너무 지루하고 전형적이다. 1/128박자의 미친듯한 래칫(Ratchet)과 기계적인 오류를 추가해라.'")
    
    # 2. Draft 2.0 (LLM generates micro-timing and ratchets based on critique)
    draft_2_json = """
    [
      {"step": 0, "sample": "K", "ratchet": 1},
      {"step": 2, "sample": "K", "ratchet": 2}, 
      {"step": 4, "sample": "H", "ratchet": 4}, 
      {"step": 8, "sample": "S", "ratchet": 1},
      {"step": 11, "sample": "H", "ratchet": 2},
      {"step": 14, "sample": "K", "ratchet": 1},
      {"step": 16, "sample": "H", "ratchet": 8},
      {"step": 20, "sample": "H", "ratchet": 1},
      {"step": 22, "sample": "K", "ratchet": 3},
      {"step": 24, "sample": "S", "ratchet": 1},
      {"step": 27, "sample": "S", "ratchet": 2},
      {"step": 30, "sample": "H", "ratchet": 4}
    ]
    """
    audio_v2 = tracker.sequence_from_json(draft_2_json)
    sf.write(OUT_DIR / "draft_v2_glitch.wav", audio_v2, tracker.sr)
    print("Saved Draft v2 (Neural Glitch Tracker).")

if __name__ == "__main__":
    demo()