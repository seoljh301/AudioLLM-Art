import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as signal

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("infinite_tracker")

OUT_DIR = Path("runs/infinite_tracker")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class DynamicDSPGenerator:
    """Mocks AudioLDM: Parses text descriptions and generates specific one-shots."""
    def __init__(self, sr=48000):
        self.sr = sr
        
    def generate_sound(self, description: str, duration_s: float = 0.3) -> np.ndarray:
        desc = description.lower()
        t = np.linspace(0, duration_s, int(self.sr * duration_s), endpoint=False)
        out = np.zeros_like(t)
        # Deterministic generation based on text so "glass shatter" always sounds like the SAME "glass shatter"
        rng = np.random.default_rng(sum(ord(c) for c in desc)) 
        
        # Heuristic AudioLDM Mock logic
        if "kick" in desc or "thump" in desc or "sub" in desc:
            freq = 60 if "sub" in desc else 120
            out = np.sin(2 * np.pi * (freq * np.exp(-20 * t) + 40) * t) * np.exp(-15 * t)
        elif "metal" in desc or "clang" in desc or "pipe" in desc:
            # FM synth for metal
            mod = np.sin(2 * np.pi * rng.uniform(400, 1200) * t)
            out = np.sin(2 * np.pi * rng.uniform(200, 600) * t + 2.0 * mod) * np.exp(-10 * t)
        elif "glass" in desc or "shatter" in desc or "snap" in desc:
            noise = rng.standard_normal(len(t))
            b, a = signal.butter(4, 5000, fs=self.sr, btype='high')
            out = signal.lfilter(b, a, noise) * np.exp(-40 * t)
        elif "laser" in desc or "zap" in desc or "synth" in desc:
            # Pitch drop sine
            out = signal.sawtooth(2 * np.pi * (1000 * np.exp(-50 * t)) * t) * np.exp(-20 * t)
        else:
            # Default noisy glitch
            out = rng.standard_normal(len(t)) * np.exp(-30 * t)
            
        # Add random distortion if requested
        if "distort" in desc or "harsh" in desc:
            out = np.tanh(out * 5.0)
            
        return out.astype(np.float32)

class InfiniteTracker:
    def __init__(self, sr=48000, bpm=130):
        self.sr = sr
        self.bpm = bpm
        self.bps = bpm / 60.0
        self.steps_per_bar = 32 
        self.samples_per_bar = int(self.sr * 4 * (60.0 / bpm))
        self.samples_per_step = self.samples_per_bar // self.steps_per_bar
        self.dsp = DynamicDSPGenerator(sr)
        self.palette = {}
        
    def update_palette(self, json_palette_str: str):
        """LLM defines the symbols and what they sound like."""
        items = json.loads(json_palette_str)
        log.info(f"--- Loading New Sonic Palette from LLM ---")
        for item in items:
            sym = item['symbol']
            desc = item['description']
            dur = item.get('duration', 0.2)
            log.info(f"  [{sym}] -> '{desc}' (Rendering...)")
            self.palette[sym] = self.dsp.generate_sound(desc, dur)
            
    def sequence_from_json(self, json_grid_str: str, bars: int = 1) -> np.ndarray:
        grid = json.loads(json_grid_str)
        total_samples = self.samples_per_bar * bars
        out_audio = np.zeros(total_samples, dtype=np.float32)
        
        for event in grid:
            step = event.get("step")
            sample_key = event.get("sample")
            ratchet = event.get("ratchet", 1)
            
            if sample_key not in self.palette:
                continue
                
            sample_data = self.palette[sample_key]
            step_start_idx = step * self.samples_per_step
            ratchet_spacing = self.samples_per_step // ratchet
            
            for r in range(ratchet):
                pos = step_start_idx + (r * ratchet_spacing)
                end_pos = min(pos + len(sample_data), total_samples)
                length_to_write = end_pos - pos
                if length_to_write > 0:
                    out_audio[pos:end_pos] += sample_data[:length_to_write]
                    
        return np.clip(out_audio, -1.0, 1.0)

def loop_demo():
    print("=== Infinite Tracker: Dynamic Palette Loop Demo ===")
    tracker = InfiniteTracker(bpm=120)
    
    # --- Generation 1 ---
    print("\n[Generation 1: Industrial Factory]")
    # 1. LLM defines its own instruments
    palette_gen1 = """
    [
      {"symbol": "K", "description": "heavy distorted sub kick", "duration": 0.4},
      {"symbol": "P", "description": "metallic pipe clang", "duration": 0.3},
      {"symbol": "S", "description": "harsh steam release hiss", "duration": 0.15}
    ]
    """
    tracker.update_palette(palette_gen1)
    
    # 2. LLM sequences those instruments
    seq_gen1 = """
    [
      {"step": 0, "sample": "K"}, {"step": 8, "sample": "S"},
      {"step": 16, "sample": "P"}, {"step": 24, "sample": "S", "ratchet": 2},
      {"step": 28, "sample": "K", "ratchet": 2}
    ]
    """
    audio1 = tracker.sequence_from_json(seq_gen1, bars=1)
    sf.write(OUT_DIR / "gen1_industrial.wav", audio1, tracker.sr)
    print("Saved: gen1_industrial.wav")
    
    # --- Generation 2 ---
    print("\n[LLM Judge Critique]: '너무 비어있고 낡은 공장 소리 같다. 미래지향적인 디테일이 필요하다. 하이역대를 채우기 위해 유리 깨지는 소리와 레이저 신스를 발명해서 추가해라. 베이스는 유지해라.'")
    print("\n[Generation 2: Sci-Fi Glitch Evolution]")
    
    # 3. LLM invents new sounds based on critique
    palette_gen2 = """
    [
      {"symbol": "K", "description": "heavy distorted sub kick", "duration": 0.4},
      {"symbol": "Z", "description": "laser zap synth drop", "duration": 0.1},
      {"symbol": "G", "description": "glass shatter bright", "duration": 0.2}
    ]
    """
    tracker.update_palette(palette_gen2)
    
    # 4. LLM writes a new, more complex sequence
    seq_gen2 = """
    [
      {"step": 0, "sample": "K"}, {"step": 4, "sample": "Z", "ratchet": 4},
      {"step": 8, "sample": "G"}, {"step": 12, "sample": "Z", "ratchet": 2},
      {"step": 16, "sample": "K", "ratchet": 2}, {"step": 24, "sample": "G", "ratchet": 3},
      {"step": 28, "sample": "Z", "ratchet": 8}
    ]
    """
    audio2 = tracker.sequence_from_json(seq_gen2, bars=1)
    sf.write(OUT_DIR / "gen2_scifi.wav", audio2, tracker.sr)
    print("Saved: gen2_scifi.wav")
    
    print("\nSUCCESS: Infinite Dynamic Palette loop executed. The LLM successfully invented new sounds and sequenced them.")

if __name__ == "__main__":
    loop_demo()