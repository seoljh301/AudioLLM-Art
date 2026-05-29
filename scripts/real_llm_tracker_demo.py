import logging
import soundfile as sf
import torch
import gc
from pathlib import Path
from src.core.v2.real_llm_agents import RealTextComposerAgent
from src.core.v2.real_audio_judge import RealAudioJudge
from scripts.infinite_tracker_loop import InfiniteTracker

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("real_llm_demo")

OUT_DIR = Path("runs/micro_sequencer/real_llm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_real_llm_demo():
    print("=== The Ultimate Closed-Loop: Real LLM Composer + Real AudioLLM Judge ===")
    
    # 1. Initialize the Real LLM Agent (The Composer)
    composer = RealTextComposerAgent(model_id="Qwen/Qwen1.5-1.8B-Chat")
    
    # 2. Initialize the Audio Engine (The Instrument)
    tracker = InfiniteTracker(bpm=130)
    
    theme = "A futuristic cybernetic factory malfunctioning"
    print(f"\n[Theme] {theme}")
    
    # 3. Composer invents the Palette
    print("\n>> Composer is inventing instruments...")
    palette_json = composer.generate_palette(theme)
    tracker.update_palette(palette_json)
    
    # 4. Composer sequences the Beat (Draft 1)
    print("\n>> Composer is writing the sequence (Draft 1)...")
    seq_json_v1 = composer.generate_sequence(palette_json, theme)
    
    draft_1_path = OUT_DIR / "real_llm_draft_1.wav"
    try:
        audio_v1 = tracker.sequence_from_json(seq_json_v1, bars=2)
        sf.write(draft_1_path, audio_v1, tracker.sr)
        print(f"Saved: {draft_1_path}")
    except Exception as e:
        print(f"Failed to parse or sequence V1: {e}")
        return

    # To save VRAM, we unload the composer while the judge listens
    del composer
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Initialize the Real AudioLLM Judge
    print("\n>> Summoning the AudioLLM Judge...")
    judge = RealAudioJudge()
    
    # Judge listens to Draft 1
    print("\n>> Judge is listening to Draft 1...")
    real_critique = judge.critique_audio(str(draft_1_path), theme)
    
    # Unload judge, reload composer
    del judge
    torch.cuda.empty_cache()
    gc.collect()
    composer = RealTextComposerAgent(model_id="Qwen/Qwen1.5-1.8B-Chat")
    
    # 6. Composer re-sequences based on the REAL critique
    print("\n>> Composer is rewriting the sequence based on the REAL critique (Draft 2)...")
    seq_json_v2 = composer.generate_sequence(palette_json, theme, critique=real_critique)
    
    draft_2_path = OUT_DIR / "real_llm_draft_2.wav"
    try:
        audio_v2 = tracker.sequence_from_json(seq_json_v2, bars=2)
        sf.write(draft_2_path, audio_v2, tracker.sr)
        print(f"Saved: {draft_2_path}")
    except Exception as e:
        print(f"Failed to parse or sequence V2: {e}")
        
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_real_llm_demo()
