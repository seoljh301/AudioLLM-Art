import numpy as np
import logging
from src.modules.mvp_j.mimi_bend import MimiBender, MimiBendParams
from src.core.v2.llm_helmsman import LLMHelmsman
from src.core.v2.disagreement_noise import MultiAgentNoise

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("v2_orchestrator")

def v2_demo_loop():
    print("--- AudioArt V2: Semantic Orchestration Demo ---")
    rng = np.random.default_rng(42)
    
    # 1. Multi-Agent Disagreement ( Babel Tower )
    agent_noise = MultiAgentNoise()
    captions = [
        "A calm ocean wave hitting the shore",
        "Ethereal white noise with a soft bell",
        "Aggressive industrial metal grinding"
    ]
    disagreement = agent_noise.calculate_entropy(captions)
    print(f"[ Babel ] Models disagreed on the sound. Score: {disagreement:.3f}")

    # 2. LLM Helmsman ( The Navigator )
    helmsman = LLMHelmsman()
    final_decision = captions[2] # Let's assume the aggressive one won
    steer = helmsman.analyze_caption(final_decision)
    print(f"[ Navigator ] Setting symphony to: Distortion={steer.distortion_intensity}, FilterBias={steer.filter_cutoff_bias}")

    # 3. Mimi Semantic Bending ( The Soul )
    bender = MimiBender()
    params = MimiBendParams(mode="intertwined", semantic_rate=0.05, acoustic_rate=0.2)
    print(f"[ Soul ] Mimi strategy initiated. Bending meaning by 5% and texture by 20%.")

    print("\n--- V2 Framework Ready for Integration ---")

if __name__ == "__main__":
    v2_demo_loop()
