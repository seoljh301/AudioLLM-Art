import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Any

log = logging.getLogger("llm_helmsman")

@dataclass
class SteeringParams:
    lfo_rate_scale: float = 1.0
    filter_cutoff_bias: float = 0.0
    distortion_intensity: float = 0.0
    semantic_anchoring: float = 1.0

class LLMHelmsman:
    """V2 Orchestrator: Maps AudioLLM captions to symphony parameters."""
    
    def analyze_caption(self, caption: str) -> SteeringParams:
        """Heuristic mapping from text to control signals.
        
        Example captions:
        - "Dark, glitchy industrial noise" -> High distortion, Low filter
        - "Sparkling ethereal crystals" -> High filter, Low LFO
        - "Aggressive screaming chaos" -> High LFO, High distortion
        """
        cap = caption.lower()
        p = SteeringParams()

        # 1. Darkness/Brightness (Filter mapping)
        if any(w in cap for w in ["dark", "deep", "bass", "underwater", "heavy"]):
            p.filter_cutoff_bias = -0.3 # Close the filter
        elif any(w in cap for w in ["bright", "sparkle", "sharp", "high", "crystal"]):
            p.filter_cutoff_bias = 0.4  # Open the filter

        # 2. Chaos/Stability (Distortion/LFO mapping)
        if any(w in cap for w in ["glitch", "noise", "broken", "chaos", "aggressive"]):
            p.distortion_intensity = 0.5
            p.lfo_rate_scale = 1.8
        elif any(w in cap for w in ["calm", "smooth", "ambient", "ethereal"]):
            p.distortion_intensity = 0.05
            p.lfo_rate_scale = 0.6

        # 3. Meaning focus
        if "voice" in cap or "speech" in cap:
            p.semantic_anchoring = 1.5 # Protect the meaning tokens more

        log.info(f"Helmsman steer for %r: %s", caption[:40], p)
        return p

def steer_net_dynamic(base_config: Dict[str, Any], steer: SteeringParams):
    """Applies the helmsman's will to a Multinet configuration."""
    new_config = base_config.copy()
    # In V2, we would scale envelope speeds and filter ranges by steer.lfo_rate_scale etc.
    return new_config
