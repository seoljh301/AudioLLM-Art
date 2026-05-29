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
    # New V2 Rhythm Parameters
    bpm_multiplier: float = 1.0     # 1.0 = base BPM, 2.0 = double time (Drum n Bass)
    glitch_density: float = 0.1     # 0.0 to 1.0 probability of inserting random ghost notes/stutters
    kick_decay: float = 15.0        # Higher = shorter/punchier kick, Lower = booming 808 sub kick

class LLMHelmsman:
    """V2 Orchestrator: Maps AudioLLM captions to symphony parameters."""
    
    def analyze_caption(self, caption: str) -> SteeringParams:
        """Heuristic mapping from text to control signals.
        
        Example captions:
        - "Dark, glitchy industrial noise" -> High distortion, dense rhythm
        - "Calm ambient drone" -> Slow rhythm, booming long kicks
        - "Aggressive screaming chaos" -> Double BPM, extreme glitch
        """
        cap = caption.lower()
        p = SteeringParams()

        # 1. Darkness/Brightness (Filter mapping)
        if any(w in cap for w in ["dark", "deep", "bass", "underwater", "heavy"]):
            p.filter_cutoff_bias = -0.3 
            p.kick_decay = 8.0  # Long, booming 808-style kicks
        elif any(w in cap for w in ["bright", "sparkle", "sharp", "high", "crystal"]):
            p.filter_cutoff_bias = 0.4  
            p.kick_decay = 30.0 # Short, punchy, clicky kicks

        # 2. Chaos/Stability (Distortion, LFO, Rhythm mapping)
        if any(w in cap for w in ["glitch", "noise", "broken", "chaos", "aggressive"]):
            p.distortion_intensity = 0.5
            p.lfo_rate_scale = 1.8
            p.bpm_multiplier = 1.5      # Speed up the beat
            p.glitch_density = 0.85     # Extreme stutter and ghost notes
        elif any(w in cap for w in ["calm", "smooth", "ambient", "ethereal", "drone"]):
            p.distortion_intensity = 0.05
            p.lfo_rate_scale = 0.6
            p.bpm_multiplier = 0.75     # Slow down the beat
            p.glitch_density = 0.0      # Clean, simple rhythm

        # 3. Meaning focus
        if "voice" in cap or "speech" in cap:
            p.semantic_anchoring = 1.5 

        log.info(f"Helmsman steer for %r: %s", caption[:40], p)
        return p

def steer_net_dynamic(base_config: Dict[str, Any], steer: SteeringParams):
    """Applies the helmsman's will to a Multinet configuration."""
    new_config = base_config.copy()
    # In V2, we would scale envelope speeds and filter ranges by steer.lfo_rate_scale etc.
    return new_config
