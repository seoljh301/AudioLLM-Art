import numpy as np
import logging
from typing import List

log = logging.getLogger("multi_agent_disagreement")

class MultiAgentNoise:
    """V2 Entropy Engine: Calculates disagreement between multiple AI models.
    
    The core idea is 'The Babel Tower of Machines'. If models disagree on 
    what they are hearing, the dissonance is converted into physical noise.
    """
    
    def calculate_entropy(self, captions: List[str]) -> float:
        """Computes a simple 'Disagreement Score' based on vocabulary variance.
        
        Args:
            captions: List of strings from different models (e.g. Qwen, SALMONN, LTU)
            
        Returns:
            entropy: float from 0.0 (full agreement) to 1.0 (pure chaos)
        """
        if not captions:
            return 0.0
            
        # 1. Simple word-set overlap analysis
        all_words = []
        for cap in captions:
            all_words.append(set(cap.lower().split()))
            
        # Calculate Jaccard-style distance between agents
        similarities = []
        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                intersection = len(all_words[i].intersection(all_words[j]))
                union = len(all_words[i].union(all_words[j]))
                similarities.append(intersection / union if union > 0 else 0)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        disagreement = 1.0 - avg_similarity
        
        log.info(f"Disagreement Score: {disagreement:.3f} across {len(captions)} agents")
        return disagreement

def apply_disagreement_noise(audio: np.ndarray, disagreement: float, rng: np.random.Generator):
    """Injects noise proportional to how much the models argued."""
    noise_level = disagreement * 0.5 # Scale down to usable levels
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    return audio + noise * noise_level
