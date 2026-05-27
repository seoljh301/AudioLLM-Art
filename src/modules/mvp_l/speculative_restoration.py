import torch
import numpy as np

class SpeculativeRestoration:
    """MVP-L: Speculates what sound noise 'wants' to be."""
    
    def dream_from_noise(self, duration_s: float, sr: int, noise_type: str = "white") -> np.ndarray:
        """Generates pure noise as a 'blank canvas' for the model to interpret."""
        N = int(duration_s * sr)
        if noise_type == "white":
            return np.random.standard_normal(N).astype(np.float32)
        elif noise_type == "pink":
            # Simple pink noise approximation
            white = np.random.standard_normal(N)
            pink = np.cumsum(white) # Very crude brown/pink drift
            return (pink / np.max(np.abs(pink))).astype(np.float32)
        return np.zeros(N, dtype=np.float32)

    def force_hallucination(self, z: torch.Tensor, intensity: float = 2.0) -> torch.Tensor:
        """Amplifies the latent features of noise to trigger model hallucinations."""
        # By multiplying the latent of noise, we force the decoder to 
        # 'over-interpret' the minimal structure present in the noise.
        return z * intensity
