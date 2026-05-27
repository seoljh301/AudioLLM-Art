import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class WarpParams:
    # Scale factor for each quantizer layer (e.g. [1.0, 1.0, 0.8, 0.5])
    # 1.0 = normal speed, < 1.0 = slow motion
    layer_speeds: np.ndarray = np.ones(8, dtype=np.float32)

class NeuralTemporalWarp:
    """MVP-K: Warps the flow of time within neural layers (Quantizers)."""
    
    def warp_tokens(self, tokens: torch.Tensor, params: WarpParams) -> torch.Tensor:
        """Independently stretches each layer's time axis.
        
        Args:
            tokens: (n_q, T) tensor
            
        Returns:
            warped_tokens: (n_q, T) tensor with layers stretched/interpolated
        """
        n_q, T = tokens.shape
        out = tokens.clone()
        
        for q in range(n_q):
            speed = params.layer_speeds[q]
            if speed == 1.0:
                continue
                
            # Calculate source indices for the warp
            # If speed is 0.5, we want to play the first half of the tokens over the full T
            src_indices = np.linspace(0, T - 1, T) * speed
            src_indices = np.clip(src_indices, 0, T - 1).astype(np.int64)
            
            out[q, :] = tokens[q, src_indices]
            
        return out
