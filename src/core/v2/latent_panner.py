import numpy as np
import torch

class LatentPanner:
    """Core V2: Maps Z-space coordinates to Stereo Panning."""
    
    def calculate_pan(self, z: torch.Tensor, x_dim: int = 0, y_dim: int = 1) -> np.ndarray:
        """Determines stereo position based on latent vector coordinates.
        
        Args:
            z: (latent_dim, T_z) tensor
            x_dim: The latent dimension used for Left/Right bias
            y_dim: The latent dimension used for Front/Back (intensity) bias
            
        Returns:
            pan_curve: (T_z,) array of pan values [-1.0, 1.0]
        """
        # Extract the coordinate dims and normalize them roughly
        # RAVE latents are often N(0,1)-ish, so tanh makes a good pan map
        x_coords = z[x_dim].cpu().numpy()
        pan_curve = np.tanh(x_coords)
        
        return pan_curve

    def apply_latent_stereo(self, mono_audio: np.ndarray, pan_curve: np.ndarray) -> np.ndarray:
        """Applies a time-varying pan curve to mono audio."""
        T = len(mono_audio)
        # Resample pan curve to audio length
        t_z = np.linspace(0, 1, len(pan_curve))
        t_audio = np.linspace(0, 1, T)
        pan_interpolated = np.interp(t_audio, t_z, pan_curve)
        
        # Equal power panning
        l_gain = np.cos((pan_interpolated + 1.0) * np.pi / 4.0)
        r_gain = np.sin((pan_interpolated + 1.0) * np.pi / 4.0)
        
        stereo = np.zeros((T, 2), dtype=np.float32)
        stereo[:, 0] = mono_audio * l_gain
        stereo[:, 1] = mono_audio * r_gain
        
        return stereo
