"""Generative rendering for Codebook Organ.
"""

from __future__ import annotations
import logging
import numpy as np

from .codebook_organ import OrganParams, generate_tokens
from src.modules.mvp_c.codec_io import CodecHandle, decode_tokens
from src.core.mix import MixConfig, soft_limiter

logger = logging.getLogger(__name__)

def render_generative(
    handle: CodecHandle,
    params: OrganParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate audio purely from codebook patterns."""
    tokens = generate_tokens(handle.n_quantizers, params, rng)
    logger.info("Generated tokens: %s (mode=%s)", tokens.shape, params.mode)
    
    out = decode_tokens(handle, tokens)
    
    # Simple normalization & limiter
    out = soft_limiter(out, drive=1.0)
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.9
        
    return out.astype(np.float32)
