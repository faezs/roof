"""
Device utilities for heliostat project

Provides consistent device selection across the codebase with proper fallbacks
for MPS (Apple Silicon), CUDA, and CPU devices.
"""

import platform
import torch
import logging

logger = logging.getLogger(__name__)

def get_default_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. MPS (Metal Performance Shaders) on macOS with Apple Silicon
    2. CUDA if available and working
    3. CPU as fallback
    
    Returns
    -------
    torch.device
        The best available device
    """
    # Check for MPS on macOS (Apple Silicon)
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            logger.debug("Using MPS device (Apple Silicon)")
            return torch.device("mps")
    
    # Check for CUDA
    if torch.cuda.is_available():
        try:
            # Test CUDA by creating a small tensor
            test_tensor = torch.randn(2, 2, device="cuda")
            del test_tensor
            logger.debug("Using CUDA device")
            return torch.device("cuda")
        except RuntimeError as e:
            logger.warning(f"CUDA available but not working: {e}")
    
    # Fallback to CPU
    logger.debug("Using CPU device")
    return torch.device("cpu")

def ensure_device(tensor_or_device, default_device=None):
    """
    Ensure a tensor is on the specified device or convert device strings.
    
    Parameters
    ----------
    tensor_or_device : torch.Tensor, torch.device, str, or None
        Tensor to move or device specification
    default_device : torch.device, optional
        Default device if None provided
        
    Returns
    -------
    torch.Tensor or torch.device
        Tensor moved to device or device object
    """
    if default_device is None:
        default_device = get_default_device()
    
    if tensor_or_device is None:
        return default_device
    
    if isinstance(tensor_or_device, str):
        return torch.device(tensor_or_device)
    
    if isinstance(tensor_or_device, torch.device):
        return tensor_or_device
        
    if isinstance(tensor_or_device, torch.Tensor):
        return tensor_or_device.to(default_device)
    
    return default_device