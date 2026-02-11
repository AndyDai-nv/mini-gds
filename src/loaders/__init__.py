"""Model loaders: HuggingFace vs NIXL GDS."""

from .hf_loader import HFLoader
from .nixl_gds_loader import NIXLGDSLoader

__all__ = ["HFLoader", "NIXLGDSLoader"]
