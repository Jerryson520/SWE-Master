"""Build and verify immutable task images before agent rollout."""

from .manifest import (
    DEFAULT_PREPARED_REGISTRY,
    MANIFEST_SCHEMA_VERSION,
    PreparedImageSpec,
    load_manifest,
    resolve_prepared_image,
)
from .base import DatasetImageAdapter, ImageBuildContext, VerificationContext

__all__ = [
    "DEFAULT_PREPARED_REGISTRY",
    "MANIFEST_SCHEMA_VERSION",
    "PreparedImageSpec",
    "load_manifest",
    "resolve_prepared_image",
    "DatasetImageAdapter",
    "ImageBuildContext",
    "VerificationContext",
]
