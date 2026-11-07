"""Dataset-neutral contracts for prepared image construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .manifest import PreparedImageSpec


@dataclass(frozen=True)
class ImageBuildContext:
    spec: PreparedImageSpec
    tasks: tuple[dict[str, Any], ...]
    test_command: str = ""


@dataclass(frozen=True)
class VerificationContext:
    spec: PreparedImageSpec
    tasks: tuple[dict[str, Any], ...]
    test_command: str = ""


class DatasetImageAdapter(Protocol):
    """Dataset-specific preparation and validation behind a common builder."""

    dataset_type: str

    def create_prepare_script(self, context: ImageBuildContext) -> str: ...

    def verify_image(
        self,
        client: Any,
        image: str,
        context: VerificationContext,
    ) -> Any: ...
