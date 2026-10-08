"""
Dataset definitions for NeuroStroke-XAI CT imaging pipeline.

Supports standard Python/PIL/NumPy usage and optional PyTorch Dataset
when torch is installed in the active environment.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    _HAS_TORCH = True
except ImportError:
    torch = None
    TorchDataset = object
    _HAS_TORCH = False


# Canonical class-to-index mapping (strictly consistent across the project)
CLASS_TO_IDX: Dict[str, int] = {
    "normal": 0,
    "ischemic": 1,
    "hemorrhagic": 2,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}


class CTStrokeDataset(TorchDataset):
    """
    Dataset loader for curated 3-class brain CT images.

    Reads from pre-generated group-level split CSVs (train.csv, validation.csv, test.csv).
    Guarantees consistent label mapping and deterministic image loading.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        project_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_path: Path to the split CSV file.
            project_root: Root directory of the project for resolving relative image paths.
            transform: Callable transformation pipeline to apply to PIL Image.
        """
        self.csv_path = Path(csv_path)
        self.project_root = (
            Path(project_root) if project_root else self.csv_path.resolve().parents[2]
        )
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found at: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self._validate_schema()

    def _validate_schema(self) -> None:
        required_cols = {"image_path", "class"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"CSV must contain columns {required_cols}, found: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Resolve image path
        raw_path = Path(row["image_path"])
        image_path = raw_path if raw_path.is_absolute() else self.project_root / raw_path

        # Load RGB image (medical CT rendered pseudo-color)
        image = Image.open(image_path).convert("RGB")

        # Map label
        class_name = row["class"]
        if class_name not in CLASS_TO_IDX:
            raise KeyError(
                f"Unknown class '{class_name}' at index {idx}. "
                f"Expected one of {list(CLASS_TO_IDX.keys())}."
            )
        label = CLASS_TO_IDX[class_name]

        # Apply transformation if specified
        if self.transform is not None:
            image = self.transform(image)

        # If PyTorch is available and label is raw int, model loaders expect tensor labels
        if _HAS_TORCH and torch is not None:
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = label

        return image, target

    def get_metadata(self, idx: int) -> dict:
        """Return the raw metadata row for a given index."""
        row = self.df.iloc[idx]
        return {
            "image_path": str(row["image_path"]),
            "class": row["class"],
            "case_id": row.get("case_id", None),
            "original_split": row.get("original_split", None),
        }
