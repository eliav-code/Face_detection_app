from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import time
import uuid

@dataclass
class FaceProfile:
    """
    Represents a registered person in the face recognition database.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    encodings: List[np.ndarray] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    thumbnail: Optional[str] = None  # Base64 JPEG string for preview in UI

    @property
    def primary_encoding(self) -> Optional[np.ndarray]:
        """Return the primary (first) face encoding if available."""
        return self.encodings[0] if self.encodings else None


@dataclass
class DetectionResult:
    """
    Represents a single detected face in a video frame.
    """
    name: str
    confidence: float  # Percentage (0 - 100)
    distance: float    # Euclidean face distance
    location: Tuple[int, int, int, int]  # (top, right, bottom, left) in original frame coords
    is_known: bool

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Alias for location (top, right, bottom, left)."""
        return self.location
