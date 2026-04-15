from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Frame:
    """A single decoded video frame with capture metadata."""

    image: np.ndarray  # BGR, shape (H, W, 3)
    timestamp_utc: datetime
    source_label: str
