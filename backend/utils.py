from dataclasses import dataclass
from typing import List

@dataclass
class MapPoint:
    markerOffset: int
    name: str
    coordinated: List[float] # x and y coords
