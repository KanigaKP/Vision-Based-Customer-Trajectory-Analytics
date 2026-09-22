"""
Vision-Based Customer Trajectory Analytics — Spatial Zone Definitions.

This module defines the Zone data model and configuration loader/saver.
Zones are represented in 2D floor-coordinate space (meters or normalized ground grid)
matching the homography coordinate plane.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import yaml


@dataclass
class Zone:
    """
    Represents a spatial retail store zone defined on a 2D floor plane.

    Attributes:
        zone_id (str): Unique string identifier for the zone (e.g., 'shelf_a').
        name (str): Human-readable zone name (e.g., 'Shelf A (Snacks & Beverages)').
        polygon (List[Tuple[float, float]]): Vertices defining the polygon boundary.
        color (str): Hex color code or string for visualization overlay.
        description (Optional[str]): Detailed description of retail zone purpose.
    """
    zone_id: str
    name: str
    polygon: List[Tuple[float, float]]
    color: str = "#3498db"
    description: Optional[str] = None

    def __post_init__(self):
        """Validate zone parameters upon creation."""
        if not self.zone_id or not isinstance(self.zone_id, str):
            raise ValueError("zone_id must be a non-empty string.")
        
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string.")

        if not isinstance(self.polygon, (list, tuple)) or len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.zone_id}' polygon must contain at least 3 vertices. "
                f"Got: {self.polygon}"
            )

        # Convert vertices to list of float tuples
        formatted_vertices = []
        for pt in self.polygon:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise ValueError(f"Each vertex in zone '{self.zone_id}' must be a (x, y) coordinate pair. Got: {pt}")
            formatted_vertices.append((float(pt[0]), float(pt[1])))
        
        self.polygon = formatted_vertices

    def to_dict(self) -> Dict[str, Any]:
        """Convert Zone object to dictionary representation."""
        data = {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": [list(pt) for pt in self.polygon],
            "color": self.color,
        }
        if self.description:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        """Create Zone instance from dictionary."""
        return cls(
            zone_id=data["zone_id"],
            name=data["name"],
            polygon=data["polygon"],
            color=data.get("color", "#3498db"),
            description=data.get("description"),
        )


class ZoneConfig:
    """Helper class to load and save Zone definitions from YAML configuration files."""

    @staticmethod
    def load_from_config(
        config_path_or_dict: Any,
        dataset_key: str = "merl"
    ) -> Tuple[List[Zone], int]:
        """
        Load zones and hysteresis window from a YAML config file or dictionary.

        Args:
            config_path_or_dict (Union[str, Dict]): File path or loaded YAML dict.
            dataset_key (str): Dataset key under 'zones' section (e.g. 'merl', 'mot17_04').

        Returns:
            Tuple[List[Zone], int]: List of parsed Zone objects and the hysteresis window N.
        """
        if isinstance(config_path_or_dict, str):
            if not os.path.exists(config_path_or_dict):
                raise FileNotFoundError(f"Configuration file not found: {config_path_or_dict}")
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        elif isinstance(config_path_or_dict, dict):
            config = config_path_or_dict
        else:
            raise TypeError("config_path_or_dict must be a file path string or dictionary.")

        zones_sec = config.get("zones", {})
        hysteresis_window = int(zones_sec.get("hysteresis_window", 3))

        zone_list_data = zones_sec.get(dataset_key, [])
        if not zone_list_data:
            return [], hysteresis_window

        zones = [Zone.from_dict(z_data) for z_data in zone_list_data]
        return zones, hysteresis_window

    @staticmethod
    def save_to_yaml(
        zones: List[Zone],
        output_path: str,
        dataset_key: str = "merl",
        hysteresis_window: int = 3
    ) -> None:
        """
        Save Zone definitions and hysteresis window to a standalone YAML file.

        Args:
            zones (List[Zone]): List of Zone objects.
            output_path (str): Destination file path.
            dataset_key (str): Dataset key label.
            hysteresis_window (int): Frame buffer count N.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        zones_data = [z.to_dict() for z in zones]
        content = {
            "zones": {
                "hysteresis_window": hysteresis_window,
                dataset_key: zones_data
            }
        }
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)
