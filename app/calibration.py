"""
Calibration math: converting a raw object count per video frame into a
concentration figure (million sperm / mL).
"""

from dataclasses import dataclass


@dataclass
class CalibrationParams:
    microns_per_pixel: float
    chamber_depth_microns: float
    dilution_factor: float = 1.0


def field_of_view_area_mm2(frame_width_px: int, frame_height_px: int, microns_per_pixel: float) -> float:
    width_microns = frame_width_px * microns_per_pixel
    height_microns = frame_height_px * microns_per_pixel
    area_microns2 = width_microns * height_microns
    return area_microns2 / 1_000_000.0


def counted_volume_ml(frame_width_px: int, frame_height_px: int, params: CalibrationParams) -> float:
    area_mm2 = field_of_view_area_mm2(frame_width_px, frame_height_px, params.microns_per_pixel)
    depth_mm = params.chamber_depth_microns / 1000.0
    volume_mm3 = area_mm2 * depth_mm
    return volume_mm3 / 1000.0


def concentration_million_per_ml(
    average_objects_per_frame: float,
    frame_width_px: int,
    frame_height_px: int,
    params: CalibrationParams,
) -> float:
    volume_ml = counted_volume_ml(frame_width_px, frame_height_px, params)
    if volume_ml <= 0:
        return 0.0
    cells_per_ml = (average_objects_per_frame / volume_ml) * params.dilution_factor
    return cells_per_ml / 1_000_000.0
