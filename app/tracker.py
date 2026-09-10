"""Phase-2 motility tracker: greedy nearest-neighbor tracking + WHO-style classification."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from app.detector import DetectedObject, detect_objects

MAX_MATCH_DISTANCE_PX = 25.0
MIN_TRACK_LENGTH_FRAMES = 5
PROGRESSIVE_VSL_THRESHOLD_UM_S = 25.0
IMMOTILE_VCL_THRESHOLD_UM_S = 5.0


@dataclass
class Track:
    points_px: List[Tuple[float, float, int]] = field(default_factory=list)

    def path_length_px(self) -> float:
        total = 0.0
        for i in range(1, len(self.points_px)):
            x0, y0, _ = self.points_px[i - 1]
            x1, y1, _ = self.points_px[i]
            total += float(np.hypot(x1 - x0, y1 - y0))
        return total

    def net_displacement_px(self) -> float:
        if len(self.points_px) < 2:
            return 0.0
        x0, y0, _ = self.points_px[0]
        x1, y1, _ = self.points_px[-1]
        return float(np.hypot(x1 - x0, y1 - y0))

    def duration_frames(self) -> int:
        if len(self.points_px) < 2:
            return 0
        return self.points_px[-1][2] - self.points_px[0][2]


@dataclass
class MotilityResult:
    tracks_analyzed: int
    progressive_percent: float
    non_progressive_percent: float
    immotile_percent: float
    total_motility_percent: float
    confidence_score: float
    warnings: List[str] = field(default_factory=list)


def _link_frames_to_tracks(per_frame_detections: List[List[DetectedObject]]) -> List[Track]:
    open_tracks: List[Track] = []
    closed_tracks: List[Track] = []

    for frame_idx, detections in enumerate(per_frame_detections):
        unmatched = set(range(len(detections)))
        for track in open_tracks:
            if not track.points_px:
                continue
            last_x, last_y, last_frame = track.points_px[-1]
            if frame_idx - last_frame > 1:
                continue
            best_idx: Optional[int] = None
            best_dist = MAX_MATCH_DISTANCE_PX
            for idx in unmatched:
                d = detections[idx]
                dist = float(np.hypot(d.x - last_x, d.y - last_y))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                d = detections[best_idx]
                track.points_px.append((d.x, d.y, frame_idx))
                unmatched.discard(best_idx)

        for idx in unmatched:
            d = detections[idx]
            open_tracks.append(Track(points_px=[(d.x, d.y, frame_idx)]))

        still_open = []
        for track in open_tracks:
            last_frame = track.points_px[-1][2]
            if last_frame == frame_idx or frame_idx - last_frame <= 1:
                still_open.append(track)
            else:
                closed_tracks.append(track)
        open_tracks = still_open

    closed_tracks.extend(open_tracks)
    return closed_tracks


def _classify_track(track: Track, fps: float, microns_per_pixel: float) -> str:
    duration_s = track.duration_frames() / max(fps, 1.0)
    if duration_s <= 0:
        return "immotile"
    vcl = (track.path_length_px() * microns_per_pixel) / duration_s
    vsl = (track.net_displacement_px() * microns_per_pixel) / duration_s
    if vcl < IMMOTILE_VCL_THRESHOLD_UM_S:
        return "immotile"
    if vsl >= PROGRESSIVE_VSL_THRESHOLD_UM_S:
        return "progressive"
    return "non_progressive"


def analyze_motility(
    frames: List,
    fps: float,
    microns_per_pixel: float,
) -> MotilityResult:
    warnings: List[str] = [
        "تصنيف الحركة تقريبي (تتبع بسيط) ولم يُعايَر سريريًا بعد — راجع النتائج يدويًا."
    ]
    if not frames:
        return MotilityResult(0, 0.0, 0.0, 100.0, 0.0, 0.2, warnings)

    per_frame = [detect_objects(f) for f in frames]
    tracks = _link_frames_to_tracks(per_frame)
    usable = [t for t in tracks if t.duration_frames() >= MIN_TRACK_LENGTH_FRAMES]

    if not usable:
        warnings.append("لم يُعثر على مسارات كافية لتصنيف الحركة.")
        return MotilityResult(0, 0.0, 0.0, 100.0, 0.0, 0.3, warnings)

    labels = [_classify_track(t, fps, microns_per_pixel) for t in usable]
    n = len(labels)
    pr = 100.0 * labels.count("progressive") / n
    np_ = 100.0 * labels.count("non_progressive") / n
    im = 100.0 * labels.count("immotile") / n
    total_mot = pr + np_
    confidence = 0.55 if n >= 10 else 0.35

    return MotilityResult(
        tracks_analyzed=n,
        progressive_percent=pr,
        non_progressive_percent=np_,
        immotile_percent=im,
        total_motility_percent=total_mot,
        confidence_score=confidence,
        warnings=warnings,
    )
