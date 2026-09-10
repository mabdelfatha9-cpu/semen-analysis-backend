"""Builds the final JSON report with WHO 6th-edition reference limits."""

from dataclasses import dataclass, field
from typing import List
import statistics

WHO_CONCENTRATION_LOWER_LIMIT_MILLION_PER_ML = 16.0


@dataclass
class ConcentrationReport:
    sample_id: str
    frames_analyzed: int
    average_objects_per_frame: float
    estimated_concentration_million_per_ml: float
    confidence_score: float
    below_reference_limit: bool
    who_reference_limit_million_per_ml: float = WHO_CONCENTRATION_LOWER_LIMIT_MILLION_PER_ML
    warnings: List[str] = field(default_factory=list)


def compute_confidence(per_frame_counts: List[int]) -> float:
    if len(per_frame_counts) < 2:
        return 0.3
    mean = statistics.mean(per_frame_counts)
    if mean == 0:
        return 0.2
    stdev = statistics.stdev(per_frame_counts)
    coefficient_of_variation = stdev / mean
    if coefficient_of_variation < 0.15:
        return 0.8
    elif coefficient_of_variation < 0.35:
        return 0.6
    elif coefficient_of_variation < 0.6:
        return 0.4
    else:
        return 0.2


def build_report(
    sample_id: str,
    per_frame_counts: List[int],
    estimated_concentration_million_per_ml: float,
) -> ConcentrationReport:
    average = statistics.mean(per_frame_counts) if per_frame_counts else 0.0
    confidence = compute_confidence(per_frame_counts)
    below_limit = estimated_concentration_million_per_ml < WHO_CONCENTRATION_LOWER_LIMIT_MILLION_PER_ML

    warnings: List[str] = [
        "نتيجة أولية بمساعدة الحاسوب باستخدام كشف كلاسيكي (OpenCV) — لم يتم التحقق السريري من دقتها بعد.",
        "التقنية الحالية تقيس التركيز فقط، ولا تشمل الحركة أو الشكل.",
    ]

    if confidence < 0.5:
        warnings.append("تباين كبير بين الفريمات — يُنصح بإعادة التصوير أو بمراجعة يدوية دقيقة للنتيجة.")

    if len(per_frame_counts) < 10:
        warnings.append("عدد الفريمات المحللة قليل نسبيًا — دقة التقدير محدودة.")

    return ConcentrationReport(
        sample_id=sample_id,
        frames_analyzed=len(per_frame_counts),
        average_objects_per_frame=average,
        estimated_concentration_million_per_ml=estimated_concentration_million_per_ml,
        confidence_score=confidence,
        below_reference_limit=below_limit,
        warnings=warnings,
    )
