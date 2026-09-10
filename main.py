"""
FastAPI backend for the Semen Analysis Assist app.

IMPORTANT: This is a research/assistive prototype, not a certified medical
device. See README.md for validation requirements before any real clinical use.

Railway / Docker:
    Deploy via Dockerfile. Railway sets $PORT automatically.
"""

import os
import shutil
import tempfile
import uuid
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.calibration import CalibrationParams, concentration_million_per_ml
from app.detector import detect_objects
from app.report import build_report
from app.tracker import analyze_motility
from app.video_processing import extract_frames, extract_sequential_frames, frame_dimensions

WHO_TOTAL_MOTILITY_LOWER_LIMIT_PERCENT = 42.0
WHO_PROGRESSIVE_MOTILITY_LOWER_LIMIT_PERCENT = 30.0

app = FastAPI(title="Semen Analysis Assist API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalibrationResponse(BaseModel):
    micronsPerPixel: float
    warnings: List[str]


class AnalysisResponse(BaseModel):
    sampleId: str
    framesAnalyzed: int
    averageObjectsPerFrame: float
    estimatedConcentrationMillionPerMl: float
    confidenceScore: float
    belowReferenceLimit: bool
    whoReferenceLimitMillionPerMl: float
    warnings: List[str]


class MotilityResponse(BaseModel):
    sampleId: str
    tracksAnalyzed: int
    progressiveMotilityPercent: float
    nonProgressiveMotilityPercent: float
    immotilePercent: float
    totalMotilityPercent: float
    confidenceScore: float
    belowReferenceLimit: bool
    whoTotalMotilityLimitPercent: float
    whoProgressiveMotilityLimitPercent: float
    warnings: List[str]


@app.get("/")
def root():
    return {"status": "ok", "service": "semen-analysis-assist", "docs": "/docs"}


@app.post("/calibrate", response_model=CalibrationResponse)
async def calibrate(
    image: UploadFile = File(...),
    known_distance_microns: float = Form(...),
):
    warnings = [
        "المعايرة التلقائية من الصورة ما زالت placeholder — استخدم الإدخال اليدوي في التطبيق."
    ]
    return CalibrationResponse(micronsPerPixel=0.5, warnings=warnings)


@app.post("/analyze/concentration", response_model=AnalysisResponse)
async def analyze_concentration(
    video: UploadFile = File(...),
    microns_per_pixel: float = Form(...),
    chamber_depth_microns: float = Form(...),
    dilution_factor: float = Form(1.0),
):
    if microns_per_pixel <= 0 or chamber_depth_microns <= 0:
        raise HTTPException(status_code=400, detail="Calibration values must be positive.")

    sample_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, video.filename or "upload.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        frames = extract_frames(video_path, max_frames=30)
        if not frames:
            raise HTTPException(status_code=422, detail="Could not read any frames from the uploaded video.")

        width_px, height_px = frame_dimensions(frames[0])
        per_frame_counts = []
        for frame in frames:
            objects = detect_objects(frame)
            per_frame_counts.append(len(objects))

        average_objects_per_frame = sum(per_frame_counts) / len(per_frame_counts)

        params = CalibrationParams(
            microns_per_pixel=microns_per_pixel,
            chamber_depth_microns=chamber_depth_microns,
            dilution_factor=dilution_factor,
        )
        concentration = concentration_million_per_ml(
            average_objects_per_frame=average_objects_per_frame,
            frame_width_px=width_px,
            frame_height_px=height_px,
            params=params,
        )

        report = build_report(
            sample_id=sample_id,
            per_frame_counts=per_frame_counts,
            estimated_concentration_million_per_ml=concentration,
        )

    return AnalysisResponse(
        sampleId=report.sample_id,
        framesAnalyzed=report.frames_analyzed,
        averageObjectsPerFrame=report.average_objects_per_frame,
        estimatedConcentrationMillionPerMl=report.estimated_concentration_million_per_ml,
        confidenceScore=report.confidence_score,
        belowReferenceLimit=report.below_reference_limit,
        whoReferenceLimitMillionPerMl=report.who_reference_limit_million_per_ml,
        warnings=report.warnings,
    )


@app.post("/analyze/motility", response_model=MotilityResponse)
async def analyze_motility_endpoint(
    video: UploadFile = File(...),
    microns_per_pixel: float = Form(...),
):
    if microns_per_pixel <= 0:
        raise HTTPException(status_code=400, detail="microns_per_pixel must be positive.")

    sample_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, video.filename or "upload.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        frames, fps = extract_sequential_frames(video_path, max_frames=60)
        if not frames:
            raise HTTPException(status_code=422, detail="Could not read any frames from the uploaded video.")

        motility = analyze_motility(frames=frames, fps=fps, microns_per_pixel=microns_per_pixel)

    below_limit = motility.total_motility_percent < WHO_TOTAL_MOTILITY_LOWER_LIMIT_PERCENT

    warnings = list(motility.warnings)
    if fps <= 1.0:
        warnings.append(
            "لم يتم قراءة معدل الفريمات (FPS) من الفيديو بدقة — سرعات الحركة المحسوبة قد تكون غير دقيقة."
        )

    return MotilityResponse(
        sampleId=sample_id,
        tracksAnalyzed=motility.tracks_analyzed,
        progressiveMotilityPercent=motility.progressive_percent,
        nonProgressiveMotilityPercent=motility.non_progressive_percent,
        immotilePercent=motility.immotile_percent,
        totalMotilityPercent=motility.total_motility_percent,
        confidenceScore=motility.confidence_score,
        belowReferenceLimit=below_limit,
        whoTotalMotilityLimitPercent=WHO_TOTAL_MOTILITY_LOWER_LIMIT_PERCENT,
        whoProgressiveMotilityLimitPercent=WHO_PROGRESSIVE_MOTILITY_LOWER_LIMIT_PERCENT,
        warnings=warnings,
    )
