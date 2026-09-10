"""
FastAPI backend for Mono Chrome Semen Analysis Assist.
Powered by Mono Chrome — FOR IVD SOLUTIONS
"""

import os
import shutil
import tempfile
import uuid
from typing import List

import cv2
import numpy as np
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
WHO_NORMAL_FORMS_LOWER_LIMIT_PERCENT = 4.0

app = FastAPI(title="Mono Chrome Semen Analysis API", version="0.2.0")

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


class MorphologyResponse(BaseModel):
    sampleId: str
    normalFormsPercent: float
    confidenceScore: float
    warnings: List[str]


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "mono-chrome-semen-analysis",
        "powered_by": "Mono Chrome",
        "docs": "/docs",
    }


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


@app.post("/analyze/concentration-image", response_model=AnalysisResponse)
async def analyze_concentration_image(
    image: UploadFile = File(...),
    microns_per_pixel: float = Form(...),
    chamber_depth_microns: float = Form(...),
    dilution_factor: float = Form(1.0),
):
    """Concentration from a single still microscope image."""
    if microns_per_pixel <= 0 or chamber_depth_microns <= 0:
        raise HTTPException(status_code=400, detail="Calibration values must be positive.")

    sample_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, image.filename or "upload.jpg")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        frame = cv2.imread(image_path)
        if frame is None:
            raise HTTPException(status_code=422, detail="Could not read the uploaded image.")

        width_px, height_px = frame_dimensions(frame)
        objects = detect_objects(frame)
        count = float(len(objects))

        params = CalibrationParams(
            microns_per_pixel=microns_per_pixel,
            chamber_depth_microns=chamber_depth_microns,
            dilution_factor=dilution_factor,
        )
        concentration = concentration_million_per_ml(
            average_objects_per_frame=count,
            frame_width_px=width_px,
            frame_height_px=height_px,
            params=params,
        )

        report = build_report(
            sample_id=sample_id,
            per_frame_counts=[int(count)],
            estimated_concentration_million_per_ml=concentration,
        )

    warnings = list(report.warnings)
    warnings.append("تحليل من صورة واحدة — الدقة أقل من تحليل فيديو متعدد الإطارات.")
    warnings.append("Powered by Mono Chrome")

    return AnalysisResponse(
        sampleId=report.sample_id,
        framesAnalyzed=1,
        averageObjectsPerFrame=count,
        estimatedConcentrationMillionPerMl=report.estimated_concentration_million_per_ml,
        confidenceScore=min(report.confidence_score, 0.45),
        belowReferenceLimit=report.below_reference_limit,
        whoReferenceLimitMillionPerMl=report.who_reference_limit_million_per_ml,
        warnings=warnings,
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


@app.post("/analyze/morphology", response_model=MorphologyResponse)
async def analyze_morphology(image: UploadFile = File(...)):
    """
    Prototype morphology estimate from classical shape features.
    Real clinical morphology needs stained slides + trained models + expert labels
    (self-learning path: accumulate technician corrections from the app).
    """
    sample_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, image.filename or "upload.jpg")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        frame = cv2.imread(image_path)
        if frame is None:
            raise HTTPException(status_code=422, detail="Could not read the uploaded image.")

        objects = detect_objects(frame)
        if not objects:
            return MorphologyResponse(
                sampleId=sample_id,
                normalFormsPercent=0.0,
                confidenceScore=0.2,
                warnings=[
                    "لم يُكتشف أجسام كافية لتقدير الشكل.",
                    "Morphology prototype — requires expert review and self-learning data.",
                ],
            )

        # Heuristic: circularity in mid-range as proxy for "more normal-looking" heads.
        normalish = sum(1 for o in objects if 0.55 <= o.circularity <= 0.95)
        normal_pct = 100.0 * normalish / len(objects)

    return MorphologyResponse(
        sampleId=sample_id,
        normalFormsPercent=normal_pct,
        confidenceScore=0.25,
        warnings=[
            "تقدير أولي للشكل (morphology) غير مُعتمد سريريًا.",
            "الحد المرجعي WHO للأشكال الطبيعية ≈ 4٪ (Kruger strict criteria context).",
            "فعّل مراجعة الفني في التطبيق لتغذية مسار self-learning.",
            "Powered by Mono Chrome",
        ],
    )
