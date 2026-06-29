"""FastAPI routes for Video Clipping — автоматическая нарезка ключевых моментов."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from fie.clipping.clipper import ClipEvent, ClipExtractor, ClipResult, HighlightsBuilder

router = APIRouter(prefix="/clipping", tags=["Clipping"])

CLIPS_DIR = Path("data/clips")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClipEventInput(BaseModel):
    frame_idx: int
    timestamp: float
    event_type: str = Field(description="shot / pass / tackle / dribble / clearance / ball_receipt / carry")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    player_id: int | None = None
    team: str | None = None


class ExtractClipsRequest(BaseModel):
    video_path: str = Field(description="Путь к видеофайлу на сервере")
    events: list[ClipEventInput]
    pre_seconds: float = Field(default=5.0, description="Секунд до события")
    post_seconds: float = Field(default=3.0, description="Секунд после события")
    output_dir: str | None = Field(default=None, description="Куда сохранять (default: data/clips)")
    min_confidence: float = Field(default=0.3, description="Минимальная уверенность события")


class ClipResultOut(BaseModel):
    event_type: str
    timestamp: float
    confidence: float
    player_id: int | None
    team: str | None
    clip_path: str
    duration_s: float
    start_frame: int
    end_frame: int


class ExtractClipsResponse(BaseModel):
    total_events: int
    extracted_clips: int
    clips: list[ClipResultOut]
    output_dir: str


class HighlightsRequest(BaseModel):
    video_path: str = Field(description="Путь к исходному видео")
    events: list[ClipEventInput]
    output_path: str | None = Field(default=None, description="Куда сохранить highlights")
    pre_seconds: float = Field(default=5.0)
    post_seconds: float = Field(default=3.0)
    sort_by: str = Field(default="time", description="'time' или 'confidence'")
    add_title: bool = Field(default=True, description="Текстовый оверлей с типом события")
    min_confidence: float = Field(default=0.3)
    return_file: bool = Field(default=False, description="Вернуть файл в ответе (только для малых видео)")


# ---------------------------------------------------------------------------
# POST /clipping/extract-clips
# ---------------------------------------------------------------------------

@router.post("/extract-clips", response_model=ExtractClipsResponse)
async def extract_clips(request: ExtractClipsRequest) -> ExtractClipsResponse:
    """
    Нарезать видеоклипы вокруг событий.

    Для каждого события создаёт отдельный MP4 файл:
    - `pre_seconds` секунд до события
    - `post_seconds` секунд после события

    Возвращает список путей к созданным клипам.
    """
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    if not request.events:
        raise HTTPException(status_code=422, detail="events list is empty")

    out_dir = Path(request.output_dir) if request.output_dir else CLIPS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        extractor = ClipExtractor(
            video_path=video_path,
            output_dir=out_dir,
            pre_seconds=request.pre_seconds,
            post_seconds=request.post_seconds,
            min_confidence=request.min_confidence,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    events = [
        ClipEvent(
            frame_idx=e.frame_idx,
            timestamp=e.timestamp,
            event_type=e.event_type,
            confidence=e.confidence,
            player_id=e.player_id,
            team=e.team,
        )
        for e in request.events
    ]

    clips = extractor.extract_batch(events)

    logger.info(
        "Extract clips | events={} | extracted={} | dir={}",
        len(events), len(clips), out_dir,
    )

    return ExtractClipsResponse(
        total_events=len(events),
        extracted_clips=len(clips),
        clips=[ClipResultOut(**c.to_dict()) for c in clips],
        output_dir=str(out_dir),
    )


# ---------------------------------------------------------------------------
# POST /clipping/highlights
# ---------------------------------------------------------------------------

@router.post("/highlights")
async def build_highlights(request: HighlightsRequest):
    """
    Собрать highlights-видео из событий.

    Нарезает клипы вокруг каждого события и склеивает их в одно видео.

    Если `return_file=true` — возвращает MP4 файл напрямую.
    Если `return_file=false` — возвращает путь к сохранённому файлу.
    """
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    if not request.events:
        raise HTTPException(status_code=422, detail="events list is empty")

    # Временная директория для промежуточных клипов
    tmp_dir = Path(tempfile.mkdtemp(prefix="fie_clips_"))

    try:
        extractor = ClipExtractor(
            video_path=video_path,
            output_dir=tmp_dir,
            pre_seconds=request.pre_seconds,
            post_seconds=request.post_seconds,
            min_confidence=request.min_confidence,
        )

        events = [
            ClipEvent(
                frame_idx=e.frame_idx,
                timestamp=e.timestamp,
                event_type=e.event_type,
                confidence=e.confidence,
                player_id=e.player_id,
                team=e.team,
            )
            for e in request.events
        ]

        clips = extractor.extract_batch(events)

        if not clips:
            raise HTTPException(
                status_code=422,
                detail="No clips extracted — check confidence threshold or frame indices",
            )

        builder = HighlightsBuilder(
            clips=clips,
            sort_by=request.sort_by,
            add_title=request.add_title,
        )

        if request.output_path:
            out_path = Path(request.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            CLIPS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = CLIPS_DIR / "highlights.mp4"

        builder.build(out_path)

        logger.info(
            "Highlights built | clips={} | duration={:.1f}s | path={}",
            len(clips), builder.total_duration(), out_path,
        )

        if request.return_file:
            return FileResponse(
                path=str(out_path),
                media_type="video/mp4",
                filename="highlights.mp4",
            )

        return {
            "status": "ok",
            "clips_count": len(clips),
            "total_duration_s": round(builder.total_duration(), 1),
            "output_path": str(out_path),
            "events": [c.to_dict() for c in clips],
        }

    except Exception as e:
        logger.error("Highlights build failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /clipping/clips — список сохранённых клипов
# ---------------------------------------------------------------------------

@router.get("/clips")
async def list_clips() -> dict:
    """Список всех сохранённых клипов в data/clips/."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    return {
        "clips_dir": str(CLIPS_DIR),
        "total": len(clips),
        "files": [
            {
                "name": c.name,
                "path": str(c),
                "size_mb": round(c.stat().st_size / 1024 / 1024, 2),
            }
            for c in clips
        ],
    }


# ---------------------------------------------------------------------------
# GET /clipping/clips/{filename} — скачать клип
# ---------------------------------------------------------------------------

@router.get("/clips/{filename}")
async def download_clip(filename: str) -> FileResponse:
    """Скачать конкретный клип по имени файла."""
    path = CLIPS_DIR / filename
    if not path.exists() or path.suffix != ".mp4":
        raise HTTPException(status_code=404, detail=f"Clip not found: {filename}")
    return FileResponse(path=str(path), media_type="video/mp4", filename=filename)
