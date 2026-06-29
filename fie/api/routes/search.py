"""FastAPI routes for Video Search by Question."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.search.video_index import VideoIndex, VideoIndexer
from fie.search.query_engine import QueryEngine

router = APIRouter(prefix="/search", tags=["Video Search"])

INDEX_DIR = Path("data/search_index")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IndexVideoRequest(BaseModel):
    video_path: str = Field(description="Путь к видео файлу")
    video_id: str | None = Field(default=None, description="Уникальный ID (default: имя файла)")
    checkpoint_path: str = Field(default="checkpoints/best_model-v2.ckpt")
    device: str = Field(default="cuda", description="'cuda' или 'cpu'")
    max_frames: int = Field(default=0, description="Максимум кадров (0 = всё видео)")
    start_frame: int = Field(default=0, description="Начать с этого кадра")
    match_duration_min: float = Field(default=90.0, description="Длительность матча в минутах")
    video_offset_s: float = Field(default=0.0, description="Смещение начала матча в секундах")
    min_confidence: float = Field(default=0.35, description="Порог уверенности событий")
    save: bool = Field(default=True, description="Сохранить индекс в файл")


class IndexVideoResponse(BaseModel):
    video_id: str
    total_events: int
    duration_s: float
    indexed_at: str
    event_types: dict
    saved_path: str | None


class SearchRequest(BaseModel):
    query: str = Field(description="Вопрос на естественном языке: 'все удары в первом тайме'")
    video_id: str = Field(description="ID индекса видео (из /search/index-video)")
    max_results: int = Field(default=20, description="Максимум результатов")
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class SearchResponse(BaseModel):
    query: str
    backend: str
    matched_count: int
    parsed_query: dict
    events: list[dict]


class SearchAndClipRequest(BaseModel):
    query: str = Field(description="Вопрос на естественном языке")
    video_id: str = Field(description="ID индекса видео")
    video_path: str = Field(description="Путь к видео для нарезки клипов")
    output_dir: str = Field(default="clips/search", description="Папка для клипов")
    pre_seconds: float = Field(default=4.0, description="Секунд до события")
    post_seconds: float = Field(default=3.0, description="Секунд после события")
    max_clips: int = Field(default=10)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class IndexFromEventsRequest(BaseModel):
    """Построить индекс из готового списка событий (без запуска детектора)."""
    video_path: str
    video_id: str | None = None
    events: list[dict] = Field(
        description="[{frame_idx, event_type, confidence, player_id?}]"
    )
    match_duration_min: float = 90.0
    video_offset_s: float = 0.0
    min_confidence: float = 0.35
    save: bool = True


# ---------------------------------------------------------------------------
# POST /search/index-video
# ---------------------------------------------------------------------------

@router.post("/index-video", response_model=IndexVideoResponse)
async def index_video(request: IndexVideoRequest) -> IndexVideoResponse:
    """
    Проиндексировать видео: запустить EventDetector и сохранить все события.

    Этот процесс может занять несколько минут для полного матча.
    Индекс сохраняется в `data/search_index/{video_id}.json` и
    используется повторно без повторной обработки видео.
    """
    if not Path(request.video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    indexer = VideoIndexer(
        video_path=request.video_path,
        video_id=request.video_id,
        device=request.device,
        checkpoint_path=request.checkpoint_path,
        save_dir=INDEX_DIR,
        match_duration_min=request.match_duration_min,
        video_offset_s=request.video_offset_s,
        min_confidence=request.min_confidence,
    )

    logger.info("Starting video indexing: {}", request.video_path)
    index = indexer.build(
        max_frames=request.max_frames,
        start_frame=request.start_frame,
    )

    saved_path = None
    if request.save:
        saved_path = str(index.save(INDEX_DIR))

    stats = index.stats()
    return IndexVideoResponse(
        video_id=index.video_id,
        total_events=stats["total_events"],
        duration_s=stats["duration_s"],
        indexed_at=stats["indexed_at"],
        event_types=stats["event_types"],
        saved_path=saved_path,
    )


# ---------------------------------------------------------------------------
# POST /search/index-from-events
# ---------------------------------------------------------------------------

@router.post("/index-from-events", response_model=IndexVideoResponse)
async def index_from_events(request: IndexFromEventsRequest) -> IndexVideoResponse:
    """
    Построить индекс из готового списка событий (без запуска детектора).

    Полезно если события уже были получены через /tracking или другой API:
    передаёте список {frame_idx, event_type, confidence, player_id?}
    и получаете проиндексированный, готовый к поиску индекс.
    """
    indexer = VideoIndexer(
        video_path=request.video_path,
        video_id=request.video_id,
        save_dir=INDEX_DIR,
        match_duration_min=request.match_duration_min,
        video_offset_s=request.video_offset_s,
        min_confidence=request.min_confidence,
    )

    index = indexer.build_from_events(request.events)

    saved_path = None
    if request.save:
        saved_path = str(index.save(INDEX_DIR))

    stats = index.stats()
    return IndexVideoResponse(
        video_id=index.video_id,
        total_events=stats["total_events"],
        duration_s=stats["duration_s"],
        indexed_at=stats["indexed_at"],
        event_types=stats["event_types"],
        saved_path=saved_path,
    )


# ---------------------------------------------------------------------------
# POST /search/query
# ---------------------------------------------------------------------------

@router.post("/query", response_model=SearchResponse)
async def search_query(request: SearchRequest) -> SearchResponse:
    """
    Найти события по текстовому запросу.

    Примеры запросов:
    - "все удары в первом тайме" / "all shots in first half"
    - "опасные моменты после 75 минуты" / "dangerous moments after 75 minutes"
    - "передачи игрока #7" / "passes by player #7"
    - "last 10 minutes tackles"
    - "голевые моменты второго тайма"

    Поддерживает русский и английский язык.
    Без API ключа работает в rule-based режиме.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")

    index = _load_index(request.video_id)

    engine = QueryEngine(
        anthropic_api_key=request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )

    result = engine.search(request.query, index, max_results=request.max_results)

    logger.info(
        "Search: '{}' → {} results (backend={})",
        request.query, len(result.matched_events), result.backend,
    )

    return SearchResponse(
        query=result.query,
        backend=result.backend,
        matched_count=len(result.matched_events),
        parsed_query=result.parsed.to_dict(),
        events=[e.to_dict() for e in result.matched_events],
    )


# ---------------------------------------------------------------------------
# POST /search/query-and-clip
# ---------------------------------------------------------------------------

@router.post("/query-and-clip")
async def search_and_clip(request: SearchAndClipRequest) -> dict:
    """
    Найти события и нарезать клипы за один запрос.

    Клипы сохраняются в `output_dir` и пути возвращаются в ответе.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")

    if not Path(request.video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    index = _load_index(request.video_id)

    engine = QueryEngine(
        anthropic_api_key=request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )

    result = engine.search_and_clip(
        query=request.query,
        index=index,
        video_path=request.video_path,
        output_dir=request.output_dir,
        pre_seconds=request.pre_seconds,
        post_seconds=request.post_seconds,
        max_clips=request.max_clips,
    )

    logger.info(
        "Search+clip: '{}' → {} clips extracted",
        request.query, result.get("clips_extracted", 0),
    )

    return result


# ---------------------------------------------------------------------------
# GET /search/index/{video_id}
# ---------------------------------------------------------------------------

@router.get("/index/{video_id}", response_model=IndexVideoResponse)
async def get_index(video_id: str) -> IndexVideoResponse:
    """
    Получить статистику существующего индекса.
    """
    index = _load_index(video_id)
    stats = index.stats()

    return IndexVideoResponse(
        video_id=index.video_id,
        total_events=stats["total_events"],
        duration_s=stats["duration_s"],
        indexed_at=stats["indexed_at"],
        event_types=stats["event_types"],
        saved_path=str(INDEX_DIR / f"{video_id}.json"),
    )


# ---------------------------------------------------------------------------
# GET /search/list
# ---------------------------------------------------------------------------

@router.get("/list")
async def list_indexes() -> dict:
    """Список всех доступных видеоиндексов."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    indexes = []

    for path in sorted(INDEX_DIR.glob("*.json")):
        try:
            idx = VideoIndex.load(path)
            stats = idx.stats()
            indexes.append(stats)
        except Exception:
            continue

    return {"total": len(indexes), "indexes": indexes}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_index(video_id: str) -> VideoIndex:
    path = INDEX_DIR / f"{video_id}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index not found: {video_id}. Run POST /search/index-video first.",
        )
    return VideoIndex.load(path)
