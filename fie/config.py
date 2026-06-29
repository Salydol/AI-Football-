from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Model ---
    yolo_model: str = "yolov8l.pt"
    yolo_confidence: float = 0.4
    yolo_device: str = "cuda"

    # --- Tracking ---
    bytetrack_track_thresh: float = 0.5
    bytetrack_match_thresh: float = 0.8
    bytetrack_frame_rate: int = 25

    # --- Field (FIFA standard) ---
    field_length_m: float = 105.0
    field_width_m: float = 68.0

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # --- Video ---
    max_frame_width: int = 1280


settings = Settings()
