"""Central configuration for the AnemiaVision AI project.

This module keeps all runtime settings in one place so the training scripts,
evaluation utilities, and Flask application stay aligned.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", os.getenv("ANEMIA_TORCH_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.getenv("ANEMIA_TORCH_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("ANEMIA_TORCH_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.getenv("ANEMIA_TORCH_THREADS", "1"))

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
TORCH_NUM_THREADS = max(1, int(os.getenv("ANEMIA_TORCH_THREADS", "1")))

try:
    torch.set_num_threads(TORCH_NUM_THREADS)
except RuntimeError:
    pass

try:
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
except RuntimeError:
    pass


def normalize_token(value: str) -> str:
    """Normalize a string for case-insensitive filesystem matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_path(path_value: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve relative and absolute paths consistently."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_runtime_root() -> Path | None:
    """Return the configured persistent runtime root when one is available.

    Priority:
    1. ``ANEMIA_RUNTIME_ROOT`` — use on any host (Render disk, Fly volume, custom NAS).
    2. Render defaults — when ``RENDER`` is set, try ``RENDER_DISK_ROOT`` / ``/var/data``.

    **Vercel serverless:** there is no durable local disk between invocations. Set
    ``DATABASE_URL`` to a managed Postgres (Neon, Supabase, Vercel Postgres) so patient
    *rows* survive redeploys. For scan images and Grad-CAM files, either mount external
    object storage (S3 / Vercel Blob) in a future release or deploy this Docker app on
    Render/Fly/Railway with a persistent volume (see ``render.yaml``).
    """

    explicit_root = os.getenv("ANEMIA_RUNTIME_ROOT")
    if explicit_root:
        return resolve_path(explicit_root)

    if not os.getenv("RENDER"):
        return None

    for env_key in ("RENDER_DISK_ROOT", "RENDER_DATA_DIR", "RENDER_PERSISTENT_DIR"):
        candidate = os.getenv(env_key)
        if candidate:
            return resolve_path(candidate)

    default_render_root = Path("/var/data")
    if default_render_root.exists():
        return (default_render_root / "anemiavision").resolve()

    return None


def normalize_database_uri(uri: str) -> str:
    """Normalize incoming database URLs for SQLAlchemy compatibility."""

    normalized = uri.strip()
    if normalized.startswith("postgres://"):
        normalized = "postgresql://" + normalized[len("postgres://") :]

    if normalized.startswith("postgresql://") and "://" in normalized:
        scheme, remainder = normalized.split("://", 1)
        if "+" not in scheme:
            normalized = f"postgresql+psycopg://{remainder}"

    return normalized


def build_database_uri() -> str:
    """Resolve the active database URI from the environment or local SQLite."""

    explicit_uri = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")
    if explicit_uri:
        return normalize_database_uri(explicit_uri)
    return f"sqlite:///{DATABASE_PATH.as_posix()}"


def database_backend_name(database_uri: str) -> str:
    """Return the backend token used by the configured database URI."""

    scheme = database_uri.split("://", 1)[0].lower()
    return scheme.split("+", 1)[0]


def build_engine_options(database_uri: str) -> dict[str, object]:
    """Build SQLAlchemy engine options appropriate for the active backend."""

    if database_backend_name(database_uri) == "sqlite":
        return {
            "connect_args": {
                "check_same_thread": False,
                "timeout": float(os.getenv("ANEMIA_SQLITE_TIMEOUT_SECONDS", "1")),
            }
        }

    return {
        "pool_pre_ping": True,
        "pool_recycle": int(os.getenv("ANEMIA_DB_POOL_RECYCLE_SECONDS", "300")),
    }


def first_existing_path(candidates: Iterable[str | Path]) -> Path:
    """Return the first path that exists, or the first resolved candidate."""

    resolved_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        resolved = resolve_path(candidate)
        resolved_candidates.append(resolved)
        if resolved.exists():
            return resolved

    if not resolved_candidates:
        raise ValueError("At least one candidate path must be provided.")

    return resolved_candidates[0]


def find_split_directory(dataset_root: Path, aliases: Iterable[str]) -> Path:
    """Resolve a dataset split folder by matching aliases case-insensitively."""

    alias_tokens = {normalize_token(alias) for alias in aliases}
    if dataset_root.exists():
        for child in dataset_root.iterdir():
            if child.is_dir() and normalize_token(child.name) in alias_tokens:
                return child.resolve()
    primary_alias = next(iter(aliases))
    return (dataset_root / primary_alias).resolve()


def detect_device() -> torch.device:
    """Pick the best available device in CUDA -> MPS -> CPU order."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


TRAINING_PROFILE_ENV = os.getenv("ANEMIA_TRAIN_PROFILE", "auto").strip().lower()


def resolve_training_profile(device: torch.device) -> str:
    """Resolve the training profile from the environment and hardware."""

    profile = TRAINING_PROFILE_ENV
    if profile not in {"auto", "full", "fast"}:
        profile = "auto"

    if profile == "auto":
        return "fast" if device.type == "cpu" else "full"
    return profile


DATASET_ROOT = first_existing_path(
    [
        os.getenv("ANEMIA_DATASET_ROOT"),
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "dataset",
        PROJECT_ROOT.parent / "data",
    ]
)

TRAIN_DATASET_DIR = find_split_directory(DATASET_ROOT, ("train", "training"))
VALID_DATASET_DIR = find_split_directory(DATASET_ROOT, ("valid", "val", "validation"))
TEST_DATASET_DIR = find_split_directory(DATASET_ROOT, ("test", "testing"))
RUNTIME_ROOT = resolve_runtime_root()
RUNTIME_MEDIA_ROOT = RUNTIME_ROOT / "media" if RUNTIME_ROOT is not None else None

MODELS_DIR = resolve_path(os.getenv("ANEMIA_MODELS_DIR", PROJECT_ROOT / "models"))
STATIC_DIR = resolve_path(os.getenv("ANEMIA_STATIC_DIR", PROJECT_ROOT / "static"))
TEMPLATES_DIR = resolve_path(os.getenv("ANEMIA_TEMPLATES_DIR", PROJECT_ROOT / "templates"))
DATABASE_DIR = resolve_path(
    os.getenv("ANEMIA_DATABASE_DIR", (RUNTIME_ROOT / "database") if RUNTIME_ROOT else (PROJECT_ROOT / "database"))
)
REPORTS_DIR = resolve_path(
    os.getenv("ANEMIA_REPORTS_DIR", (RUNTIME_ROOT / "reports") if RUNTIME_ROOT else (PROJECT_ROOT / "reports"))
)
LOGS_DIR = resolve_path(
    os.getenv("ANEMIA_LOGS_DIR", (RUNTIME_ROOT / "logs") if RUNTIME_ROOT else (PROJECT_ROOT / "logs"))
)
EVALUATION_LOGS_DIR = resolve_path(
    os.getenv("ANEMIA_EVALUATION_LOGS_DIR", LOGS_DIR / "evaluation")
)
UPLOADS_DIR = resolve_path(
    os.getenv("ANEMIA_UPLOADS_DIR", (RUNTIME_MEDIA_ROOT / "uploads") if RUNTIME_MEDIA_ROOT else (STATIC_DIR / "uploads"))
)
GRADCAM_DIR = resolve_path(
    os.getenv("ANEMIA_GRADCAM_DIR", (RUNTIME_MEDIA_ROOT / "gradcam") if RUNTIME_MEDIA_ROOT else (STATIC_DIR / "gradcam"))
)

DATABASE_PATH = resolve_path(os.getenv("ANEMIA_DATABASE_PATH", DATABASE_DIR / "anemia_vision.db"))
DATABASE_URI = build_database_uri()
DATABASE_BACKEND = database_backend_name(DATABASE_URI)
_DEFAULT_BEST_CHECKPOINT = first_existing_path(
    [
        MODELS_DIR / "20260428_2028_best_model.pth",
        MODELS_DIR / "best_model.pth",
    ]
)
BEST_CHECKPOINT_PATH = resolve_path(os.getenv("ANEMIA_BEST_CHECKPOINT", str(_DEFAULT_BEST_CHECKPOINT)))
_DEFAULT_LATEST_CHECKPOINT = first_existing_path(
    [
        MODELS_DIR / "20260428_2028_last_checkpoint.pth",
        MODELS_DIR / "last_checkpoint.pth",
    ]
)
LATEST_CHECKPOINT_PATH = resolve_path(
    os.getenv("ANEMIA_LATEST_CHECKPOINT", str(_DEFAULT_LATEST_CHECKPOINT))
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ("non-anemic", "anemic")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
API_RATE_LIMIT_PER_MINUTE = int(os.getenv("ANEMIA_RATE_LIMIT_PER_MINUTE", "10"))
API_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("ANEMIA_RATE_LIMIT_WINDOW_SECONDS", "60"))
GUNICORN_WORKERS = int(os.getenv("ANEMIA_GUNICORN_WORKERS", "1"))
GUNICORN_TIMEOUT = int(os.getenv("ANEMIA_GUNICORN_TIMEOUT", "180"))
ENABLE_GRADCAM = os.getenv("ANEMIA_ENABLE_GRADCAM", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters shared across scripts."""

    image_size: tuple[int, int] = (300, 300)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10
    warmup_epochs: int = 5
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-6
    num_workers: int = field(default_factory=lambda: max(0, min(8, (os.cpu_count() or 2) - 1)))
    random_seed: int = 42
    device: torch.device = field(default_factory=detect_device)
    profile: str = field(default="auto")
    use_amp: bool = field(init=False)
    pin_memory: bool = field(init=False)

    def __post_init__(self) -> None:
        resolved_profile = resolve_training_profile(self.device)
        object.__setattr__(self, "profile", resolved_profile)

        if resolved_profile == "fast":
            object.__setattr__(self, "image_size", (224, 224))
            object.__setattr__(self, "batch_size", min(self.batch_size, 16))
            object.__setattr__(self, "epochs", min(self.epochs, 50))          # allow more epochs
            object.__setattr__(self, "early_stopping_patience", min(self.early_stopping_patience, 10))
            object.__setattr__(self, "warmup_epochs", min(self.warmup_epochs, 3))

        object.__setattr__(self, "use_amp", self.device.type == "cuda")
        object.__setattr__(self, "pin_memory", self.device.type == "cuda")

        num_workers_override = os.getenv("ANEMIA_NUM_WORKERS")
        if num_workers_override is not None:
            try:
                object.__setattr__(self, "num_workers", max(0, int(num_workers_override)))
            except ValueError:
                pass
        elif sys.platform.startswith("win") and self.device.type == "cpu":
            object.__setattr__(self, "num_workers", 0)


@dataclass(frozen=True)
class AugmentationConfig:
    """Image augmentation settings for training."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.3
    rotation_degrees: int = 15
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.1
    affine_translate: tuple[float, float] = (0.05, 0.05)
    affine_scale: tuple[float, float] = (0.95, 1.05)
    affine_shear: int = 5
    gaussian_blur_probability: float = 0.2
    gaussian_blur_kernel_size: int = 3
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0)

    def __post_init__(self) -> None:
        if TRAINING_CONFIG.profile != "fast":
            return

        object.__setattr__(self, "vertical_flip_probability", 0.0)
        object.__setattr__(self, "rotation_degrees", 10)
        object.__setattr__(self, "brightness", 0.2)
        object.__setattr__(self, "contrast", 0.2)
        object.__setattr__(self, "saturation", 0.1)
        object.__setattr__(self, "hue", 0.05)
        object.__setattr__(self, "affine_translate", (0.0, 0.0))
        object.__setattr__(self, "affine_scale", (1.0, 1.0))
        object.__setattr__(self, "affine_shear", 0)
        object.__setattr__(self, "gaussian_blur_probability", 0.0)


@dataclass(frozen=True)
class FlaskConfig:
    """Flask settings for the hospital-facing application."""

    secret_key: str = field(
        default_factory=lambda: os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
    )
    max_content_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "16")) * 1024 * 1024
    )
    upload_folder: Path = UPLOADS_DIR
    gradcam_folder: Path = GRADCAM_DIR
    reports_folder: Path = REPORTS_DIR
    database_uri: str = field(
        default_factory=lambda: DATABASE_URI
    )
    sqlalchemy_track_modifications: bool = False
    sqlalchemy_engine_options: dict[str, object] = field(
        default_factory=lambda: build_engine_options(DATABASE_URI)
    )
    allowed_extensions: tuple[str, ...] = tuple(sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS))

    def to_flask_dict(self) -> dict[str, object]:
        """Convert the config into a structure Flask understands."""

        return {
            "SECRET_KEY": self.secret_key,
            "MAX_CONTENT_LENGTH": self.max_content_length,
            "UPLOAD_FOLDER": str(self.upload_folder),
            "GRADCAM_FOLDER": str(self.gradcam_folder),
            "REPORTS_FOLDER": str(self.reports_folder),
            "SQLALCHEMY_DATABASE_URI": self.database_uri,
            "SQLALCHEMY_TRACK_MODIFICATIONS": self.sqlalchemy_track_modifications,
            "SQLALCHEMY_ENGINE_OPTIONS": self.sqlalchemy_engine_options,
        }


TRAINING_CONFIG = TrainingConfig()
AUGMENTATION_CONFIG = AugmentationConfig()
FLASK_CONFIG = FlaskConfig()

MODEL_ARCH_ENV = os.getenv("ANEMIA_MODEL_ARCH")
MODEL_ARCH = (
    MODEL_ARCH_ENV.strip().lower()
    if MODEL_ARCH_ENV
    else ("efficientnet_b0" if TRAINING_CONFIG.profile == "fast" else "efficientnet_b3")
)


def ensure_runtime_directories() -> None:
    """Create every runtime directory used by the project."""

    for directory in (
        MODELS_DIR,
        STATIC_DIR,
        TEMPLATES_DIR,
        DATABASE_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        EVALUATION_LOGS_DIR,
        UPLOADS_DIR,
        GRADCAM_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def serializable_project_config() -> dict[str, object]:
    """Return a JSON-friendly snapshot of the project configuration."""

    return {
        "project_root": str(PROJECT_ROOT),
        "runtime_root": str(RUNTIME_ROOT) if RUNTIME_ROOT is not None else None,
        "dataset_root": str(DATASET_ROOT),
        "train_dataset_dir": str(TRAIN_DATASET_DIR),
        "valid_dataset_dir": str(VALID_DATASET_DIR),
        "test_dataset_dir": str(TEST_DATASET_DIR),
        "models_dir": str(MODELS_DIR),
        "database_path": str(DATABASE_PATH),
        "database_uri": DATABASE_URI,
        "database_backend": DATABASE_BACKEND,
        "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
        "latest_checkpoint_path": str(LATEST_CHECKPOINT_PATH),
        "evaluation_logs_dir": str(EVALUATION_LOGS_DIR),
        "class_names": list(CLASS_NAMES),
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "training_profile": TRAINING_CONFIG.profile,
        "model_arch": MODEL_ARCH,
        "training": {
            **asdict(TRAINING_CONFIG),
            "device": str(TRAINING_CONFIG.device),
        },
        "augmentation": asdict(AUGMENTATION_CONFIG),
        "flask": {
            "secret_key": FLASK_CONFIG.secret_key,
            "max_content_length": FLASK_CONFIG.max_content_length,
            "upload_folder": str(FLASK_CONFIG.upload_folder),
            "gradcam_folder": str(FLASK_CONFIG.gradcam_folder),
            "reports_folder": str(FLASK_CONFIG.reports_folder),
            "database_uri": FLASK_CONFIG.database_uri,
            "allowed_extensions": list(FLASK_CONFIG.allowed_extensions),
        },
    }


ensure_runtime_directories()
