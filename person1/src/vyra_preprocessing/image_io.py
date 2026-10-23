"""Non-destructive decoding with explicit RGB, alpha, EXIF and PNG handling."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class LoadedImage:
    rgb: np.ndarray
    alpha_mask: np.ndarray | None
    metadata: dict[str, Any]


def repair_png_profile_crc(data: bytes) -> tuple[bytes, bool]:
    """Repair only a bad iCCP CRC; every other invalid chunk is an error."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data, False
    result = bytearray(data[:8])
    pos, repaired, ended = 8, False, False
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        end = pos + length + 12
        if end > len(data):
            raise InputError("CORRUPT_IMAGE", "Truncated PNG chunk")
        tag = data[pos + 4:pos + 8]
        stored = struct.unpack_from(">I", data, end - 4)[0]
        actual = zlib.crc32(data[pos + 4:end - 4]) & 0xFFFFFFFF
        if stored != actual:
            if tag != b"iCCP":
                raise InputError("CORRUPT_IMAGE", f"Invalid PNG {tag.decode(errors='replace')} checksum")
            repaired = True
        result.extend(data[pos:end - 4])
        result.extend(struct.pack(">I", actual))
        pos = end
        if tag == b"IEND":
            ended = True
            break
    if not ended:
        raise InputError("CORRUPT_IMAGE", "PNG has no complete IEND chunk")
    result.extend(data[pos:])
    return (bytes(result), True) if repaired else (data, False)


def load_image(image: str | Path | Image.Image | np.ndarray) -> LoadedImage:
    metadata: dict[str, Any] = {"png_profile_crc_repaired": False, "exif_orientation": 1}
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise InputError("UNSUPPORTED_DTYPE", "NumPy input must be uint8 RGB; do not pass a normalized tensor")
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            raise InputError("GRAYSCALE_INPUT", "Color information is required")
        if image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
            raise InputError("INVALID_CHANNELS", "NumPy input must have shape H x W x 3 in RGB order")
        rgb = np.ascontiguousarray(image).copy()
        metadata.update({"source_mode": "RGB", "oriented_size": [rgb.shape[1], rgb.shape[0]]})
        return LoadedImage(rgb, None, metadata)
    try:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise InputError("UNREADABLE_IMAGE", "Image file is missing or is not a regular file")
            data, repaired = repair_png_profile_crc(path.read_bytes())
            metadata["png_profile_crc_repaired"] = repaired
            with Image.open(BytesIO(data)) as source:
                source.load()
                pil = source.copy()
        elif isinstance(image, Image.Image):
            image.load()
            exif = image.getexif()
            pil = image.copy()
            if exif:
                pil.info["exif"] = exif.tobytes()
        else:
            raise InputError("UNSUPPORTED_INPUT", "Use a path, PIL image or uint8 RGB array")
        if pil.mode in {"1", "L", "LA", "I", "F", "I;16", "I;16B", "I;16L"}:
            raise InputError("GRAYSCALE_INPUT", "Color information is required")
        metadata["source_mode"] = pil.mode
        metadata["source_size_before_orientation"] = list(pil.size)
        metadata["exif_orientation"] = int(pil.getexif().get(274, 1))
        pil = ImageOps.exif_transpose(pil)
        metadata["oriented_size"] = list(pil.size)
        alpha_mask = None
        if "A" in pil.getbands() or "transparency" in pil.info:
            rgba = pil.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"))
            if np.any(alpha < 255):
                alpha_mask = alpha >= 128
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            pil = pil.convert("RGB")
        return LoadedImage(np.asarray(pil, dtype=np.uint8).copy(), alpha_mask, metadata)
    except InputError:
        raise
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InputError("UNREADABLE_IMAGE", f"Unable to decode image ({type(exc).__name__})") from exc

