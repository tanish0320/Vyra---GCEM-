"""Extraction into a new staging directory; no writes to source archives."""

from pathlib import Path, PurePosixPath
import hashlib
import shutil
import stat
from zipfile import ZipFile


def safe_destination(root: Path, name: str) -> Path:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or "\x00" in normalized or path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"Unsafe archive entry: {name!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive entry escapes extraction directory: {name!r}")
    return target


def extract_zip(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as archive:
        for entry in archive.infolist():
            target = safe_destination(destination, entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"Duplicate archive path: {entry.filename}")
                with archive.open(entry) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)


def extract_rar(source: Path, destination: Path) -> None:
    try:
        import libarchive
    except (ImportError, OSError) as exc:
        raise RuntimeError("RAR extraction requires the 'data' extra and the system libarchive library; alternatively pass --nails with an extracted directory") from exc
    destination.mkdir(parents=True, exist_ok=True)
    with libarchive.file_reader(str(source)) as archive:
        for entry in archive:
            target = safe_destination(destination, entry.pathname)
            if entry.isdir:
                target.mkdir(parents=True, exist_ok=True)
            elif entry.isfile and not entry.issym and not entry.islnk:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f"RAR path collides with a directory: {entry.pathname}")
                    expected = hashlib.sha256()
                    for block in entry.get_blocks():
                        expected.update(block)
                    actual = hashlib.sha256(target.read_bytes()).digest()
                    if expected.digest() != actual:
                        raise ValueError(f"RAR path conflicts with an existing extracted file: {entry.pathname}")
                else:
                    with target.open("xb") as stream:
                        for block in entry.get_blocks():
                            stream.write(block)
            else:
                raise ValueError(f"Unsupported RAR entry: {entry.pathname}")


def stage_source(source: Path, destination: Path, *, nails: bool = False) -> None:
    if source.is_dir():
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Dataset directory must not contain symlinks")
        shutil.copytree(source, destination)
    elif source.is_file() and source.suffix.lower() == ".zip":
        extract_zip(source, destination)
    elif nails and source.is_file() and source.suffix.lower() == ".rar":
        extract_rar(source, destination)
    else:
        raise ValueError(f"Expected a dataset ZIP, extracted directory or nail RAR: {source}")
    nested = sorted(destination.rglob("*.rar"))
    for rar in nested:
        extract_rar(rar, destination)

