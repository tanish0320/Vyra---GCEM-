"""Build a metadata-only GitHub source ZIP after checking the recorded evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=package.parent / "person1-github-ready.zip")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip" or args.output.exists():
        raise SystemExit("Choose a new .zip output path; existing archives are never overwritten")
    evidence = json.loads((package / "reports/verification/package_validation.json").read_text())
    if not evidence["passed"]:
        raise SystemExit("Package verification has not passed")
    current = {p.name: digest(p) for p in sorted((package / "src/vyra_preprocessing").glob("*.py"))}
    if current != evidence["core_source_sha256"]:
        raise SystemExit("Source changed after verification; rerun package verification first")
    if digest(package / "requirements-tested-py314.txt") != evidence["dependency_file_sha256"]:
        raise SystemExit("Tested dependency file changed after verification")
    reports = package / "reports/observed-data"
    run = json.loads((reports / "run.json").read_text())
    observed = json.loads((reports / "real_data_validation.json").read_text())
    if not observed["passed"] or observed["preprocessing_fingerprint"] != evidence["preprocessing_fingerprint"] or run["preprocessing_fingerprint"] != evidence["preprocessing_fingerprint"]:
        raise SystemExit("Data and package verification fingerprints differ")
    for name, expected in run["artifact_sha256"].items():
        if digest(reports / name) != expected:
            raise SystemExit("Exported report changed after verification: " + name)
    excluded = {"data", ".venv", "build", "dist", "__pycache__", ".pytest_cache", ".git"}
    allowed = {".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".json", ".csv", ".in"}
    selected = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if not path.is_file() or any(p in excluded or p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.suffix.lower() in allowed or path.name == ".gitignore":
            selected.append(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in selected:
            entry = ZipInfo(path.relative_to(package).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    print(json.dumps({"archive": str(args.output), "files": len(selected),
                      "bytes": args.output.stat().st_size, "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
