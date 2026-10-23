"""Safely inspect and extract CP-AnemiC V2 and EYES-DEFY-ANEMIA zip archives."""

import zipfile
import os
from pathlib import Path

def safe_extract(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    abs_target = target_dir.resolve()
    print(f"Extracting {zip_path.name} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            # Security check: prevent directory traversal
            member_path = (target_dir / member.filename).resolve()
            if not str(member_path).startswith(str(abs_target)):
                raise ValueError(f"Unsafe path detected: {member.filename}")
            z.extract(member, target_dir)
    print(f"Successfully extracted {zip_path.name}")

if __name__ == "__main__":
    cp_zip = Path("data/raw/cp_anemic/cp_anemic_v2.zip")
    cp_dest = Path("data/raw/cp_anemic")
    safe_extract(cp_zip, cp_dest)

    eda_zip = Path("data/raw/eyes_defy_anemia/eyes_defy_anemia.zip")
    eda_dest = Path("data/raw/eyes_defy_anemia")
    safe_extract(eda_zip, eda_dest)
    print("Done extracting archives.")
