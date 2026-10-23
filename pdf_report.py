"""Professional PDF report generation for AnemiaVision AI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from config import REPORTS_DIR
from predict import medical_advice_lines
from storage_utils import resolve_storage_path


PRIMARY_TEAL = colors.HexColor("#0A6E6E")
ACCENT_RED = colors.HexColor("#E63946")
ACCENT_GREEN = colors.HexColor("#2DC653")
TEXT_NAVY = colors.HexColor("#0D1B2A")
CARD_BORDER = colors.HexColor("#D6E2E9")


def _safe_text(value: Any, fallback: str = "N/A") -> str:
    """Convert arbitrary values into printable PDF strings."""

    if value in (None, ""):
        return fallback
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _chunk_text(value: str, width: int = 82) -> list[str]:
    """Wrap long content into compact PDF-safe lines."""

    lines: list[str] = []
    for paragraph in value.splitlines() or [""]:
        wrapped = wrap(paragraph.strip(), width=width) or [""]
        lines.extend(wrapped)
    return lines


def _draw_header(pdf: canvas.Canvas, page_width: float, page_height: float) -> None:
    """Render the branded report header with a simple medical icon."""

    pdf.setFillColor(PRIMARY_TEAL)
    pdf.rect(0, page_height - 80, page_width, 80, fill=1, stroke=0)

    icon_x = 42
    icon_y = page_height - 40
    pdf.setStrokeColor(colors.white)
    pdf.setLineWidth(2)
    pdf.ellipse(icon_x - 16, icon_y - 8, icon_x + 16, icon_y + 8, fill=0, stroke=1)
    pdf.circle(icon_x, icon_y, 5, stroke=1, fill=1)
    pdf.line(icon_x + 26, icon_y + 8, icon_x + 38, icon_y + 20)
    pdf.circle(icon_x + 44, icon_y + 24, 5, stroke=1, fill=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(90, page_height - 43, "AnemiaVision AI")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(90, page_height - 60, "Hospital-grade anemia screening report")


def _draw_patient_table(pdf: canvas.Canvas, x: int, y: int, record: Any) -> int:
    """Draw the patient information card and return the next y position."""

    card_height = 102
    pdf.setFillColor(colors.white)
    pdf.setStrokeColor(CARD_BORDER)
    pdf.roundRect(x, y - card_height, 520, card_height, 12, fill=1, stroke=1)

    pdf.setFillColor(TEXT_NAVY)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(x + 16, y - 22, "Patient Information")

    fields = [
        ("Scan ID", _safe_text(getattr(record, "public_scan_id", getattr(record, "id", "N/A")))),
        ("Timestamp", _safe_text(getattr(record, "created_at", None))),
        ("Patient Name", _safe_text(getattr(record, "patient_name", None))),
        ("Age", _safe_text(getattr(record, "age", None))),
        ("Gender", _safe_text(getattr(record, "gender", None))),
        ("Phone", _safe_text(getattr(record, "phone", None))),
        ("Image Type", _safe_text(getattr(record, "image_type", None), fallback="Unknown")),
        ("Risk Level", _safe_text(getattr(record, "risk_level", None))),
    ]

    pdf.setFont("Helvetica", 10)
    left_x = x + 16
    right_x = x + 275
    row_y = y - 42
    for index in range(0, len(fields), 2):
        left_label, left_value = fields[index]
        right_label, right_value = fields[index + 1]
        pdf.drawString(left_x, row_y, f"{left_label}: {left_value}")
        pdf.drawString(right_x, row_y, f"{right_label}: {right_value}")
        row_y -= 18

    return y - card_height - 16


def _draw_verdict_card(pdf: canvas.Canvas, x: int, y: int, record: Any) -> int:
    """Render the primary prediction summary card."""

    prediction = _safe_text(getattr(record, "prediction", None))
    confidence = float(getattr(record, "confidence", 0.0) or 0.0)
    card_color = ACCENT_RED if prediction == "Anemic" else ACCENT_GREEN

    pdf.setFillColor(card_color)
    pdf.roundRect(x, y - 74, 520, 74, 12, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(x + 16, y - 28, prediction.upper())
    pdf.setFont("Helvetica", 11)
    pdf.drawString(x + 16, y - 48, f"Model confidence: {confidence * 100:.1f}%")

    return y - 90


def _draw_confidence_bar(pdf: canvas.Canvas, x: int, y: int, confidence: float) -> None:
    """Draw a simple horizontal confidence bar chart."""

    width = 250
    height = 14
    fill_width = max(0.0, min(1.0, confidence)) * width
    fill_color = ACCENT_RED if confidence >= 0.70 else PRIMARY_TEAL

    pdf.setFillColor(colors.HexColor("#E8EFF3"))
    pdf.roundRect(x, y, width, height, 7, fill=1, stroke=0)
    pdf.setFillColor(fill_color)
    pdf.roundRect(x, y, fill_width, height, 7, fill=1, stroke=0)
    pdf.setFillColor(TEXT_NAVY)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(x + width + 12, y + 2, f"{confidence * 100:.1f}%")


def _draw_images(pdf: canvas.Canvas, x: int, y: int, record: Any) -> int:
    """Draw the original image and Grad-CAM image side by side when available."""

    image_width = 238
    image_height = 158
    original_path = resolve_storage_path(getattr(record, "image_path", None), "uploads")
    gradcam_path = (
        resolve_storage_path(getattr(record, "gradcam_path", None), "gradcam")
        if getattr(record, "gradcam_path", None)
        else None
    )

    pdf.setFont("Helvetica-Bold", 13)
    pdf.setFillColor(TEXT_NAVY)
    pdf.drawString(x, y, "Image Review")
    y -= 12

    slots = [
        (original_path, x, "Original scan"),
        (gradcam_path, x + image_width + 24, "Grad-CAM explainability"),
    ]

    for path, slot_x, label in slots:
        pdf.setStrokeColor(CARD_BORDER)
        pdf.roundRect(slot_x, y - image_height, image_width, image_height, 10, fill=0, stroke=1)
        if path and path.exists():
            try:
                pdf.drawImage(
                    ImageReader(str(path)),
                    slot_x + 6,
                    y - image_height + 6,
                    width=image_width - 12,
                    height=image_height - 28,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto",
                )
            except Exception:
                pdf.setFont("Helvetica", 10)
                pdf.drawString(slot_x + 12, y - 36, "Image preview unavailable")
        else:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(slot_x + 12, y - 36, "Image preview unavailable")

        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(TEXT_NAVY)
        pdf.drawString(slot_x + 10, y - image_height - 12, label)

    return y - image_height - 28


def _draw_text_section(pdf: canvas.Canvas, x: int, y: int, title: str, lines: list[str]) -> int:
    """Render a titled text block and return the next y position."""

    pdf.setFillColor(TEXT_NAVY)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(x, y, title)
    y -= 18

    pdf.setFont("Helvetica", 10)
    for line in lines:
        pdf.drawString(x + 8, y, f"- {line}")
        y -= 14
    return y - 8


def generate_pdf_report(record: Any, output_path: str | Path | None = None) -> Path:
    """Generate a production-ready screening report PDF with robust fallbacks."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(output_path).resolve() if output_path else REPORTS_DIR / f"anemiavision_report_{getattr(record, 'id', 'scan')}.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(report_path), pagesize=A4)
    page_width, page_height = A4

    _draw_header(pdf, page_width, page_height)

    cursor_y = page_height - 102
    cursor_y = _draw_patient_table(pdf, 36, cursor_y, record)
    cursor_y = _draw_verdict_card(pdf, 36, cursor_y, record)

    pdf.setFillColor(TEXT_NAVY)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(36, cursor_y, "Prediction Summary")
    cursor_y -= 22

    pdf.setFont("Helvetica", 10)
    pdf.drawString(36, cursor_y, f"Anemic probability: {float(getattr(record, 'anemic_probability', 0.0) or 0.0) * 100:.1f}%")
    pdf.drawString(280, cursor_y, f"Non-anemic probability: {float(getattr(record, 'non_anemic_probability', 0.0) or 0.0) * 100:.1f}%")
    cursor_y -= 18
    pdf.drawString(36, cursor_y, f"Hemoglobin estimate: {_safe_text(getattr(record, 'hemoglobin_estimate', None))}")
    cursor_y -= 22
    _draw_confidence_bar(pdf, 36, cursor_y, float(getattr(record, "confidence", 0.0) or 0.0))
    cursor_y -= 34

    cursor_y = _draw_images(pdf, 36, cursor_y, record)

    advice = getattr(record, "medical_advice", None)
    if isinstance(advice, str) and advice.strip():
        advice_lines = _chunk_text(advice, width=90)
    elif isinstance(advice, list) and advice:
        advice_lines = [str(item) for item in advice]
    else:
        advice_lines = medical_advice_lines(_safe_text(getattr(record, "prediction", None)))
    cursor_y = _draw_text_section(pdf, 36, cursor_y, "Medical Advice", advice_lines)

    notes = _safe_text(getattr(record, "notes", None), fallback="")
    if notes:
        cursor_y = _draw_text_section(pdf, 36, cursor_y, "Clinical Notes", _chunk_text(notes, width=90)[:4])

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColor(TEXT_NAVY)
    pdf.drawString(
        36,
        28,
        "For screening purposes only. Consult a doctor for diagnosis, treatment, and laboratory confirmation.",
    )
    pdf.save()
    return report_path
