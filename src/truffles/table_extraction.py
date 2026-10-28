"""Optional table-layout adapter. Tesseract recognizes text; img2table finds cells."""

import csv
import io
import json
import math
import sys
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .processing import (
    _PDF_LOCK,
    ProcessingLimits,
    _dependency,
    _ocr_options,
    _prepare_image,
    _read,
    _rotation_options,
)


@dataclass(frozen=True)
class TableResult:
    """Candidate table, always requiring review. Coordinates are corrected-image pixels."""

    rows: tuple[tuple[str | None, ...], ...]
    bbox: tuple[int, int, int, int]
    source: str
    page: int
    metadata: dict
    warnings: tuple[str, ...]
    native: Any = field(repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "rows": [list(row) for row in self.rows],
            "bbox": list(self.bbox),
            "source": self.source,
            "page": self.page,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "needs_review": True,
        }

    def to_csv(self) -> str:
        """Export raw OCR cell strings; the first row remains data, not an inferred header."""
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(self.rows)
        return stream.getvalue()

    def to_document(self) -> Document:
        return Document(
            page_content=json.dumps(self.to_dict(), ensure_ascii=False),
            metadata={
                "source": self.source,
                "page": self.page,
                "processor": "table_extraction",
                "needs_review": True,
                "bbox": list(self.bbox),
                **self.metadata,
            },
        )


def _engine(language, timeout, psm, remove_lines):
    native = _dependency("img2table.ocr", "tables")
    tesseract = _dependency("pytesseract", "tables")

    class TimedTesseract(native.TesseractOCR):
        """Reuse native hOCR parsing with pytesseract subprocess timeout and cleanup."""

        def __init__(self):
            # of() uses n_threads and hocr(). Initialization/engine errors come from pytesseract.
            self.n_threads = 1

        def hocr(self, image):
            if remove_lines:
                cv = _dependency("cv2", "tables")
                gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
                horizontal = cv.morphologyEx(
                    binary,
                    cv.MORPH_OPEN,
                    cv.getStructuringElement(cv.MORPH_RECT, (max(40, image.shape[1] // 20), 1)),
                )
                vertical = cv.morphologyEx(
                    binary,
                    cv.MORPH_OPEN,
                    cv.getStructuringElement(cv.MORPH_RECT, (1, max(40, image.shape[0] // 20))),
                )
                mask = cv.dilate(
                    horizontal | vertical, cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
                )
                image = image.copy()
                image[mask > 0] = 255
            return tesseract.image_to_pdf_or_hocr(
                image, extension="hocr", lang=language, config=f"--psm {psm}", timeout=timeout
            ).decode("utf-8")

    return TimedTesseract()


def extract_tables(
    source: str | Path | bytes,
    *,
    language: str = "eng",
    psm: int = 3,
    remove_lines: bool = True,
    rotation: str | int | None = "auto",
    min_orientation_confidence: float = 10,
    deskew: bool = False,
    borderless: bool = False,
    implicit_rows: bool = False,
    implicit_columns: bool = False,
    min_confidence: int = 50,
    timeout: float = 30,
    dpi: int = 200,
    password: str | None = None,
    limits: ProcessingLimits | None = None,
    max_tables: int = 50,
    max_cells: int = 10000,
) -> list[TableResult]:
    """Extract candidate tables from image/PDF input using local layout analysis and OCR.

    Table structure is heuristic, not certified by OCR confidence. Empty results do
    not prove the input has no tables. The optional tables extra supports Python <3.15.
    timeout bounds each OSD/hOCR subprocess, not layout processing or total runtime.
    """
    limits = limits or ProcessingLimits()
    _ocr_options(language, psm, timeout)
    _rotation_options(rotation, min_orientation_confidence)
    for value in (max_tables, max_cells):
        if type(value) is not int or value < 1:
            raise ValueError("max_tables and max_cells must be positive integers.")
    if type(min_confidence) is not int or not 0 <= min_confidence <= 99:
        raise ValueError("min_confidence must be an integer from 0 to 99.")
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer from 72 to 600.")
    if sys.version_info >= (3, 15):
        raise RuntimeError("The optional table backend requires Python 3.11-3.14.")
    data, name = _read(source, limits)
    native = _dependency("img2table.document", "tables")
    pil = _dependency("PIL.Image", "tables")
    engine = _engine(language, timeout, psm, remove_lines)
    results = []
    cells = chars = 0

    def process(image, page_number):
        nonlocal cells, chars
        prepared, orientation = _prepare_image(
            image,
            rotation=rotation,
            min_orientation_confidence=min_orientation_confidence,
            timeout=timeout,
            preprocess=True,
            limits=limits,
        )
        with prepared:
            # The native Tesseract hOCR parser expects at most four-digit pixel coordinates.
            if max(prepared.size) >= 10000:
                raise ValueError(
                    "Table images must be below 10000 pixels on each axis; reduce dpi."
                )
            buffer = io.BytesIO()
            prepared.save(buffer, format="PNG")
        document = native.Image(src=buffer.getvalue(), detect_rotation=deskew)
        for pixels in document.images:
            height, width = pixels.shape[:2]
            if height * width > limits.max_pixels or max(height, width) >= 10000:
                raise ValueError("Deskewed table image exceeds max_pixels or axis limit.")
        candidates = document.extract_tables(
            ocr=engine,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            borderless_tables=borderless,
            min_confidence=min_confidence,
            max_workers=1,
        )
        if len(results) + len(candidates) > max_tables:
            raise ValueError("Extraction exceeds max_tables.")
        for candidate in candidates:
            rows = tuple(tuple(cell.value for cell in row) for row in candidate.content.values())
            cells += sum(len(row) for row in rows)
            chars += sum(len(value or "") for row in rows for value in row)
            if cells > max_cells or chars > limits.max_chars:
                raise ValueError("Table extraction exceeds max_cells or max_chars.")
            warnings = ["table_structure_unverified"]
            if orientation["orientation_status"] in {"low_confidence", "unavailable"}:
                warnings.append("page_orientation_unverified")
            if borderless:
                warnings.append("borderless_detection_enabled")
            box = candidate.bbox
            results.append(
                TableResult(
                    rows=rows,
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    source=name,
                    page=page_number,
                    metadata={
                        **orientation,
                        "deskew": deskew,
                        "coordinate_space": "processed_image",
                        "min_word_confidence": min_confidence,
                        "psm": psm,
                        "remove_lines": remove_lines,
                    },
                    warnings=tuple(warnings),
                    native=candidate,
                )
            )

    is_pdf = (
        not isinstance(source, bytes) and Path(source).suffix.lower() == ".pdf"
    ) or data.startswith(b"%PDF-")
    if is_pdf:
        pdfium = _dependency("pypdfium2", "pdf")
        with _PDF_LOCK, closing(pdfium.PdfDocument(data, password=password)) as pdf:
            if len(pdf) > limits.max_pages:
                raise ValueError("PDF exceeds max_pages.")
            for index in range(len(pdf)):
                with closing(pdf[index]) as page:
                    width, height = page.get_size()
                    scale = dpi / 72
                    if math.ceil(width * scale) * math.ceil(height * scale) > limits.max_pixels:
                        raise ValueError("Rendered PDF page exceeds max_pixels; reduce dpi.")
                    with closing(page.render(scale=scale)) as bitmap:
                        with bitmap.to_pil() as image:
                            process(image, index + 1)
    else:
        with pil.open(io.BytesIO(data)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Split multi-frame images into individual pages first.")
            process(image, 1)
    return results
