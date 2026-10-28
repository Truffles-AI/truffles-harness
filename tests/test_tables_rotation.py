"""Rotation contracts, bounded table extraction, and real local OCR integration."""

import io
import json
import shutil
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from truffles import ProcessingLimits, extract_tables, ocr, processing_tools


def picture(image=None):
    image = image or Image.new("RGB", (120, 80), "white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_auto_and_manual_rotation(monkeypatch, angle):
    import pytesseract

    calls = []
    monkeypatch.setattr(
        pytesseract, "image_to_osd", lambda *a, **kw: {"rotate": angle, "orientation_conf": 20}
    )
    monkeypatch.setattr(
        pytesseract, "image_to_string", lambda im, **kw: calls.append(im.size) or "text"
    )
    for mode in ("auto", angle):
        result = ocr(picture(), rotation=mode)
        assert calls[-1] == ((80, 120) if angle in (90, 270) else (120, 80))
        assert result.metadata["rotation_clockwise"] == angle


@pytest.mark.parametrize("failure", ["low", "timeout", "osd", "disabled"])
def test_uncertain_orientation_is_explicit(monkeypatch, failure):
    import pytesseract

    def detect(*a, **kw):
        if failure == "timeout":
            raise RuntimeError("timeout")
        if failure == "osd":
            raise pytesseract.TesseractError(1, "Too few characters")
        if failure == "disabled":
            pytest.fail("OSD must not be called")
        return {"rotate": 90, "orientation_conf": 1}

    monkeypatch.setattr(pytesseract, "image_to_osd", detect)
    monkeypatch.setattr(pytesseract, "image_to_string", lambda im, **kw: "text")
    result = ocr(picture(), rotation=None if failure == "disabled" else "auto")
    assert result.metadata["rotation_clockwise"] == 0
    assert (
        result.metadata["orientation_status"]
        == {
            "low": "low_confidence",
            "disabled": "disabled",
            "timeout": "unavailable",
            "osd": "unavailable",
        }[failure]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"rotation": 45},
        {"rotation": True},
        {"min_orientation_confidence": -1},
        {"min_orientation_confidence": float("nan")},
        {"min_confidence": 100},
        {"max_tables": 0},
        {"max_cells": True},
        {"dpi": 20},
    ],
)
def test_table_option_validation(options):
    with pytest.raises(ValueError):
        extract_tables(b"unused", **options)


@pytest.fixture
def native_tables(monkeypatch):
    from img2table.document import Image as NativeImage

    calls = []
    candidate = SimpleNamespace(
        content={
            0: [SimpleNamespace(value="Product"), SimpleNamespace(value="Amount")],
            1: [SimpleNamespace(value="Apples"), SimpleNamespace(value="120")],
        },
        bbox=SimpleNamespace(x1=1, y1=2, x2=110, y2=70),
    )

    def extract(self, **kwargs):
        calls.append(kwargs)
        return [candidate]

    monkeypatch.setattr(NativeImage, "extract_tables", extract)
    return calls, candidate


def test_table_contract_exports_and_agent_tool(tmp_path, native_tables):
    calls, candidate = native_tables
    path = tmp_path / "table.png"
    path.write_bytes(picture())
    table = extract_tables(
        path, rotation=None, borderless=True, implicit_rows=True, implicit_columns=True
    )[0]
    assert table.native is candidate
    assert table.rows == (("Product", "Amount"), ("Apples", "120"))
    assert table.to_csv() == "Product,Amount\r\nApples,120\r\n"
    assert table.to_dict()["needs_review"] is True
    assert "borderless_detection_enabled" in table.warnings
    assert json.loads(table.to_document().page_content)["rows"][1] == ["Apples", "120"]
    assert (
        calls[0]["borderless_tables"] and calls[0]["implicit_rows"] and calls[0]["implicit_columns"]
    )
    tool = next(
        t
        for t in processing_tools(tmp_path, tables=True, rotation=None)
        if t.name == "extract_file_tables"
    )
    message = tool.invoke(
        {"name": tool.name, "args": {"file": "table.png"}, "id": "t1", "type": "tool_call"}
    )
    assert message.artifact[0].metadata["source"] == "table.png"
    assert message.artifact[0].metadata["needs_review"]
    (tmp_path.parent / "outside.png").write_bytes(picture())
    with pytest.raises(ValueError):
        tool.invoke({"file": "../outside.png"})


def test_table_output_and_input_limits(native_tables, tmp_path):
    for options in (
        {"max_cells": 1},
        {"limits": ProcessingLimits(max_chars=1)},
        {"limits": ProcessingLimits(max_pixels=1)},
    ):
        with pytest.raises(ValueError):
            extract_tables(picture(), rotation=None, **options)
    with pytest.raises(ValueError, match="10000"):
        extract_tables(picture(Image.new("RGB", (10000, 10))), rotation=None)
    buffer = io.BytesIO()
    with Image.new("RGB", (120, 80), "white") as image:
        image.save(buffer, format="PDF", save_all=True, append_images=[image])
    pdf = buffer.getvalue()
    assert [t.page for t in extract_tables(pdf, rotation=None, dpi=72)] == [1, 2]
    with pytest.raises(ValueError, match="max_tables"):
        extract_tables(pdf, rotation=None, max_tables=1)
    with pytest.raises(ValueError, match="max_pages"):
        extract_tables(pdf, rotation=None, limits=ProcessingLimits(max_pages=1))
    with pytest.raises(ValueError, match="max_pixels"):
        extract_tables(pdf, rotation=None, limits=ProcessingLimits(max_pixels=1))
    buffer = io.BytesIO()
    with Image.new("RGB", (100, 100)) as a, Image.new("RGB", (100, 100), "white") as b:
        a.save(buffer, format="TIFF", save_all=True, append_images=[b])
    with pytest.raises(ValueError, match="multi-frame"):
        extract_tables(buffer.getvalue(), rotation=None)


def test_table_ocr_contract(monkeypatch):
    import pytesseract

    from truffles.table_extraction import _engine

    calls = []
    monkeypatch.setattr(
        pytesseract, "image_to_pdf_or_hocr", lambda im, **kw: calls.append(kw) or b"<html/>"
    )
    assert _engine("eng+hin", 7, 6, False).hocr(object()) == "<html/>"
    assert calls == [{"extension": "hocr", "lang": "eng+hin", "config": "--psm 6", "timeout": 7}]


def synthetic_page():
    image = Image.new("RGB", (1600, 1800), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=30)
    lines = [
        "Truffles document processing verifies the orientation of this printed page.",
        "Invoices contain product descriptions quantities prices and payment terms.",
        "Review the original document before accepting extracted financial values.",
        "This synthetic sample is used to test reliable text recognition locally.",
    ]
    for i in range(15):
        draw.text((50, 40 + i * 55), lines[i % 4], font=font, fill="black")
    xs, ys = [60, 650, 950, 1450], [1000, 1110, 1220, 1330]
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=4)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=4)
    for row, values in enumerate(
        [["Product", "Quantity", "Amount"], ["Apples", "3", "120"], ["Bananas", "5", "200"]]
    ):
        for col, value in enumerate(values):
            draw.text((xs[col] + 25, ys[row] + 35), value, font=font, fill="black")
    return image


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Requires local Tesseract")
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_real_rotated_pages_and_table_cells(angle):
    with synthetic_page() as original, original.rotate(angle, expand=True) as rotated:
        data = picture(rotated)
        doc = ocr(data)
        assert doc.metadata["rotation_clockwise"] == angle
        assert "Truffles document processing" in doc.page_content
        tables = extract_tables(data)
        assert len(tables) == 1
        assert tables[0].rows == (
            ("Product", "Quantity", "Amount"),
            ("Apples", "3", "120"),
            ("Bananas", "5", "200"),
        )
        assert tables[0].metadata["rotation_clockwise"] == angle
        assert tables[0].to_dict()["needs_review"]


def test_exif_precedes_detection_and_bad_osd_propagates(monkeypatch):
    import pytesseract

    with Image.new("RGB", (120, 80), "white") as image:
        exif = image.getexif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)

    def detect(image, **kwargs):
        assert image.size == (80, 120)
        return {"rotate": 0, "orientation_conf": 20}

    monkeypatch.setattr(pytesseract, "image_to_osd", detect)
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **kw: "text")
    assert ocr(stream.getvalue()).metadata["exif_transposed"]
    monkeypatch.setattr(
        pytesseract, "image_to_osd", lambda *a, **kw: {"rotate": 45, "orientation_conf": 20}
    )
    with pytest.raises(ValueError, match="Invalid Tesseract"):
        ocr(picture())

    def missing(*args, **kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_osd", missing)
    with pytest.raises(RuntimeError, match="Tesseract executable"):
        ocr(picture())


def test_table_uncertainty_and_backend_constraints(monkeypatch, native_tables):
    import numpy as np
    import pytesseract
    from img2table.document import Image as NativeImage

    import truffles.table_extraction as module

    monkeypatch.setattr(
        pytesseract, "image_to_osd", lambda *a, **kw: {"rotate": 90, "orientation_conf": 1}
    )
    assert "page_orientation_unverified" in extract_tables(picture())[0].warnings
    monkeypatch.setattr(NativeImage, "images", property(lambda self: [np.zeros((10, 10000, 3))]))
    with pytest.raises(ValueError, match="Deskewed"):
        extract_tables(picture(), rotation=None, deskew=True)
    monkeypatch.setattr(module.sys, "version_info", (3, 15))
    with pytest.raises(RuntimeError, match="3.11-3.14"):
        extract_tables(picture())
