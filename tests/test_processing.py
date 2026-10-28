"""Local OCR/PDF integration and native audio contract tests. No live services."""

import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from langchain_core.messages import AIMessage
from PIL import Image, ImageDraw, ImageFont

from truffles import (
    Observer,
    ProcessingLimits,
    agent,
    ocr,
    processing_tools,
    rag,
    read_documents,
    speech_model,
    transcribe,
)
from truffles.testing import ScriptedChatModel


def image_bytes(text="TRUFFLES INVOICE 12345", *, format="PNG"):
    with Image.new("RGB", (1100, 200), "white") as image:
        ImageDraw.Draw(image).text(
            (35, 60), text, fill="black", font=ImageFont.load_default(size=48)
        )
        stream = io.BytesIO()
        image.save(stream, format=format)
        return stream.getvalue()


def pdf_bytes(*pages):
    """Minimal real PDF with text/blank pages, no extra fixture-generation dependency."""
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(pages):
        content_id = 5 + index * 2
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream = f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    result = b"%PDF-1.4\n"
    offsets = [0]
    for number, item in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + item + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    result += f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref}\n%%EOF".encode()
    return result


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Install Tesseract for real OCR checks"
)
def test_real_tesseract_image_pdf_and_rag(tmp_path):
    image = tmp_path / "invoice.png"
    image.write_bytes(image_bytes())
    document = ocr(image, psm=6)
    assert "TRUFFLES" in document.page_content and "12345" in document.page_content
    assert document.metadata["processor"] == "tesseract"
    assert "INVOICE" in read_documents(image)[0].page_content
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(image_bytes(format="PDF"))
    assert read_documents(pdf, pdf_mode="text")[0].page_content == ""
    extracted = read_documents(pdf)
    assert "TRUFFLES" in extracted[0].page_content
    assert extracted[0].metadata["page"] == 1
    assert extracted[0].metadata["processor"] == "tesseract"
    assert "12345" in rag(extracted).retrieve("invoice")[0].page_content


def test_ocr_options_and_preprocessing(monkeypatch):
    import pytesseract

    calls = []

    def extract(image, **options):
        calls.append((image.mode, image.getpixel((0, 0)), options))
        return "  recognized  "

    monkeypatch.setattr(pytesseract, "image_to_string", extract)
    doc = ocr(image_bytes(), language="eng+hin", psm=6, timeout=3)
    assert doc.page_content == "recognized" and doc.metadata["source"] == "memory"
    assert calls[0] == ("L", 255, {"lang": "eng+hin", "config": "--psm 6", "timeout": 3})
    ocr(image_bytes(), preprocess=False)
    assert calls[-1][0] == "RGB"
    with pytest.raises(ValueError, match="max_pixels"):
        ocr(image_bytes(), limits=ProcessingLimits(max_pixels=5))
    with pytest.raises(ValueError, match="max_chars"):
        ocr(image_bytes(), limits=ProcessingLimits(max_chars=1))


def test_ocr_missing_binary_timeout_and_multiframe(monkeypatch):
    import pytesseract

    def missing(*args, **kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", missing)
    with pytest.raises(RuntimeError, match="Tesseract executable"):
        ocr(image_bytes())

    def timeout(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_string", timeout)
    with pytest.raises(RuntimeError, match="timeout"):
        ocr(image_bytes())
    data = io.BytesIO()
    with (
        Image.new("RGB", (10, 10), "white") as first,
        Image.new("RGB", (10, 10), "black") as second,
    ):
        first.save(data, format="TIFF", save_all=True, append_images=[second])
    with pytest.raises(ValueError, match="Multi-frame"):
        ocr(data.getvalue())
    with pytest.raises(OSError):
        ocr(b"not an image")


@pytest.mark.parametrize(
    "options",
    [{"language": "eng --output x"}, {"psm": 0}, {"timeout": float("inf")}, {"timeout": 0}],
)
def test_invalid_ocr_options(options):
    with pytest.raises(ValueError):
        ocr(b"unused", **options)


def test_text_file_limits_and_validation(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Unicode text: हिन्दी", encoding="utf-8")
    assert "हिन्दी" in read_documents(path)[0].page_content
    assert read_documents(path)[0].metadata["processor"] == "text"
    for options in [{"max_bytes": 1}, {"max_chars": 1}]:
        with pytest.raises(ValueError):
            read_documents(path, limits=ProcessingLimits(**options))
    for options in [{"max_bytes": 0}, {"max_pages": True}, {"max_pixels": -1}]:
        with pytest.raises(ValueError):
            ProcessingLimits(**options)
    for options in [{"pdf_mode": "unknown"}, {"dpi": 2}]:
        with pytest.raises(ValueError):
            read_documents(path, **options)
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_documents(path)
    assert read_documents(path, encoding="latin-1")[0].page_content == "ÿ"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        read_documents(path)
    with pytest.raises(ValueError):
        read_documents(tmp_path / "x.docx")
    with pytest.raises(ValueError):
        read_documents(tmp_path / "missing.txt")
    with pytest.raises(ValueError):
        ocr(b"x" * 4, limits=ProcessingLimits(max_bytes=2))


def test_pdf_text_auto_ocr_modes_and_page_metadata(tmp_path, monkeypatch):
    import pytesseract

    calls = []

    def extract(image, **kwargs):
        calls.append(image.size)
        return "OCR page"

    monkeypatch.setattr(pytesseract, "image_to_string", extract)
    path = tmp_path / "mixed.pdf"
    path.write_bytes(pdf_bytes("Hello Truffles", ""))
    docs = read_documents(path, pdf_mode="auto")
    assert [d.page_content for d in docs] == ["Hello Truffles", "OCR page"]
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert len(calls) == 1
    assert calls[0][0] > 1000
    assert [d.page_content for d in read_documents(path, pdf_mode="text")] == ["Hello Truffles", ""]
    assert len(calls) == 1
    assert [d.page_content for d in read_documents(path, pdf_mode="ocr")] == ["OCR page"] * 2
    assert len(calls) == 3
    for options in ({"max_pages": 1}, {"max_pixels": 5}, {"max_chars": 5}):
        with pytest.raises(ValueError):
            read_documents(path, limits=ProcessingLimits(**options))
    # Each OCR page fits individually, but the document total does not.
    with pytest.raises(ValueError, match="max_chars"):
        read_documents(path, pdf_mode="ocr", limits=ProcessingLimits(max_chars=10))


def test_pdf_concurrent_readers_are_serialized(tmp_path):
    path = tmp_path / "real.pdf"
    path.write_bytes(pdf_bytes("Thread safe extraction"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: read_documents(path, pdf_mode="text"), range(12)))
    assert all(docs[0].page_content == "Thread safe extraction" for docs in results)


def test_pdf_corrupt_and_pixel_limit(tmp_path):
    import pypdfium2

    path = tmp_path / "bad.pdf"
    path.write_bytes(b"corrupt")
    with pytest.raises(pypdfium2.PdfiumError):
        read_documents(path)
    path.write_bytes(pdf_bytes("Readable"))
    with pytest.raises(ValueError, match="max_pixels"):
        read_documents(path, pdf_mode="ocr", dpi=600, limits=ProcessingLimits(max_pixels=100))


def test_transcription_native_wire_and_owned_lifecycle(tmp_path, respx_mock):
    route = respx_mock.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "meeting transcript"})
    )
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"RIFF-synthetic-audio")
    doc = transcribe(path, language="en")
    assert doc.page_content == "meeting transcript"
    assert doc.metadata["source"] == str(path)
    request = route.calls.last.request
    assert b'filename="meeting.wav"' in request.content
    assert b"RIFF-synthetic-audio" in request.content and b"whisper-1" in request.content


def test_custom_transcription_borrowed_adapter_and_output_limits():
    class Custom:
        def transcribe(self, audio, **kwargs):
            assert audio == b"audio"
            assert kwargs["filename"] == "recording.wav"
            return "local transcript"

        def close(self):
            raise AssertionError("Borrowed adapter must stay open")

    doc = transcribe(b"audio", speech=Custom())
    assert doc.metadata["source"] == "recording.wav"
    with pytest.raises(ValueError, match="max_chars"):
        transcribe(b"audio", speech=Custom(), limits=ProcessingLimits(max_chars=1))
    with pytest.raises(ValueError, match="filename"):
        transcribe(b"audio", speech=Custom(), filename="../x.wav")
    with pytest.raises(ValueError, match="max_bytes"):
        transcribe(b"audio", speech=Custom(), limits=ProcessingLimits(max_bytes=1))


def test_owned_adapter_closed_on_error_and_nontext_rejected(monkeypatch):
    from truffles import processing

    class Custom:
        closed = False

        def transcribe(self, *args, **kwargs):
            return None

        def close(self):
            self.closed = True

    custom = Custom()
    monkeypatch.setattr(processing, "speech_model", lambda: custom)
    with pytest.raises(TypeError):
        transcribe(b"audio")
    assert custom.closed


def test_compatible_audio_endpoint(respx_mock):
    route = respx_mock.post("https://speech.example/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "compatible service"})
    )
    speech = speech_model(
        "openai", base_url="https://speech.example/v1", api_key="test", max_retries=0
    )
    try:
        assert (
            transcribe(b"audio", speech=speech, filename="clip.webm").page_content
            == "compatible service"
        )
        assert b'filename="clip.webm"' in route.calls.last.request.content
    finally:
        speech.close()


def test_scoped_tools_limits_artifacts_and_symlinks(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "notes.txt").write_text("Visible document contents")
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    (root / "escape.txt").symlink_to(outside)
    tools = processing_tools(root, max_tool_chars=7)
    assert [t.name for t in tools] == ["read_file", "ocr_file"]
    result = tools[0].invoke(
        {"name": "read_file", "args": {"file": "notes.txt"}, "id": "call-read", "type": "tool_call"}
    )
    assert json.loads(result.content) == {"text": "Visible", "truncated": True, "documents": 1}
    assert result.artifact[0].page_content == "Visible document contents"
    assert result.artifact[0].metadata["source"] == "notes.txt"
    for path in ("../private.txt", str(outside), "escape.txt", "."):
        with pytest.raises(ValueError):
            tools[0].invoke({"file": path})
    for kwargs in ({"max_tool_chars": 0}, {"pdf_mode": "wrong"}):
        with pytest.raises(ValueError):
            processing_tools(root, **kwargs)
    with pytest.raises(ValueError):
        processing_tools(outside)


@pytest.mark.asyncio
async def test_tools_async_agent_source_artifacts_and_trace(tmp_path):
    (tmp_path / "policy.txt").write_text("Support opens at nine.")
    tools = processing_tools(tmp_path)
    observer = Observer()
    bot = agent(
        ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "read-1", "name": "read_file", "args": {"file": "policy.txt"}}
                    ],
                ),
                AIMessage(content="Support opens at nine."),
            ]
        ),
        tools=tools,
    )
    result = await bot.arun("Read policy.txt", config={"callbacks": [observer]})
    assert result.sources[0].page_content == "Support opens at nine."
    assert result.sources[0].metadata["source"] == "policy.txt"
    assert observer.summary()["tool_calls"] == 1
    assert json.loads(await tools[0].ainvoke({"file": "policy.txt"}))["truncated"] is False


def test_ocr_and_audio_agent_tools(tmp_path, monkeypatch):
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **kw: "scanned text")
    (tmp_path / "scan.png").write_bytes(image_bytes())
    (tmp_path / "clip.wav").write_bytes(b"audio")

    class Speech:
        def transcribe(self, data, *, filename, language):
            assert filename == "clip.wav" and language == "en"
            return "spoken words"

    tools = {
        t.name: t for t in processing_tools(tmp_path, speech=Speech(), transcription_language="en")
    }
    assert json.loads(tools["ocr_file"].invoke({"file": "scan.png"}))["text"] == "scanned text"
    assert (
        json.loads(tools["transcribe_file"].invoke({"file": "clip.wav"}))["text"] == "spoken words"
    )


def test_optional_dependency_hints_do_not_mask_transitive_errors(monkeypatch):
    from truffles import processing

    def missing(name):
        raise ModuleNotFoundError("missing", name="PIL")

    monkeypatch.setattr(processing, "import_module", missing)
    with pytest.raises(ImportError, match="ocr.*extra"):
        processing._dependency("PIL.Image", "ocr")

    def transitive(name):
        raise ModuleNotFoundError("broken dependency", name="unrelated_dependency")

    monkeypatch.setattr(processing, "import_module", transitive)
    with pytest.raises(ModuleNotFoundError, match="broken dependency"):
        processing._dependency("PIL.Image", "ocr")
