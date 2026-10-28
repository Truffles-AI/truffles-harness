"""Optional document/audio processors and directory-scoped native agent tools."""

import io
import json
import math
import re
from contextlib import closing
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from .speech import speech_model

_PDF_LOCK = RLock()  # PDFium cannot be called concurrently, even for different documents.
_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
_TEXT_TYPES = {".txt", ".md", ".csv", ".json", ".log"}


@dataclass(frozen=True)
class ProcessingLimits:
    """Application bounds, not an OS sandbox or a native-decoder memory guarantee."""

    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 50
    max_pixels: int = 25_000_000
    max_chars: int = 200_000

    def __post_init__(self):
        for value in (self.max_bytes, self.max_pages, self.max_pixels, self.max_chars):
            if type(value) is not int or value < 1:
                raise ValueError("Processing limits must be positive integers.")


def _dependency(module, extra):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module.split(".")[0] or exc.name == module:
            raise ImportError(
                f"Install the '{extra}' extra from your Truffles wheel or source checkout."
            ) from exc
        raise


def _read(source, limits):
    if isinstance(source, bytes):
        data, name = source, "memory"
    else:
        path = Path(source)
        if not path.is_file():
            raise ValueError("Source must be a regular file.")
        if path.stat().st_size > limits.max_bytes:
            raise ValueError("File exceeds max_bytes.")
        with path.open("rb") as stream:
            data = stream.read(limits.max_bytes + 1)
        name = str(path)
    if not data or len(data) > limits.max_bytes:
        raise ValueError("Input must be nonempty and no larger than max_bytes.")
    return data, name


def _check_text(text, limits):
    if not isinstance(text, str):
        raise TypeError("Processor must return text.")
    if len(text) > limits.max_chars:
        raise ValueError("Extracted text exceeds max_chars.")
    return text


def _ocr_options(language, psm, timeout):
    if not isinstance(language, str) or not re.fullmatch(
        r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language
    ):
        raise ValueError("Use installed language codes such as 'eng' or 'eng+hin'.")
    if type(psm) is not int or not 3 <= psm <= 13:
        raise ValueError("psm must be an integer from 3 to 13.")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite.")


def _ocr_image(image, *, language, psm, timeout, preprocess, limits):
    if image.width * image.height > limits.max_pixels:
        raise ValueError("Image exceeds max_pixels.")
    tesseract = _dependency("pytesseract", "ocr")
    ops = _dependency("PIL.ImageOps", "ocr")
    pil = _dependency("PIL.Image", "ocr")
    # Work on an owned copy. Correct EXIF orientation, flatten alpha on white.
    with ops.exif_transpose(image) as oriented:
        with oriented.convert("RGBA") as rgba:
            with pil.new("RGBA", rgba.size, "white") as background:
                background.alpha_composite(rgba)
                prepared = (
                    ops.autocontrast(ops.grayscale(background))
                    if preprocess
                    else background.convert("RGB")
                )
                try:
                    text = tesseract.image_to_string(
                        prepared, lang=language, config=f"--psm {psm}", timeout=timeout
                    )
                except tesseract.TesseractNotFoundError as exc:
                    raise RuntimeError(
                        "Install the Tesseract executable and put it on PATH; "
                        "pip only installs its wrapper."
                    ) from exc
                finally:
                    prepared.close()
    return _check_text(text.strip(), limits)


def ocr(
    source: str | Path | bytes,
    *,
    language: str = "eng",
    psm: int = 3,
    timeout: float = 30,
    preprocess: bool = True,
    limits: ProcessingLimits | None = None,
) -> Document:
    """OCR one image with local Tesseract. Multi-frame images are rejected explicitly."""
    limits = limits or ProcessingLimits()
    _ocr_options(language, psm, timeout)
    data, name = _read(source, limits)
    pil = _dependency("PIL.Image", "ocr")
    with pil.open(io.BytesIO(data)) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(
                "Multi-frame images are unsupported; convert frames to individual images."
            )
        text = _ocr_image(
            image, language=language, psm=psm, timeout=timeout, preprocess=preprocess, limits=limits
        )
    return Document(
        page_content=text,
        metadata={
            "source": name,
            "processor": "tesseract",
            "language": language,
            "page": 1,
        },
    )


def _pdf(data, name, *, pdf_mode, language, psm, timeout, dpi, password, limits):
    pdfium = _dependency("pypdfium2", "pdf")
    documents = []
    chars = 0
    with _PDF_LOCK, closing(pdfium.PdfDocument(data, password=password)) as pdf:
        if len(pdf) > limits.max_pages:
            raise ValueError("PDF exceeds max_pages; no pages were processed.")
        for index in range(len(pdf)):
            with closing(pdf[index]) as page:
                text = ""
                processor = "pdfium"
                if pdf_mode != "ocr":
                    with closing(page.get_textpage()) as textpage:
                        if textpage.count_chars() + chars > limits.max_chars:
                            raise ValueError("Extracted text exceeds max_chars.")
                        text = textpage.get_text_bounded().replace("\r\n", "\n").strip()
                if pdf_mode == "ocr" or (pdf_mode == "auto" and not text):
                    width, height = page.get_size()
                    scale = dpi / 72
                    if math.ceil(width * scale) * math.ceil(height * scale) > limits.max_pixels:
                        raise ValueError("Rendered PDF page exceeds max_pixels; reduce dpi.")
                    with closing(page.render(scale=scale)) as bitmap:
                        with bitmap.to_pil() as image:
                            text = _ocr_image(
                                image,
                                language=language,
                                psm=psm,
                                timeout=timeout,
                                preprocess=True,
                                limits=limits,
                            )
                    processor = "tesseract"
                chars += len(text)
                if chars > limits.max_chars:
                    raise ValueError("Extracted text exceeds max_chars.")
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": name,
                            "page": index + 1,
                            "processor": processor,
                            **({"language": language} if processor == "tesseract" else {}),
                        },
                    )
                )
    return documents


def read_documents(
    path: str | Path,
    *,
    pdf_mode: str = "auto",
    language: str = "eng",
    psm: int = 3,
    timeout: float = 30,
    dpi: int = 200,
    password: str | None = None,
    encoding: str = "utf-8",
    limits: ProcessingLimits | None = None,
) -> list[Document]:
    """Read text, images or PDF pages into RAG-ready Documents (one-based page metadata).

    auto: use PDF text, OCR textless pages. text: never OCR. ocr: rasterize all pages.
    Text files are read literally, without parsing CSV/JSON or executing markup.
    """
    limits = limits or ProcessingLimits()
    if pdf_mode not in {"auto", "text", "ocr"}:
        raise ValueError("pdf_mode must be 'auto', 'text' or 'ocr'.")
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer from 72 to 600.")
    _ocr_options(language, psm, timeout)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _IMAGE_TYPES:
        return [ocr(path, language=language, psm=psm, timeout=timeout, limits=limits)]
    if suffix not in _TEXT_TYPES | {".pdf"}:
        raise ValueError("Unsupported file type; use PDF, a supported image or a plain-text file.")
    data, name = _read(path, limits)
    if suffix == ".pdf":
        return _pdf(
            data,
            name,
            pdf_mode=pdf_mode,
            language=language,
            psm=psm,
            timeout=timeout,
            dpi=dpi,
            password=password,
            limits=limits,
        )
    text = _check_text(data.decode(encoding), limits)
    return [Document(page_content=text, metadata={"source": name, "processor": "text"})]


def transcribe(
    source: str | Path | bytes,
    *,
    speech: Any = None,
    filename: str | None = None,
    language: str | None = None,
    limits: ProcessingLimits | None = None,
) -> Document:
    """Transcribe a bounded audio file via the speech adapter; default is OpenAI.

    Creating the default adapter is explicit network-capable work. A supplied adapter
    is borrowed and never closed. Native endpoint format/language restrictions apply.
    """
    limits = limits or ProcessingLimits()
    data, name = _read(source, limits)
    filename = filename or (Path(name).name if name != "memory" else "recording.wav")
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a file name without a directory.")
    owned = speech is None
    adapter = speech_model() if owned else speech_model(speech)
    try:
        text = _check_text(adapter.transcribe(data, filename=filename, language=language), limits)
    finally:
        if owned:
            adapter.close()
    return Document(
        page_content=text,
        metadata={
            "source": name if name != "memory" else filename,
            "processor": "transcription",
            **({"language": language} if language else {}),
        },
    )


def processing_tools(
    root: str | Path,
    *,
    speech: Any = None,
    language: str = "eng",
    transcription_language: str | None = None,
    limits: ProcessingLimits | None = None,
    max_tool_chars: int = 12000,
    pdf_mode: str = "auto",
) -> list[StructuredTool]:
    """Agent file-reading/OCR tools; add audio only when an adapter is supplied.

    Paths must resolve inside root. Use a dedicated trusted upload directory, not a
    home directory. This check is not a sandbox against concurrent filesystem changes.
    Full bounded Documents are retained as artifacts, while model-visible text is capped.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root must be a directory.")
    if type(max_tool_chars) is not int or max_tool_chars < 1:
        raise ValueError("max_tool_chars must be a positive integer.")
    limits = limits or ProcessingLimits()
    _ocr_options(language, 3, 30)
    if pdf_mode not in {"auto", "text", "ocr"}:
        raise ValueError("pdf_mode must be 'auto', 'text' or 'ocr'.")

    def resolve(file):
        supplied = Path(file)
        if supplied.is_absolute():
            raise ValueError("Use a relative file path inside the configured root.")
        path = (root / supplied).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("File must resolve to a regular file inside the configured root.")
        return path

    def result(documents, path):
        for document in documents:
            document.metadata["source"] = path.relative_to(root).as_posix()
        text = "\n\n".join(doc.page_content for doc in documents)
        content = json.dumps(
            {
                "text": text[:max_tool_chars],
                "truncated": len(text) > max_tool_chars,
                "documents": len(documents),
            },
            ensure_ascii=False,
        )
        return content, documents

    def read_file(file: str):
        """Read a PDF, image, TXT, Markdown, CSV or JSON file from the allowed directory."""
        path = resolve(file)
        return result(
            read_documents(path, pdf_mode=pdf_mode, language=language, limits=limits), path
        )

    def ocr_file(file: str):
        """Extract printed text from one image using local Tesseract OCR."""
        path = resolve(file)
        return result([ocr(path, language=language, limits=limits)], path)

    tools = [
        StructuredTool.from_function(fn, response_format="content_and_artifact")
        for fn in (read_file, ocr_file)
    ]
    if speech is not None:

        def transcribe_file(file: str):
            """Transcribe an allowed audio file using the configured speech service."""
            path = resolve(file)
            return result(
                [transcribe(path, speech=speech, language=transcription_language, limits=limits)],
                path,
            )

        tools.append(
            StructuredTool.from_function(transcribe_file, response_format="content_and_artifact")
        )
    return tools
