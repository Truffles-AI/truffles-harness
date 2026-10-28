# OCR, document processing and transcription

Basic processing is available since **0.2.0**; rotation detection and tables since **0.3.0**. These optional processors return native LangChain
`Document` objects, so their output can go directly into RAG or a custom workflow.
The core library gains no mandatory imaging, PDF or speech dependency.

## Install

For the released wheel:

```bash
BASE="https://github.com/Truffles-AI/truffles-harness"
WHEEL="$BASE/releases/download/v0.3.0/truffles-0.3.0-py3-none-any.whl"
python -m pip install "truffles[processing] @ $WHEEL"
```

For an editable checkout: `python -m pip install -e '.[processing]'`.
Use `ocr` for Pillow/pytesseract, `pdf` for PDFium, or `audio` for the speech SDK
when you only need one integration. `processing` includes all three.

**Tesseract is a separate system executable**, not a Python wheel dependency:

```bash
# macOS
brew install tesseract
# Debian / Ubuntu
sudo apt-get install tesseract-ocr
# Verify the executable and installed recognition languages
tesseract --version
tesseract --list-langs
```

Windows users should install a build referenced by the
[official Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
and add the executable directory to PATH. Additional language data is installed
separately: for example, `tesseract-ocr-hin` on Debian/Ubuntu or `tesseract-lang` on
Homebrew. No language model is downloaded automatically by Truffles.

PDF extraction/rendering uses the pypdfium2 wheel and does not require Poppler.
Audio transcription additionally needs a configured speech endpoint or custom
adapter. OCR and PDF extraction run locally and make no model API calls.

## Image OCR

```python
from truffles import ocr

page = ocr("invoice.png", language="eng", psm=6)
print(page.page_content)
print(page.metadata)
```

`ocr` accepts a file path or image bytes and returns one Document. Metadata contains
`source`, `processor="tesseract"`, `language` and `page=1`. In-memory input uses
`source="memory"`. Images receive EXIF orientation correction and alpha compositing
on white; default preprocessing also applies grayscale and auto-contrast. Set
`preprocess=False` to skip grayscale/contrast while keeping orientation/alpha handling.

Tesseract page segmentation mode (`psm`) defaults to 3; 6 is useful for a single
uniform text block. This wrapper accepts modes 3-13. `language="eng+hin"` requests
English and Hindi together, provided both language packs are installed. This is
printed-text OCR, not a guarantee of handwriting, table structure or scan quality.
Plain OCR returns text. Use the table API below for cell layout.

`timeout=30` bounds each Tesseract subprocess. Recognition errors and recognition
timeouts propagate; orientation failures are recorded as described below. The wrapper gives an installation hint if the executable is
missing. Multi-frame TIFF/GIF images are explicitly rejected; split them into
individual images first. The native pytesseract wrapper uses temporary local files
and cleans them up; configure its temporary directory appropriately for sensitive data.

## Page rotation correction (0.3.0)

Image OCR and PDF OCR pages default to `rotation="auto"`. After EXIF correction,
Tesseract OSD estimates upright orientation in multiples of 90 degrees. The `osd`
language data must be installed (`tesseract-ocr-osd` on Debian/Ubuntu).
A correction is applied only when `orientation_confidence` reaches
`min_orientation_confidence=10`. This is Tesseract's score, not a probability.

```python
from truffles import ocr, read_documents

page = ocr("sideways.png")
print(page.metadata["rotation_clockwise"], page.metadata["orientation_status"])
forced = ocr("upside-down.png", rotation=180)  # Clockwise, after EXIF correction
pages = read_documents("scans.pdf", pdf_mode="ocr", rotation="auto")
```

Use `rotation=None` to disable OSD or `0/90/180/270` to apply a known correction.
Metadata distinguishes `upright`, `corrected`, `manual`, `disabled`, `low_confidence`
and `unavailable`. Sparse text, missing OSD data or an OSD timeout leave the image
unchanged and record uncertainty; they do not establish that the page is upright.
Check metadata or supply a manual correction when orientation matters. The OSD and
recognition subprocesses each have their own timeout. Plain OCR does not correct
small-angle skew. Native PDF text extraction bypasses OSD because it does not OCR.

## Table extraction (0.3.0)

Tesseract recognizes text; it does not reconstruct reliable table structure by
itself. Truffles combines it with optional **img2table** layout analysis:

```bash
python -m pip install -e '.[tables]'
# For a released wheel, use the installation URL above with truffles[tables].
```

The `tables` extra includes OCR, PDFium and OpenCV through img2table. It is separate
from `processing` to keep that installation lighter. The pinned table backend
supports Python 3.11-3.14; other Truffles features retain their own Python support.

```python
from truffles import extract_tables, processing_tools, rag

tables = extract_tables("statement.pdf")
for table in tables:
    print(table.page, table.rows, table.bbox, table.warnings)
    print(table.to_csv())
    payload = table.to_dict()  # JSON-serializable, includes needs_review=True
knowledge = rag([table.to_document() for table in tables])
tools = processing_tools("./uploads", tables=True)  # Adds extract_file_tables
```

Inputs may be image/PDF paths or bytes. PDFs are rasterized page by page, including
PDFs with a text layer. Pages are one-based. Each `TableResult` contains raw OCR
cell strings (or `None`), rows, table bounds, source, processing metadata and warnings.
`native` exposes the backend's extracted table and individual cell bounds for
advanced layout handling. CSV exports raw strings without inferring numeric types,
headers or totals; spreadsheet consumers must handle formula-like text appropriately.

Defaults target ruled tables: `psm=3`, `remove_lines=True`, `min_confidence=50`.
Long ruling lines are removed only from the OCR copy; layout detection sees the
original corrected image. Line removal can damage text overlapping rules; disable
it for affected documents. `psm` controls OCR segmentation; no one mode fits all scans.
`borderless=True`, `implicit_rows=True`, `implicit_columns=True` enable additional
native heuristics. `deskew=True` asks the backend to correct small-angle skew.
These options need evaluation on your own documents. Auto quarter-turn correction
uses the same confidence checks as ordinary OCR.

Bounds use pixels in the **processed image after rotation/deskew**, not coordinates
in the original upload or PDF points. The original file is never modified.
`min_confidence` filters OCR words; it does not measure the accuracy of rows, columns,
merged cells or arithmetic. Every table has `needs_review=True` and
`table_structure_unverified`. Low-confidence/unavailable auto orientation adds
`page_orientation_unverified`. Empty results do not prove there are no tables.
Complex merged cells, borderless layouts, handwriting and poor scans are not
certified by this wrapper. Validate structure and values before downstream decisions.

`max_tables=50`, `max_cells=10000` and `ProcessingLimits` bound returned data across
the file. Images must remain below 10,000 pixels per axis, including after deskew,
because of the pinned backend's hOCR coordinate parser. Reduce `dpi` when needed.
`timeout` bounds each OSD/hOCR subprocess, not native layout work or overall runtime.
Run native processing in bounded worker processes when a hard deadline is needed.

## PDF and plain-text ingestion

```python
from truffles import read_documents, rag

documents = read_documents("statement.pdf", pdf_mode="auto")
knowledge = rag(documents)
print(knowledge.retrieve("payment due"))
```

For PDFs, one Document is returned per page with **one-based** `page` metadata:

| `pdf_mode` | Behaviour |
| --- | --- |
| `auto` (default) | Extract the existing text layer; OCR pages with no non-whitespace extracted text |
| `text` | Extract text only; textless pages return empty text |
| `ocr` | Render and OCR every page, ignoring any existing text layer |

Auto mode does not detect poor-quality text layers or images mixed into a page
that already contains text. Use `ocr` explicitly when that behaviour is required.
Rendering defaults to `dpi=200` (allowed range 72-600); higher DPI increases memory
and runtime. OCR preprocessing is enabled for PDF pages. `password=` is forwarded
to PDFium for encrypted PDFs; passwords are not put in Document metadata.

Images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`, `.gif`) route to OCR.
`.txt`, `.md`, `.csv`, `.json` and `.log` files are decoded as literal text using
`encoding="utf-8"` by default. CSV/JSON are not interpreted as records. Plain-text
reading needs no optional extra. DOCX, spreadsheets, video decoding and remote URLs
are not included in these basic processors; use a suitable loader/adapter for them.

PDFium does not infer semantic reading order or reconstruct tables. Native PDF
resources are closed explicitly. Truffles serializes its PDFium operations with a
process-local lock because PDFium is not thread-safe, even across different files.
Use separate worker processes for parallel PDF work. Calls made directly to PDFium
outside this wrapper must be coordinated separately.

## Audio transcription

```python
from truffles import transcribe

transcript = transcribe("meeting.wav", language="en")
print(transcript.page_content)
```

The default creates the existing OpenAI speech adapter, sends audio to its endpoint
and closes the owned client afterwards. Configure `OPENAI_API_KEY` and install
`audio` or `processing`. This is an explicit external-service call, not local OCR.
No paid or live transcription is performed by the examples/tests in this release.

To reuse a client, choose a compatible endpoint or run an entirely local model:

```python
import os
from truffles import speech_model, transcribe

speech = speech_model(
    "openai",
    base_url="https://speech.example.com/v1",
    api_key=os.environ["SPEECH_API_KEY"],
    transcription_model="your-transcription-model",
    timeout=20,
    max_retries=0,
)
try:
    transcript = transcribe("recording.webm", speech=speech)
finally:
    speech.close()
```

A custom object implementing `transcribe(audio_bytes, filename=..., language=...) -> str`
can be supplied as `speech=`. The helper never closes a borrowed adapter. Pass
`filename="recording.webm"` when using bytes so the service receives the right name;
bytes otherwise default to `recording.wav`. Actual audio formats, duration limits,
codec handling and language support belong to the endpoint. There is no automatic
chunking, resampling, diarization or word-timestamp extraction.

OCR language codes such as `hin` differ from speech codes such as `hi`. Use the
correct code for the selected engine. The output metadata identifies the source,
`processor="transcription"` and an explicitly supplied language.

## Give processing tools to an agent

```python
from truffles import agent, processing_tools

bot = agent(
    "openai:gpt-4.1-mini",
    tools=processing_tools("./uploads"),
)
answer = bot.run("Read invoice.pdf and summarise its contents.")
print(answer.text)
print(answer.sources)
```

The root directory must exist. By default, the factory supplies `read_file` and
`ocr_file`. It adds `transcribe_file` only when a speech adapter is explicitly
supplied. Each model-visible tool takes only a relative `file` path; the model
cannot select a new root or endpoint or override processing limits.

```python
from truffles import ProcessingLimits, processing_tools

# speech is an already-configured adapter; you own its lifecycle.
def tools_for_uploads(speech):
    return processing_tools(
        "./uploads",
        speech=speech,
        language="eng",
        transcription_language="en",
        limits=ProcessingLimits(max_bytes=5_000_000, max_pages=10),
        max_tool_chars=8000,
    )
```

Absolute paths, traversal outside the configured root, directories, and symlinks
resolving outside it are rejected. This is **not an OS sandbox**: concurrent hostile
filesystem mutation can defeat a check-then-open policy. Use a dedicated trusted
upload directory per tenant/request, not a home directory, and isolate untrusted
native document processing in workers/containers as needed.

Tool results contain JSON with `text`, `truncated` and `documents`. `max_tool_chars`
caps the extracted text shown to the model, not JSON serialization overhead. Full
bounded Documents remain tool artifacts and are available in the agent's
`Result.sources`. Source paths in tool artifacts are relative to the allowed root.
Sync and async native tool execution and Observer callbacks work through the normal
LangChain tool interface; async calls use native thread fallback. Document content
is untrusted input; apply appropriate agent instructions and tool authorization.

## Limits and operational boundaries

`ProcessingLimits` defaults to 20 MiB input bytes, 50 PDF pages, 25 million pixels
per image/rendered page and 200,000 extracted characters per file. Inputs exceeding
limits are rejected; PDF extraction does not return silent partial success. The
OCR timeout is per page/image, not a deadline for the whole document. Limits bound
ordinary work, but do not guarantee native decoder memory usage or total elapsed
time. Use process-level resource limits for adversarial inputs.

Direct helpers are synchronous and do not install tracing callbacks by themselves;
use the native processing tools or wrap them in a Runnable for workflow telemetry.
Extraction text is returned to your application; the library does not redact PII.
Applications own upload validation, user consent, temporary storage, credential
handling, retention and permission to send audio to a remote service.

## Test and run the example

```bash
uv sync --locked --extra providers --extra memory --extra qdrant --extra chroma --extra faiss --extra processing --extra tables
uv run pytest tests/test_processing.py -q
uv run python examples/processing.py
uv run python examples/tables.py
```

The example generates its own synthetic invoice image and scanned PDF, runs real
Tesseract, feeds results into RAG and executes an agent tool call. Its model and
speech responses are labelled scripted test doubles. Files stay under the ignored
`.truffles/processing/` directory. CI installs the English Tesseract engine; the
real OCR test is skipped locally if that executable is absent.

Contracts and implementation references:
[Python Tesseract wrapper](https://pypi.org/project/pytesseract/),
[PDFium Python API and threading constraints](https://pypdfium2.readthedocs.io/en/stable/python_api.html),
[Truffles speech adapter](speech.md), [extensions](extensions.md).

Table implementation references: [Tesseract quality and layout guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html), [img2table](https://pypi.org/project/img2table/).
