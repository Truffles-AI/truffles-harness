"""Real local OCR, scanned-PDF ingestion and agent tools. No live model calls.

Install the processing extra and the Tesseract executable before running.
Outputs go to .truffles/processing by default.
"""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage
from PIL import Image, ImageDraw, ImageFont

from truffles import Observer, agent, ocr, processing_tools, rag, read_documents, transcribe
from truffles.testing import ScriptedChatModel


class ScriptedSpeech:
    """Explicit test double, not speech recognition."""

    def transcribe(self, audio, *, filename, language):
        return "Synthetic meeting: review invoice 12345."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".truffles/processing"))
    directory = parser.parse_args().output_dir
    directory.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (1100, 200), "white") as image:
        ImageDraw.Draw(image).text(
            (35, 60), "TRUFFLES INVOICE 12345", fill="black", font=ImageFont.load_default(size=48)
        )
        image.save(directory / "invoice.png")
        image.save(directory / "scan.pdf")
    image_text = ocr(directory / "invoice.png", psm=6)
    documents = read_documents(directory / "scan.pdf")
    assert "12345" in image_text.page_content and "TRUFFLES" in documents[0].page_content
    assert "12345" in rag(documents).retrieve("invoice")[0].page_content

    (directory / "policy.txt").write_text("Invoices are payable within 30 days.")
    observer = Observer()
    bot = agent(
        ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "read-policy",
                            "name": "read_file",
                            "args": {"file": "policy.txt"},
                        }
                    ],
                ),
                AIMessage(content="Invoices are payable within 30 days."),
            ]
        ),
        tools=processing_tools(directory),
    )
    answer = bot.run("Read policy.txt", config={"callbacks": [observer]})
    assert answer.sources[0].metadata["source"] == "policy.txt"
    transcript = transcribe(b"synthetic audio", speech=ScriptedSpeech(), filename="meeting.wav")
    result = {
        "real_tesseract_ocr": image_text.page_content,
        "real_scanned_pdf_ocr": documents[0].page_content,
        "scripted_agent_answer": answer.text,
        "scripted_transcription": transcript.page_content,
        "observed_tool_calls": observer.summary()["tool_calls"],
    }
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
