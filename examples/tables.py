"""Real local OCR: rotate a generated page, recover its table and export candidates."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from truffles import extract_tables, ocr


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


if __name__ == "__main__":
    root = Path(".truffles/tables")
    root.mkdir(parents=True, exist_ok=True)
    with synthetic_page() as original, original.rotate(90, expand=True) as rotated:
        rotated.save(root / "sideways.png")
        rotated.save(root / "sideways.pdf", resolution=200)
    document = ocr(root / "sideways.png")
    assert document.metadata["rotation_clockwise"] == 90
    print("Applied clockwise correction:", document.metadata["rotation_clockwise"])
    tables = extract_tables(root / "sideways.pdf")
    assert len(tables) == 1
    assert tables[0].rows == (
        ("Product", "Quantity", "Amount"),
        ("Apples", "3", "120"),
        ("Bananas", "5", "200"),
    )
    (root / "table.csv").write_text(tables[0].to_csv(), encoding="utf-8")
    (root / "tables.json").write_text(
        json.dumps([t.to_dict() for t in tables], indent=2), encoding="utf-8"
    )
    print(tables[0].to_csv())
    print("Candidate structure requires review:", tables[0].warnings)
    print("Synthetic files and exports:", root)
