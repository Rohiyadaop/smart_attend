from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Raspberry_Pi_4_Pilot_Deployment_Guide.md"
OUTPUT = ROOT / "SmartAttend_Raspberry_Pi_4_Pilot_Deployment_Guide.pdf"

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 54
RIGHT = 54
TOP = 64
BOTTOM = 60
FONT_SIZE = 11
LINE_HEIGHT = 15
CHAR_WIDTH = 5.7
MAX_CHARS = int((PAGE_WIDTH - LEFT - RIGHT) / CHAR_WIDTH)


def escape_pdf_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def markdown_to_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_code = False

    for raw in text.splitlines():
        line = raw.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            lines.append("")
            continue

        if not line.strip():
            lines.append("")
            continue

        if in_code:
            prefix = "    "
            wrapped = textwrap.wrap(
                line,
                width=max(20, MAX_CHARS - len(prefix)),
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            lines.extend(prefix + item for item in wrapped)
            continue

        if line.startswith("# "):
            lines.append(line[2:].strip().upper())
            lines.append("")
            continue

        if line.startswith("## "):
            lines.append(line[3:].strip())
            lines.append("")
            continue

        if line.startswith("### "):
            lines.append(line[4:].strip())
            continue

        bullet = None
        content = line
        if line.startswith("- "):
            bullet = "- "
            content = line[2:].strip()
        elif line[:2].isdigit() and line[2:4] == ". ":
            bullet = line[:4]
            content = line[4:].strip()

        if bullet:
            wrapped = textwrap.wrap(
                content,
                width=max(20, MAX_CHARS - len(bullet)),
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            lines.append(bullet + wrapped[0])
            for extra in wrapped[1:]:
                lines.append(" " * len(bullet) + extra)
        else:
            wrapped = textwrap.wrap(
                content,
                width=MAX_CHARS,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            lines.extend(wrapped)

    return lines


def build_pages(lines: list[str]) -> list[list[str]]:
    usable_height = PAGE_HEIGHT - TOP - BOTTOM
    max_lines_per_page = usable_height // LINE_HEIGHT
    pages: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        current.append(line)
        if len(current) >= max_lines_per_page:
            pages.append(current)
            current = []

    if current:
        pages.append(current)

    return pages


def build_content_stream(page_lines: list[str], page_no: int, total_pages: int) -> bytes:
    commands = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEFT} {PAGE_HEIGHT - TOP} Td"]

    for index, line in enumerate(page_lines):
        if index > 0:
            commands.append(f"0 -{LINE_HEIGHT} Td")
        commands.append(f"({escape_pdf_text(line)}) Tj")

    footer_y = 28
    commands.extend([
        "ET",
        "BT",
        "/F1 9 Tf",
        f"{LEFT} {footer_y} Td",
        f"(Page {page_no} of {total_pages}) Tj",
        "ET",
    ])
    return "\n".join(commands).encode("latin-1", errors="replace")


def make_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    total_pages = len(pages)

    for page_no, page_lines in enumerate(pages, start=1):
        stream = build_content_stream(page_lines, page_no, total_pages)
        content_id = add_object(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") +
            stream +
            b"\nendstream"
        )
        content_ids.append(content_id)
        page_ids.append(0)

    kids_refs = " ".join(f"{idx} 0 R" for idx in range(2 + total_pages, 2 + (2 * total_pages), 1))
    pages_id = add_object(
        f"<< /Type /Pages /Count {total_pages} /Kids [{kids_refs}] >>".encode("latin-1")
    )

    first_page_obj_index = len(objects) + 1
    actual_page_ids: list[int] = []
    for content_id in content_ids:
        actual_page_ids.append(add_object(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        ))

    kids_refs = " ".join(f"{idx} 0 R" for idx in actual_page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Count {total_pages} /Kids [{kids_refs}] >>".encode("latin-1")

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    lines = markdown_to_lines(text)
    pages = build_pages(lines)
    OUTPUT.write_bytes(make_pdf(pages))
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
