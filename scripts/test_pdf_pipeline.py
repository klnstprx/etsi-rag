# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Test the full PDF→markdown pipeline against real PDFs from RustFS.

Simulates what the Go code does: pdftotext default mode → stripPageHeaders →
heading detection → title case → normalizeStructured.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

ENDPOINT = "http://localhost:9000"
BUCKET = "etsi-rag"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
PREFIX = "crawl/pdf/"
MAX_SAMPLES = 5

PAGE_NUMBER_RE = re.compile(r"(?im)^\s*p[aá]gina\s+\d+\s+de\s+\d+\s*$")

HEADING_PATTERNS = [
    "DESCRIPCION", "OBJETIVOS", "CONTENIDOS", "METODOLOGIA",
    "EVALUACION", "COMPETENCIAS", "TEMARIO", "BIBLIOGRAFIA",
    "ACTIVIDADES FORMATIVAS", "SISTEMA DE EVALUACION",
    "RESULTADOS DE APRENDIZAJE", "EQUIPO DOCENTE",
    "RECOMENDACIONES", "DISTRIBUCION DEL TRABAJO",
]
MIN_HEADING_LEN = 10

STOP_WORDS = {
    "de", "la", "el", "los", "las", "del", "en", "y", "o", "al",
    "por", "con", "para", "una", "un", "su", "sus", "e", "a",
}

ACCENT_MAP = str.maketrans(
    "ÁÉÍÓÚÑÜáéíóúñü",
    "AEIOUNUaeionnu",
)


def strip_accents(s: str) -> str:
    return s.translate(ACCENT_MAP)


def is_all_caps(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    has_letter = False
    for ch in stripped:
        if ch.isalpha():
            if ch.islower():
                return False
            has_letter = True
    return has_letter


def is_heading_line(line: str) -> bool:
    if not is_all_caps(line):
        return False
    normalized = strip_accents(line)
    for pat in HEADING_PATTERNS:
        if pat in normalized:
            return True
    return False


def to_title_case(s: str) -> str:
    words = s.split()
    result = []
    for i, w in enumerate(words):
        lower = w.lower()
        if i > 0 and lower in STOP_WORDS:
            result.append(lower)
        else:
            result.append(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper())
    return " ".join(result)


def strip_page_headers(text: str) -> str:
    if not text:
        return ""
    pages = text.split("\f")
    if len(pages) <= 1:
        return PAGE_NUMBER_RE.sub("", text).strip()

    line_counts: dict[str, int] = {}
    for page in pages:
        seen = set()
        for line in page.split("\n"):
            trimmed = line.strip()
            if trimmed and trimmed not in seen:
                seen.add(trimmed)
                line_counts[trimmed] = line_counts.get(trimmed, 0) + 1

    repeating = {line for line, count in line_counts.items() if count >= 2}

    result = []
    for page in pages:
        page_lines = []
        for line in page.split("\n"):
            trimmed = line.strip()
            if trimmed in repeating:
                continue
            if PAGE_NUMBER_RE.match(line):
                continue
            page_lines.append(line)
        cleaned = "\n".join(page_lines).strip()
        if cleaned:
            result.append(cleaned)

    return "\n\n".join(result)


def pdf_text_to_markdown(raw: str) -> str:
    if not raw:
        return ""
    text = strip_page_headers(raw)
    lines = text.split("\n")
    out = []
    for line in lines:
        trimmed = line.strip()
        if is_heading_line(trimmed):
            out.append(f"## {to_title_case(trimmed)}")
        else:
            out.append(line)

    result = "\n".join(out)
    # normalizeStructured: collapse 3+ newlines to 2, trim trailing spaces
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4"),
    )


def main():
    client = get_s3_client()
    resp = client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=100)
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    print(f"Found {len(keys)} PDF objects\n")

    if not keys:
        print("No PDFs found.")
        return

    samples = keys[:MAX_SAMPLES]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, key in enumerate(samples):
            pdf_path = Path(tmpdir) / f"sample_{i}.pdf"
            client.download_file(BUCKET, key, str(pdf_path))

            # pdftotext default mode
            result = subprocess.run(
                ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
                capture_output=True, text=True,
            )
            raw = result.stdout

            # pdfinfo for title
            info_result = subprocess.run(
                ["pdfinfo", str(pdf_path)],
                capture_output=True, text=True,
            )
            title = ""
            for line in info_result.stdout.split("\n"):
                if line.startswith("Title:"):
                    title = line.removeprefix("Title:").strip()
                    break

            # Run our pipeline
            markdown = pdf_text_to_markdown(raw)

            # Find detected headings
            headings = [l for l in markdown.split("\n") if l.startswith("## ")]

            print(f"{'='*80}")
            print(f"[{i}] {key}")
            print(f"    Title (pdfinfo): {title or '(none)'}")
            print(f"    Raw length: {len(raw)} chars")
            print(f"    Markdown length: {len(markdown)} chars")
            print(f"    Headings detected: {len(headings)}")
            for h in headings:
                print(f"      {h}")
            print(f"\n--- Markdown output (first 2000 chars) ---")
            print(markdown[:2000])
            if len(markdown) > 2000:
                print(f"\n... [{len(markdown) - 2000} more chars]")
            print()


if __name__ == "__main__":
    main()
