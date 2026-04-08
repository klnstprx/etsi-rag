# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Download PDFs from RustFS and compare pdftotext vs pdftohtml output."""

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
MAX_SAMPLES = 3


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4"),
    )


def list_pdfs(client) -> list[str]:
    resp = client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=100)
    return [obj["Key"] for obj in resp.get("Contents", [])]


def download_pdf(client, key: str, dest: Path):
    client.download_file(BUCKET, key, str(dest))


def run_pdftotext(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    return result.stdout


def run_pdftotext_layout(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    return result.stdout


def run_pdftohtml(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftohtml", "-s", "-i", "-noframes", "-enc", "UTF-8", "-stdout", str(pdf_path)],
        capture_output=True, text=True,
    )
    return result.stdout


def run_pdfinfo(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True, text=True,
    )
    return result.stdout


def main():
    client = get_s3_client()
    keys = list_pdfs(client)
    print(f"Found {len(keys)} PDF objects in RustFS")

    if not keys:
        print("No PDFs found. Has the crawler run?")
        return

    samples = keys[:MAX_SAMPLES]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, key in enumerate(samples):
            pdf_path = Path(tmpdir) / f"sample_{i}.pdf"
            download_pdf(client, key, pdf_path)
            size_kb = pdf_path.stat().st_size / 1024

            print(f"\n{'='*80}")
            print(f"[{i}] {key} ({size_kb:.1f} KB)")
            print(f"{'='*80}")

            # pdfinfo
            info = run_pdfinfo(pdf_path)
            print(f"\n--- pdfinfo ---")
            print(info.strip())

            # pdftotext (default / reading order)
            text_default = run_pdftotext(pdf_path)
            print(f"\n--- pdftotext (default, reading order) [{len(text_default)} chars] ---")
            print(text_default[:2000])
            if len(text_default) > 2000:
                print(f"... [{len(text_default) - 2000} more chars]")

            # pdftotext -layout
            text_layout = run_pdftotext_layout(pdf_path)
            print(f"\n--- pdftotext -layout [{len(text_layout)} chars] ---")
            print(text_layout[:2000])
            if len(text_layout) > 2000:
                print(f"... [{len(text_layout) - 2000} more chars]")

            # pdftohtml
            html = run_pdftohtml(pdf_path)
            print(f"\n--- pdftohtml -s -i -noframes [{len(html)} chars] ---")
            print(html[:3000])
            if len(html) > 3000:
                print(f"... [{len(html) - 3000} more chars]")


if __name__ == "__main__":
    main()
