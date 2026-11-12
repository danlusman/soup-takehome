"""Build report/report.pdf from report/report.md (Markdown -> styled HTML -> headless Edge print-to-PDF).

Run: .venv/Scripts/python report/build_pdf.py
"""
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 12mm 13mm 12mm 13mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 9.1pt; line-height: 1.32; color: #1a1a1a; }
h1 { font-size: 14.5pt; margin: 0 0 4px 0; color: #0b2545; border-bottom: 2px solid #0b2545; padding-bottom: 3px; }
h2 { font-size: 10.8pt; margin: 9px 0 3px 0; color: #0b2545; }
p { margin: 3px 0 4px 0; }
ul, ol { margin: 2px 0 4px 0; padding-left: 18px; }
li { margin: 1px 0; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 5px 0; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d1dc; padding: 2px 4px; vertical-align: top; text-align: left; }
th { background: #e8eef6; font-weight: 600; }
tr:nth-child(even) td { background: #f8fafc; }
code { font-family: Consolas, "Courier New", monospace; font-size: 7.8pt; background: #f1f3f5; padding: 0 2px; border-radius: 2px; }
strong { color: #000; }
"""


def main():
    md_text = (HERE / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>Soup take-home report</title><style>{CSS}</style></head><body>{body}</body></html>")
    html_path = HERE / "report.html"
    html_path.write_text(html, encoding="utf-8")
    pdf_path = HERE / "report.pdf"
    pdf_path.unlink(missing_ok=True)  # never count pages of a stale file
    import tempfile
    profile = tempfile.mkdtemp(prefix="edge_pdf_")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                    f"--user-data-dir={profile}",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, timeout=120)
    import time
    for _ in range(60):
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            break
        time.sleep(0.5)
    time.sleep(1)
    from pypdf import PdfReader
    print(f"wrote {pdf_path} pages={len(PdfReader(str(pdf_path)).pages)}")


if __name__ == "__main__":
    sys.exit(main())
