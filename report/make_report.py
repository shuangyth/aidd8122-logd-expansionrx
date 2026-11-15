"""Two-pass report build: docx -> PDF, read heading pages, write TOC numbers, rebuild.
Usage: python make_report.py [repo_url] [revision]   (needs LibreOffice `soffice` and `pdftotext`)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARGS = sys.argv[1:]
PAGES = HERE / "_pages.json"
DOCX, PDF = HERE / "AIDD8122_Final_Project_Report.docx", HERE / "AIDD8122_Final_Project_Report.pdf"
TITLES = ["Introduction", "Literature Review", "Research Questions and Objectives",
          "Methods and Materials", "Results and Discussion", "Conclusions", "References"]


def build():
    subprocess.run([sys.executable, "-I", str(HERE / "build_report.py"), *ARGS], check=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(DOCX)],
                   check=True, capture_output=True)


def page_texts():
    out = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True).stdout
    return out.split("\f")[:-1]


PAGES.unlink(missing_ok=True)
build()
texts = page_texts()
FRONT = 2                                   # cover + TOC are not numbered in the body
toc = []
for i, t in enumerate(TITLES):
    for n, txt in enumerate(texts):
        if n < FRONT:
            continue
        pat = rf"^\s*§\d\.\s+{re.escape(t)}\s*$" if t != "References" else r"^\s*References\s*$"
        if re.search(pat, txt, re.M):
            toc.append(n + 1 - FRONT)
            break
    else:
        raise SystemExit(f"heading not found: {t}")
PAGES.write_text(json.dumps({"toc": toc, "section_pages": len(texts) - FRONT}))
build()
texts2 = page_texts()
assert len(texts2) == len(texts), "pagination changed in the second pass"
print("TOC pages:", toc, "total PDF pages:", len(texts2))


# ---- LibreOffice in this environment cannot reach system CJK fonts, so the Chinese name is stamped onto the
# cover page of the PDF with PyMuPDF (exporting the .docx from Word embeds it natively; skip this then).
STUDENT_ID = os.environ.get("AIDD_STUDENT_ID", "P00000")
STUDENT_NAME = os.environ.get("AIDD_STUDENT_NAME", "Student")
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def stamp_cjk_name():
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(FONT, 160)
    bb = f.getbbox(STUDENT_NAME)
    im = Image.new("RGBA", (bb[2] + 16, bb[3] + 24), (255, 255, 255, 0))
    ImageDraw.Draw(im).text((8, 4), STUDENT_NAME, font=f, fill=(0, 0, 0, 255))
    png = HERE / "_name_cover.png"
    im.save(png)
    doc = fitz.open(PDF)
    page = doc[0]
    lab = page.search_for("Student Name:")[0]
    ref = page.search_for(STUDENT_ID)[0]
    h = lab.height * 0.95
    page.insert_image(fitz.Rect(ref.x0, lab.y0 + 0.5, ref.x0 + h * im.width / im.height, lab.y0 + 0.5 + h), filename=str(png))
    tmp = PDF.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(PDF)


stamp_cjk_name()
print("name stamped on cover")
