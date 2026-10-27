"""Render the write-up Markdown (a small subset: headings, paragraphs, bullets, tables, images, bold, italics) to PDF."""

import re
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

rl_config.invariant = 1  # same input, byte for byte the same PDF (no timestamps)
BASE = 8.6
MARGIN = 14 * mm
WIDTH = A4[0] - 2 * MARGIN


def _inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return t


def _col_widths(rows):
    """Share the page width by the longest cell in each column, so text columns get room and number columns stay narrow."""
    longest = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    weights = [min(max(n, 6), 48) for n in longest]
    return [WIDTH * w / sum(weights) for w in weights]


def render(md_text, out_path, title, base_dir="."):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=BASE, leading=BASE * 1.22, alignment=TA_JUSTIFY, spaceAfter=2.2)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceAfter=2, alignment=0)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, spaceBefore=4, spaceAfter=1.5, textColor=colors.HexColor("#1f3b63"), alignment=0)
    small = ParagraphStyle("s", parent=body, fontSize=BASE - 0.9, leading=(BASE - 0.9) * 1.2, alignment=0)
    cell = ParagraphStyle("c", parent=body, fontSize=BASE - 1.4, leading=(BASE - 1.4) * 1.15, alignment=0, spaceAfter=0)
    story, bullets, table = [], [], []

    def flush_bullets():
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(_inline(b), body), leftIndent=9, value="•") for b in bullets], bulletType="bullet",
                                      start="•", leftIndent=9, bulletFontSize=BASE, bulletOffsetY=0))
            bullets.clear()

    def flush_table():
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
            data = [[Paragraph(_inline(c), cell) for c in r] for r in rows]
            t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=_col_widths(rows))
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c4d4")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
            story.append(t)
            story.append(Spacer(1, 3))
            table.clear()

    pending_pic = []

    def flush_pic(caption=None):
        if pending_pic:
            story.append(KeepTogether(pending_pic + ([Paragraph(_inline(caption), small)] if caption else [])))
            pending_pic.clear()

    for line in md_text.splitlines():
        if pending_pic and line.strip():
            if line.startswith("<small>"):
                flush_pic(line.replace("<small>", "").replace("</small>", ""))
                continue
            flush_pic()
        if line.startswith("|"):
            flush_bullets()
            table.append(line)
            continue
        flush_table()
        if line.startswith("- "):
            bullets.append(line[2:])
            continue
        flush_bullets()
        if not line.strip():
            continue
        img = re.match(r"^!\[(.*)\]\((.+)\)\s*$", line)
        if img:
            path = Path(base_dir) / img.group(2)
            pic = Image(str(path))
            pic.drawHeight, pic.drawWidth = WIDTH * pic.imageHeight / pic.imageWidth, WIDTH
            pending_pic.append(pic)
        elif line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), h2))
        elif line.startswith("<small>"):
            story.append(Paragraph(_inline(line.replace("<small>", "").replace("</small>", "")), small))
        else:
            story.append(Paragraph(_inline(line), body))
    flush_bullets()
    flush_table()
    flush_pic()
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=11 * mm, bottomMargin=11 * mm, title=title, author="Mohammed Zeeshan")
    doc.build(story)
    from pypdf import PdfReader

    return len(PdfReader(str(out_path)).pages)
