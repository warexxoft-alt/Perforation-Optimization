"""
modules/report_export.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
Export & print integration. v2 had NO export path at all beyond whatever a
user could screenshot — every table lived only inside the Streamlit session
and every chart was view-only Plotly. This module adds:
  · A formatted, multi-sheet Excel workbook (openpyxl)
  · A print-ready PDF technical report (reportlab), embedding both tables
    and 600 DPI figures from `modules.pub_plots`
  · A Word (.docx) summary report (python-docx) matching the author's usual
    deliverable format
  · A zip bundle of every generated 600 DPI figure
  · Plain CSV export for any single table
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, TableStyle
from reportlab.platypus import Table as RLTable

from config import APP_TITLE, COLORS, VERSION

# ═══════════════════════════════════════════════════════════════════════════
# 1.  EXCEL WORKBOOK  (formatted, multi-sheet)
# ═══════════════════════════════════════════════════════════════════════════
_HEADER_FILL = PatternFill("solid", fgColor="2E75B6")
_HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
_THIN = Side(style="thin", color="D0D0D0")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _style_sheet(ws, df: pd.DataFrame):
    for j, col in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=j, value=str(col))
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER
    for i, row in enumerate(df.itertuples(index=False), start=2):
        for j, val in enumerate(row, start=1):
            c = ws.cell(row=i, column=j, value=val if not pd.isna(val) else None)
            c.border = _BORDER
            if isinstance(val, str) and ("✅" in val or "Yes" in val or "Better" in val):
                c.font = Font(color="1E7E34")
            elif isinstance(val, str) and ("❌" in val or "Below" in val):
                c.font = Font(color="B02A37")
    for j, col in enumerate(df.columns, start=1):
        width = max(11, min(38, int(df[col].astype(str).str.len().max() if len(df) else 10) + 2,
                             len(str(col)) + 4))
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.freeze_panes = "A2"
    if len(df) > 0 and len(df.columns) > 0:
        ref = f"A1:{get_column_letter(len(df.columns))}{len(df)+1}"
        try:
            tbl = Table(displayName=f"T{abs(hash(ws.title)) % 100000}", ref=ref)
            tbl.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
            ws.add_table(tbl)
        except Exception:
            pass


def export_excel(tables: dict, filepath: str, title: str = APP_TITLE) -> str:
    """`tables`: {sheet_name: DataFrame}. Writes to `filepath`, returns it."""
    wb = Workbook()
    wb.remove(wb.active)

    cover = wb.create_sheet("Summary", 0)
    cover["A1"] = title
    cover["A1"].font = Font(size=16, bold=True, color="2E75B6")
    cover["A2"] = f"Generated {datetime.now():%Y-%m-%d %H:%M}  ·  PerfoSim Pro v{VERSION}"
    cover["A2"].font = Font(size=10, italic=True, color="808080")
    cover["A4"] = "Contents"
    cover["A4"].font = Font(bold=True, size=12)
    for i, name in enumerate(tables.keys(), start=5):
        cover[f"A{i}"] = f"• {name}"
    cover.column_dimensions["A"].width = 50

    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        safe_name = str(name)[:31]
        ws = wb.create_sheet(safe_name)
        _style_sheet(ws, df.reset_index(drop=True) if df.index.name is None else df.reset_index())

    wb.save(filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PDF TECHNICAL REPORT  (print-ready, embeds 600 DPI figures)
# ═══════════════════════════════════════════════════════════════════════════
def _pdf_styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("PSTitle", parent=ss["Title"], textColor=rl_colors.HexColor(COLORS["primary"])))
    ss.add(ParagraphStyle("PSHeading", parent=ss["Heading2"], textColor=rl_colors.HexColor(COLORS["primary"]),
                          spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("PSBody", parent=ss["BodyText"], fontSize=10, leading=14))
    ss.add(ParagraphStyle("PSCaption", parent=ss["BodyText"], fontSize=8.5, textColor=rl_colors.grey,
                          alignment=1, spaceBefore=2, spaceAfter=10))
    return ss


def _df_to_rltable(df: pd.DataFrame, max_rows=25):
    df = df.head(max_rows)
    data = [list(df.columns)] + df.astype(str).values.tolist()
    tbl = RLTable(data, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(COLORS["primary"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#D0D0D0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F5F7FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def export_pdf_report(filepath: str, title: str, sections: list, subtitle: str = None) -> str:
    """
    `sections`: list of dicts, each with:
      {"heading": str, "text": str (optional), "table": DataFrame (optional),
       "figure_bytes": bytes (optional PNG, already 600 DPI), "caption": str}
    Renders a print-ready PDF with a title page + one flowing section per dict.
    """
    ss = _pdf_styles()
    doc = SimpleDocTemplate(filepath, pagesize=LETTER,
                            topMargin=0.85*inch, bottomMargin=0.85*inch,
                            leftMargin=0.75*inch, rightMargin=0.75*inch)
    story = [Paragraph(title, ss["PSTitle"])]
    if subtitle:
        story.append(Paragraph(subtitle, ss["PSBody"]))
    story.append(Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} · PerfoSim Pro v{VERSION}",
                           ss["PSCaption"]))
    story.append(Spacer(1, 14))

    for sec in sections:
        story.append(Paragraph(sec["heading"], ss["PSHeading"]))
        if sec.get("text"):
            story.append(Paragraph(sec["text"], ss["PSBody"]))
            story.append(Spacer(1, 6))
        if sec.get("table") is not None and len(sec["table"]) > 0:
            story.append(_df_to_rltable(sec["table"]))
            story.append(Spacer(1, 8))
        if sec.get("figure_bytes"):
            img = RLImage(io.BytesIO(sec["figure_bytes"]), width=6.2*inch, height=None)
            img.hAlign = "CENTER"
            img.drawHeight = img.drawWidth * img.imageHeight / img.imageWidth
            max_h = 6.5 * inch
            if img.drawHeight > max_h:
                scale = max_h / img.drawHeight
                img.drawHeight *= scale
                img.drawWidth *= scale
            story.append(img)
            if sec.get("caption"):
                story.append(Paragraph(sec["caption"], ss["PSCaption"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# 3.  WORD (.docx) SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
def export_word_report(filepath: str, title: str, sections: list, subtitle: str = None) -> str:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if subtitle:
        p = doc.add_paragraph(subtitle)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta = doc.add_paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M} · PerfoSim Pro v{VERSION}")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.runs[0].italic = True
    doc.add_page_break()

    for sec in sections:
        doc.add_heading(sec["heading"], level=1)
        if sec.get("text"):
            doc.add_paragraph(sec["text"])
        if sec.get("table") is not None and len(sec["table"]) > 0:
            df = sec["table"].head(30)
            t = doc.add_table(rows=1, cols=len(df.columns))
            t.style = "Light Grid Accent 1"
            for j, col in enumerate(df.columns):
                t.rows[0].cells[j].text = str(col)
            for _, row in df.iterrows():
                cells = t.add_row().cells
                for j, val in enumerate(row):
                    cells[j].text = "" if pd.isna(val) else str(val)
        if sec.get("figure_bytes"):
            doc.add_picture(io.BytesIO(sec["figure_bytes"]), width=Inches(6.0))
            if sec.get("caption"):
                cap = doc.add_paragraph(sec["caption"])
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.runs[0].italic = True
                cap.runs[0].font.size = Pt(9)
        doc.add_paragraph()

    doc.save(filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# 4.  FIGURE ZIP BUNDLE
# ═══════════════════════════════════════════════════════════════════════════
def figures_to_zip(fig_bytes: dict) -> bytes:
    """`fig_bytes`: {filename_without_ext: png_bytes} -> zip archive bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in fig_bytes.items():
            zf.writestr(f"{name}.png", data)
    buf.seek(0)
    return buf.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
