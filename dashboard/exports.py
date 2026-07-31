"""Excel/PDF export of any dashboard table DataFrame — pure functions
(no st.* calls), so download buttons in components/*.py stay one line."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def export_dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def export_dataframe_to_pdf(df: pd.DataFrame, title: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()

    data = [list(df.columns)] + df.astype(str).values.tolist()
    usable_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
    col_width = usable_width / max(len(df.columns), 1)
    table = Table(data, repeatRows=1, colWidths=[col_width] * len(df.columns))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F0F0")]),
    ]))

    doc.build([Paragraph(title, styles["Title"]), Spacer(1, 12), table])
    return buffer.getvalue()
