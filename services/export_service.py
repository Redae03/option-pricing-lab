"""Exports CSV, Excel et PDF."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from config import OUTPUT_DIR


def export_comparison_csv(rows: list[dict], filename="comparison_results.csv") -> Path:
    path = OUTPUT_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_comparison_excel(rows: list[dict], filename="comparison_results.xlsx") -> Path:
    path = OUTPUT_DIR / filename
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Comparaison")
    return path


def export_comparison_pdf(rows: list[dict], params: dict, filename="option_pricing_report.pdf") -> Path:
    path = OUTPUT_DIR / filename
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    story = [Paragraph("Option Pricing Lab - Rapport de comparaison", styles["Title"]), Spacer(1, 12)]
    ptxt = ", ".join([f"{k}={v}" for k, v in params.items()])
    story += [Paragraph("Paramètres", styles["Heading2"]), Paragraph(ptxt, styles["BodyText"]), Spacer(1, 12)]
    data = [["Méthode", "Prix", "Erreur abs.", "Erreur rel. (%)", "Temps (s)"]]
    for r in rows:
        data.append([r["method"], f"{r['price']:.6f}", f"{r['error']:.6f}", f"{r['relative_error']:.4f}", f"{r['time']:.6f}"])
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe3ef")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
    ]))
    story.append(tbl)
    doc.build(story)
    return path
