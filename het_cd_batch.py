"""
Batch HET-CD · Análisis sistemático de CD de la RPT
===================================================

Uso:
    python het_cd_batch.py data/het_cd_classifier_data.xlsx

Salida:
    ./resultados_het_cd_batch/
        resultados_revision_cd_rpt.xlsx
        informe_global_revision_cd_rpt.html
        informes_individuales/*.html

Requiere:
    pandas, openpyxl
    het_cd_engine.py en la misma carpeta o en PYTHONPATH.
"""
from __future__ import annotations

import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from het_cd_engine import analizar_rpt_completa


REQUIRED_SHEETS = ["puestos_vector", "patrones_vector", "rangos_cd"]


def load_workbook_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name) for name in xls.sheet_names}


def assert_required(sheets: Dict[str, pd.DataFrame]) -> None:
    missing = [s for s in REQUIRED_SHEETS if s not in sheets]
    if missing:
        raise ValueError(f"Faltan hojas obligatorias: {missing}")
    if sheets["puestos_vector"].empty:
        raise ValueError("La hoja puestos_vector está vacía.")


def _fmt(value: Any, col: str = "") -> str:
    try:
        if pd.isna(value):
            return "No calculable"
    except Exception:
        pass
    col_l = str(col).lower()
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            num = float(value)
            if any(k in col_l for k in ["impacto", "importe", "coste", "diferencial_anual", "periodo"]):
                return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            if "similitud" in col_l:
                return f"{num:.4f}"
            if any(k in col_l for k in ["cd", "nivel", "dotaciones", "puestos"]):
                return str(int(round(num))) if abs(num - round(num)) < 1e-9 else f"{num:.2f}"
            return f"{num:.2f}" if abs(num - round(num)) >= 1e-9 else str(int(round(num)))
    except Exception:
        pass
    return str(value)


def table_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    df = df.head(max_rows).copy()
    th = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{escape(_fmt(r.get(c, ''), c))}</td>" for c in df.columns) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_individual_report(result: Dict[str, Any]) -> str:
    ident = result.get("identificacion", {}) or {}
    res = result.get("resultado_cd", {}) or {}
    val = result.get("validacion_rango", {}) or {}
    sims = result.get("similitudes", {}) or {}
    arr = result.get("efecto_arrastre", {}) or {}
    rec = result.get("recomendacion_actuacion", {}) or {}
    impacto = result.get("impacto_economico_auto", {}) or {}
    top = pd.DataFrame(result.get("top_k_patrones", []) or [])
    comp = pd.DataFrame(result.get("comparables_internos", []) or [])
    return f"""
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Informe HET-CD</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;background:#eef1f3;color:#2f3941;margin:0;line-height:1.5}}
.page{{max-width:980px;margin:0 auto;background:white;min-height:100vh;padding:38px 46px;box-shadow:0 8px 30px rgba(0,0,0,.08)}}
h1{{color:#0b101d;text-transform:uppercase}} h2{{color:#006089;border-bottom:2px solid #cfd6dc;padding-bottom:5px;margin-top:26px}}
.box{{background:#f7f8fa;border:1px solid #cfd6dc;border-left:6px solid #006089;padding:12px 14px;margin:10px 0}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}} th,td{{border:1px solid #cfd6dc;padding:6px}} th{{background:#eef1f3}}
</style></head><body><div class="page">
<h1>Informe técnico HET-CD</h1>
<p><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
<h2>1. Identificación</h2>
<div class="box"><p><strong>Puesto:</strong> {escape(str(ident.get('denominacion_normalizada','')))}</p><p><strong>Grupo/Subgrupo:</strong> {escape(str(ident.get('grupo_subgrupo','')))} · <strong>CD vigente:</strong> {escape(str(ident.get('cd_vigente','')))}</p></div>
<h2>2. Validación y resultado</h2>
<div class="box"><p><strong>Estado rango:</strong> {escape(str(val.get('estado','')))}. {escape(str(val.get('mensaje','')))}</p><p><strong>CD K1 orientativo:</strong> {escape(str(res.get('cd_tecnico_recomendado','')))} · <strong>CD final ajustado:</strong> {escape(str(res.get('cd_tecnico_ajustado','')))} · <strong>Diferencial:</strong> {escape(str(res.get('diferencial_cd','')))}</p><p><strong>Recomendación:</strong> {escape(str(rec.get('titulo','')))}</p><p>{escape(str(rec.get('texto','')))}</p></div>
<h2>3. Similitud</h2>
<div class="box"><p>Funcional: {escape(str(sims.get('funcional','')))} · Factores CD: {escape(str(sims.get('factores_cd','')))} · Total: {escape(str(sims.get('combinada','')))}</p></div>
<h2>4. Patrones próximos</h2>{table_html(top)}
<h2>5. Arrastre interno</h2><div class="box"><p><strong>Riesgo:</strong> {escape(str(arr.get('riesgo_arrastre','')))}</p><p>{escape(str(arr.get('observacion','')))}</p></div>{table_html(comp.head(10) if not comp.empty else comp)}
<h2>6. Impacto económico</h2>
<div class="box"><p><strong>Diferencial anual por dotación:</strong> {escape(_fmt(impacto.get('diferencial_anual_por_dotacion',''), 'diferencial_anual_por_dotacion'))} €</p><p><strong>Dotaciones:</strong> {escape(str(impacto.get('dotaciones','')))} · <strong>Impacto anual total:</strong> {escape(_fmt(impacto.get('impacto_anual_total',''), 'impacto_anual_total'))} € · <strong>Impacto periodo:</strong> {escape(_fmt(impacto.get('impacto_periodo',''), 'impacto_periodo'))} €</p><p>{escape(str(impacto.get('motivo','')))}</p></div>
<h2>7. Cautela</h2><p>El resultado tiene carácter técnico auxiliar y no sustituye el expediente de modificación de RPT.</p>
</div></body></html>
"""


def build_global_report(summary: pd.DataFrame, agregado: Dict[str, Any]) -> str:
    dist = summary.groupby("resultado_preliminar", dropna=False).size().reset_index(name="puestos") if not summary.empty else pd.DataFrame()
    por_grupo = summary.groupby("grupo_subgrupo", dropna=False).agg(
        puestos=("id_het", "count"), impacto_anual_total=("impacto_anual_total", "sum")
    ).reset_index() if not summary.empty else pd.DataFrame()
    prioritarios = summary[summary["resultado_preliminar"].isin(["ANALIZAR_AL_ALZA", "REVISION_AGRUPADA", "INCIDENCIA_NORMATIVA"])] if "resultado_preliminar" in summary.columns else pd.DataFrame()
    sim_cols = [c for c in ["id_het", "denominacion_normalizada", "grupo_subgrupo", "cd_vigente", "patron_k1", "nombre_patron_k1", "cd_k1_orientativo", "similitud_total_k1", "similitud_funcional_k1", "similitud_factores_cd_k1", "resultado_preliminar"] if c in summary.columns]
    similitudes = summary[sim_cols] if sim_cols else pd.DataFrame()
    return f"""
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Informe global HET-CD RPT</title>
<style>@page{{size:A4 portrait;margin:16mm 12mm}}body{{font-family:Arial,Helvetica,sans-serif;background:#eef1f3;color:#2f3941;margin:0;line-height:1.5}}.page{{max-width:1100px;margin:0 auto;background:white;min-height:100vh;padding:38px 46px;box-shadow:0 8px 30px rgba(0,0,0,.08)}}h1{{color:#0b101d;text-transform:uppercase}} h2{{color:#006089;border-bottom:2px solid #cfd6dc;padding-bottom:5px;margin-top:26px}}.box{{background:#f7f8fa;border:1px solid #cfd6dc;border-left:6px solid #006089;padding:12px 14px;margin:10px 0}}table{{border-collapse:collapse;width:100%;font-size:10.5px;margin-top:10px;page-break-inside:auto}} th,td{{border:1px solid #cfd6dc;padding:5px;vertical-align:top;word-break:break-word}} th{{background:#eef1f3}}</style>
</head><body><div class="page"><h1>Informe global de revisión técnica de CD de la RPT</h1>
<p>El informe resume la evaluación sistemática de los puestos tipo cargados en puestos_vector. La recomendación se basa en el patrón K1 más próximo y en el ajuste al rango legal del grupo/subgrupo.</p>
<div class="box"><p><strong>Puestos analizados:</strong> {escape(str(agregado.get('puestos_analizados','')))}</p><p><strong>Impacto anual total:</strong> {escape(_fmt(agregado.get('impacto_anual_total',''), 'impacto_anual_total'))} €</p><p><strong>Impacto periodo:</strong> {escape(_fmt(agregado.get('impacto_periodo_total',''), 'impacto_periodo_total'))} €</p></div>
<h2>Distribución por resultado</h2>{table_html(dist)}
<h2>Resumen por grupo/subgrupo</h2>{table_html(por_grupo)}
<h2>Similitud técnica del conjunto de puestos</h2><p>Se muestra el patrón K1 y las similitudes funcional, de factores CD y combinada para facilitar la revisión del encaje técnico.</p>{table_html(similitudes, max_rows=300)}
<h2>Puestos de revisión prioritaria</h2>{table_html(prioritarios, max_rows=250)}
<p>Documento técnico auxiliar. Requiere revisión técnica, informes preceptivos y tramitación del expediente de RPT. El batch calcula impacto por las dotaciones informadas en el Excel o, en su defecto, por una dotación.</p>
</div></body></html>
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    excel_path = Path(sys.argv[1]).resolve()
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)
    print("Cargando Excel...", flush=True)
    sheets = load_workbook_sheets(excel_path)
    assert_required(sheets)

    print("Ejecutando análisis HET-CD...", flush=True)
    analysis = analizar_rpt_completa(
        puestos_vector=sheets["puestos_vector"],
        patrones_vector=sheets["patrones_vector"],
        rangos_cd=sheets["rangos_cd"],
        importes_cd_2026=sheets.get("importes_cd_2026"),
        pesos_modelo=sheets.get("pesos_modelo"),
        topk=5,
        meses=12,
        anio=2026,
    )

    print("Preparando salidas...", flush=True)
    out_dir = Path("resultados_het_cd_batch")
    reports_dir = out_dir / "informes_individuales"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(analysis["resumen_rows"])
    print("Escribiendo Excel de resultados...", flush=True)
    with pd.ExcelWriter(out_dir / "resultados_revision_cd_rpt.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="resultados_revision_cd_rpt", index=False)
        pd.DataFrame([analysis["agregado"]]).to_excel(writer, sheet_name="resumen_global", index=False)
        if not summary.empty:
            summary.groupby("resultado_preliminar", dropna=False).size().reset_index(name="puestos").to_excel(writer, sheet_name="distribucion_resultados", index=False)

    print("Generando informes individuales...", flush=True)
    for idx, result in enumerate(analysis["resultados"], start=1):
        ident = result.get("identificacion", {}) or {}
        raw_name = str(ident.get("denominacion_normalizada") or ident.get("id_het") or f"puesto_{idx}")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_name)[:80]
        (reports_dir / f"{idx:03d}_{safe}.html").write_text(build_individual_report(result), encoding="utf-8")

    print("Generando informe global...", flush=True)
    (out_dir / "informe_global_revision_cd_rpt.html").write_text(build_global_report(summary, analysis["agregado"]), encoding="utf-8")

    print(f"Análisis completado: {analysis['agregado']['puestos_analizados']} puestos")
    print(f"Salida: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
