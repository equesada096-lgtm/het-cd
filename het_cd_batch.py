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



def _safe_num(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def build_prioritarios_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    dif = pd.to_numeric(df.get("diferencial_cd", pd.Series(index=df.index, dtype=float)), errors="coerce")
    res = df.get("resultado_preliminar", pd.Series(index=df.index, dtype=str)).astype(str)
    k1 = pd.to_numeric(df.get("cd_k1_orientativo", pd.Series(index=df.index, dtype=float)), errors="coerce")
    final = pd.to_numeric(df.get("cd_final_ajustado", pd.Series(index=df.index, dtype=float)), errors="coerce")
    ajuste_legal = k1.notna() & final.notna() & (k1.round(0) != final.round(0))
    prior = df[(dif.fillna(0) != 0) | (res == "INCIDENCIA_NORMATIVA") | ajuste_legal].copy()
    if prior.empty:
        return pd.DataFrame()

    def motivo(row):
        d = _safe_num(row.get("diferencial_cd"))
        cd_k1 = row.get("cd_k1_orientativo")
        cd_final = row.get("cd_final_ajustado")
        resultado = str(row.get("resultado_preliminar", ""))
        try:
            legal = pd.notna(cd_k1) and pd.notna(cd_final) and int(round(float(cd_k1))) != int(round(float(cd_final)))
        except Exception:
            legal = False
        if d > 0:
            if resultado == "INCIDENCIA_NORMATIVA" or legal:
                return "Nivel técnico superior al CD actual, con ajuste al máximo legal admisible"
            return "Nivel técnico superior al CD actual"
        if resultado == "INCIDENCIA_NORMATIVA" or legal:
            return "Nivel técnico superior no aplicable por límite legal" if _safe_num(cd_k1) > _safe_num(cd_final) else "Incidencia normativa en el rango de CD"
        if d < 0:
            return "Nivel técnico inferior al CD actual"
        return "Incidencia técnica relevante"

    def conclusion(row):
        d = _safe_num(row.get("diferencial_cd"))
        cd_k1 = row.get("cd_k1_orientativo")
        cd_final = row.get("cd_final_ajustado")
        resultado = str(row.get("resultado_preliminar", ""))
        try:
            legal = pd.notna(cd_k1) and pd.notna(cd_final) and int(round(float(cd_k1))) != int(round(float(cd_final)))
        except Exception:
            legal = False
        if d > 0:
            if resultado == "INCIDENCIA_NORMATIVA" or legal:
                return "Procede proponer la adecuación del puesto al CD final admisible resultante del análisis técnico; no procede elevarlo al CD K1 bruto por superar el intervalo legal aplicable al subgrupo de clasificación."
            return "Procede proponer la adecuación del puesto al nivel de complemento de destino resultante del análisis técnico."
        if resultado == "INCIDENCIA_NORMATIVA" or legal:
            return "No procede proponer la elevación del nivel de complemento de destino por superar el intervalo legal aplicable al subgrupo de clasificación."
        if d < 0:
            return "No procede propuesta de incremento; el análisis técnico apunta a un nivel funcional inferior al actualmente asignado."
        return "No procede actuación prioritaria sobre el nivel de complemento de destino."

    prior["motivo_inclusion"] = prior.apply(motivo, axis=1)
    prior["conclusion_tecnica"] = prior.apply(conclusion, axis=1)
    cols = [c for c in ["id_het", "denominacion_normalizada", "grupo_subgrupo", "cd_vigente", "cd_k1_orientativo", "cd_final_ajustado", "diferencial_cd", "similitud_total_k1", "motivo_inclusion", "conclusion_tecnica"] if c in prior.columns]
    return prior[cols].rename(columns={
        "id_het": "Código",
        "denominacion_normalizada": "Puesto",
        "grupo_subgrupo": "Subgrupo",
        "cd_vigente": "CD actual",
        "cd_k1_orientativo": "CD K1 orientativo",
        "cd_final_ajustado": "CD final admisible",
        "diferencial_cd": "Diferencia final",
        "similitud_total_k1": "Similitud K1",
        "motivo_inclusion": "Motivo de inclusión",
        "conclusion_tecnica": "Conclusión técnica",
    })


def build_impacto_table(summary: pd.DataFrame, prioritarios_table: pd.DataFrame = None) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    dif = pd.to_numeric(df.get("diferencial_cd", pd.Series(index=df.index, dtype=float)), errors="coerce")
    res = df.get("resultado_preliminar", pd.Series(index=df.index, dtype=str)).astype(str)
    k1 = pd.to_numeric(df.get("cd_k1_orientativo", pd.Series(index=df.index, dtype=float)), errors="coerce")
    final = pd.to_numeric(df.get("cd_final_ajustado", pd.Series(index=df.index, dtype=float)), errors="coerce")
    ajuste_legal = k1.notna() & final.notna() & (k1.round(0) != final.round(0))
    prior = df[(dif.fillna(0) != 0) | (res == "INCIDENCIA_NORMATIVA") | ajuste_legal].copy()
    cols = [c for c in ["id_het", "denominacion_normalizada", "diferencial_anual_por_dotacion", "dotaciones", "impacto_anual_total", "impacto_periodo"] if c in prior.columns]
    if not cols:
        return pd.DataFrame()
    out = prior[cols].copy()
    if "impacto_anual_total" in out.columns:
        out["impacto_anual_total"] = pd.to_numeric(out["impacto_anual_total"], errors="coerce")
    if "diferencial_anual_por_dotacion" in out.columns:
        out["diferencial_anual_por_dotacion"] = pd.to_numeric(out["diferencial_anual_por_dotacion"], errors="coerce")
        out = out[out["diferencial_anual_por_dotacion"] > 0].copy()
    elif "impacto_anual_total" in out.columns:
        out = out[out["impacto_anual_total"].notna()].copy()
    return out.rename(columns={
        "id_het": "Código",
        "denominacion_normalizada": "Puesto",
        "diferencial_anual_por_dotacion": "Diferencial anual por dotación",
        "dotaciones": "Dotaciones",
        "impacto_anual_total": "Impacto anual total",
        "impacto_periodo": "Impacto periodo",
    })


def resultado_informe(row: Any) -> str:
    """Clasificación de informe basada en la conclusión técnica final.

    A diferencia de resultado_preliminar, cuenta como actuación al alza los
    casos en los que el K1 bruto excede el rango legal pero el CD final admisible
    sigue siendo superior al CD vigente.
    """
    d = _safe_num(row.get("diferencial_cd"))
    try:
        cd_k1 = row.get("cd_k1_orientativo")
        cd_final = row.get("cd_final_ajustado")
        ajuste_legal = pd.notna(cd_k1) and pd.notna(cd_final) and int(round(float(cd_k1))) != int(round(float(cd_final)))
    except Exception:
        ajuste_legal = False
    prelim = str(row.get("resultado_preliminar", ""))
    if d > 0:
        return "Adecuación al alza"
    if d < 0:
        return "No procede incremento / coherencia a la baja"
    if prelim == "INCIDENCIA_NORMATIVA" or ajuste_legal:
        return "Incidencia normativa sin incremento admisible"
    return "Sin actuación prioritaria"


def distribucion_resultado_informe(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    df["resultado"] = df.apply(resultado_informe, axis=1)
    orden = [
        "Adecuación al alza",
        "No procede incremento / coherencia a la baja",
        "Incidencia normativa sin incremento admisible",
        "Sin actuación prioritaria",
    ]
    dist = df.groupby("resultado", dropna=False).size().reset_index(name="puestos")
    dist["_orden"] = dist["resultado"].apply(lambda x: orden.index(x) if x in orden else 999)
    return dist.sort_values("_orden").drop(columns="_orden")


def resumen_por_grupo_informe(summary: pd.DataFrame) -> pd.DataFrame:
    """Resumen por grupo/subgrupo basado en la conclusión técnica final, no en etiquetas preliminares."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    df["resultado"] = df.apply(resultado_informe, axis=1)
    if "impacto_anual_total" in df.columns:
        df["impacto_anual_total"] = pd.to_numeric(df["impacto_anual_total"], errors="coerce").fillna(0.0)
    else:
        df["impacto_anual_total"] = 0.0
    if "dotaciones" in df.columns:
        df["dotaciones"] = pd.to_numeric(df["dotaciones"], errors="coerce").fillna(0.0)
    else:
        df["dotaciones"] = 0.0
    df["dotaciones_al_alza"] = df.apply(
        lambda r: float(r.get("dotaciones", 0.0) or 0.0) if r.get("resultado") == "Adecuación al alza" else 0.0,
        axis=1,
    )
    out = df.groupby("grupo_subgrupo", dropna=False).agg(
        puestos=("id_het", "count"),
        adecuacion_al_alza=("resultado", lambda s: int((s == "Adecuación al alza").sum())),
        dotaciones_afectadas_al_alza=("dotaciones_al_alza", "sum"),
        no_procede_incremento=("resultado", lambda s: int((s == "No procede incremento / coherencia a la baja").sum())),
        incidencia_normativa_sin_incremento=("resultado", lambda s: int((s == "Incidencia normativa sin incremento admisible").sum())),
        sin_actuacion_prioritaria=("resultado", lambda s: int((s == "Sin actuación prioritaria").sum())),
        impacto_anual_total=("impacto_anual_total", "sum"),
    ).reset_index()
    return out.rename(columns={
        "grupo_subgrupo": "Grupo/Subgrupo",
        "puestos": "Puestos tipo analizados",
        "adecuacion_al_alza": "Puestos tipo con adecuación al alza",
        "dotaciones_afectadas_al_alza": "Dotaciones afectadas con adecuación al alza",
        "no_procede_incremento": "Puestos tipo sin incremento / ajuste a la baja",
        "incidencia_normativa_sin_incremento": "Puestos tipo con incidencia normativa sin incremento admisible",
        "sin_actuacion_prioritaria": "Puestos tipo sin actuación prioritaria",
        "impacto_anual_total": "Impacto anual estimado sobre dotaciones informadas",
    })

def table_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    df = df.head(max_rows).copy()
    th = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{escape(_fmt(r.get(c, ''), c))}</td>" for c in df.columns) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def parametros_tecnicos_html(agregado: Dict[str, Any] | None = None, result: Dict[str, Any] | None = None) -> str:
    """Bloque HTML de trazabilidad de parámetros técnicos alfa/beta."""
    agregado = agregado or {}
    params = (result or {}).get("parametros_modelo", {}) if result else {}
    alpha = params.get("peso_funcional", agregado.get("peso_funcional", 0.40))
    beta = params.get("peso_factores_cd", agregado.get("peso_factores_cd", 0.60))
    try:
        alpha_f = float(alpha)
    except Exception:
        alpha_f = 0.40
    try:
        beta_f = float(beta)
    except Exception:
        beta_f = 0.60
    formula = params.get("formula_similitud_total") or f"{alpha_f:.2f} × similitud funcional + {beta_f:.2f} × similitud factores CD"
    return (
        '<div class="box">'
        f'<p><strong>α (alfa) · peso del subespacio funcional/verbos:</strong> {alpha_f:.2f}</p>'
        f'<p><strong>β (beta) · peso del subespacio de factores técnicos de CD:</strong> {beta_f:.2f}</p>'
        f'<p><strong>Fórmula de similitud combinada:</strong> {escape(str(formula))}</p>'
        '<p>Estos parámetros ponderan la contribución relativa del patrón funcional y de los factores técnicos de complemento de destino en la similitud total utilizada para seleccionar el patrón K1.</p>'
        '</div>'
    )


def build_individual_report(result: Dict[str, Any]) -> str:
    ident = result.get("identificacion", {}) or {}
    res = result.get("resultado_cd", {}) or {}
    val = result.get("validacion_rango", {}) or {}
    sims = result.get("similitudes", {}) or {}
    impacto = result.get("impacto_economico_auto", {}) or {}
    top = pd.DataFrame(result.get("top_k_patrones", []) or [])
    comp = pd.DataFrame(result.get("comparables_internos", []) or [])
    if not comp.empty:
        comp_cols = [c for c in ["id_het", "denominacion_normalizada", "grupo_subgrupo", "cd_vigente", "similitud_total"] if c in comp.columns]
        comp = comp[comp_cols].rename(columns={"id_het": "Código", "denominacion_normalizada": "Puesto comparable", "grupo_subgrupo": "Subgrupo", "cd_vigente": "CD vigente", "similitud_total": "Similitud"})

    def ajuste_legal() -> bool:
        try:
            cd_k1 = res.get("cd_tecnico_recomendado")
            cd_final = res.get("cd_tecnico_ajustado")
            return pd.notna(cd_k1) and pd.notna(cd_final) and int(round(float(cd_k1))) != int(round(float(cd_final)))
        except Exception:
            return False

    def motivo() -> str:
        d = _safe_num(res.get("diferencial_cd"))
        resultado = str(res.get("resultado_preliminar", ""))
        if d > 0:
            if resultado == "INCIDENCIA_NORMATIVA" or ajuste_legal():
                return "Nivel técnico superior al CD actual, con ajuste al máximo legal admisible"
            return "Nivel técnico superior al CD actual"
        if resultado == "INCIDENCIA_NORMATIVA" or ajuste_legal():
            return "Nivel técnico superior no aplicable por límite legal" if _safe_num(res.get("cd_tecnico_recomendado")) > _safe_num(res.get("cd_tecnico_ajustado")) else "Incidencia normativa en el rango de CD"
        if d < 0:
            return "Nivel técnico inferior al CD actual"
        return "Coincidencia entre CD actual y CD técnico resultante"

    def conclusion() -> str:
        d = _safe_num(res.get("diferencial_cd"))
        resultado = str(res.get("resultado_preliminar", ""))
        if d > 0:
            if resultado == "INCIDENCIA_NORMATIVA" or ajuste_legal():
                return "Procede proponer la adecuación del puesto al CD final admisible resultante del análisis técnico; no procede elevarlo al CD K1 bruto por superar el intervalo legal aplicable al subgrupo de clasificación."
            return "Procede proponer la adecuación del puesto al nivel de complemento de destino resultante del análisis técnico."
        if d < 0:
            return "No procede propuesta de incremento; el análisis técnico apunta a un nivel funcional inferior al actualmente asignado."
        if resultado == "INCIDENCIA_NORMATIVA" or ajuste_legal():
            return "No procede proponer la elevación del nivel de complemento de destino por superar el intervalo legal aplicable al subgrupo de clasificación."
        return "No procede actuación prioritaria sobre el nivel de complemento de destino."

    impacto_html = ""
    try:
        dif_eur = float(impacto.get("diferencial_anual_por_dotacion"))
    except Exception:
        dif_eur = 0.0
    if dif_eur > 0 and impacto.get("impacto_anual_total") is not None:
        impacto_html = f"""
<h2>Estimación económica</h2>
<p>La estimación económica se refiere a las dotaciones efectivamente incluidas para el puesto tipo, con independencia de que dichas dotaciones se encuentren ocupadas o vacantes en el momento del análisis.</p>
<div class="box"><p><strong>Diferencial anual por dotación:</strong> {escape(_fmt(impacto.get('diferencial_anual_por_dotacion',''), 'diferencial_anual_por_dotacion'))} €</p><p><strong>Dotaciones:</strong> {escape(str(impacto.get('dotaciones','')))} · <strong>Impacto anual total:</strong> {escape(_fmt(impacto.get('impacto_anual_total',''), 'impacto_anual_total'))} € · <strong>Impacto periodo:</strong> {escape(_fmt(impacto.get('impacto_periodo',''), 'impacto_periodo'))} €</p><p>{escape(str(impacto.get('motivo','')))}</p></div>
"""

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
<h2>1. Parámetros técnicos del análisis</h2>{parametros_tecnicos_html(result=result)}
<h2>2. Identificación</h2>
<div class="box"><p><strong>Puesto:</strong> {escape(str(ident.get('denominacion_normalizada','')))}</p><p><strong>Grupo/Subgrupo:</strong> {escape(str(ident.get('grupo_subgrupo','')))} · <strong>CD vigente:</strong> {escape(str(ident.get('cd_vigente','')))}</p></div>
<h2>3. Validación y resultado técnico</h2>
<div class="box"><p><strong>Estado rango:</strong> {escape(str(val.get('estado','')))}. {escape(str(val.get('mensaje','')))}</p><p><strong>CD K1 orientativo:</strong> {escape(str(res.get('cd_tecnico_recomendado','')))} · <strong>CD final admisible:</strong> {escape(str(res.get('cd_tecnico_ajustado','')))} · <strong>Diferencial:</strong> {escape(str(res.get('diferencial_cd','')))}</p><p><strong>Motivo técnico:</strong> {escape(motivo())}</p><p><strong>Conclusión técnica:</strong> {escape(conclusion())}</p></div>
<h2>4. Similitud</h2>
<div class="box"><p>Funcional: {escape(str(sims.get('funcional','')))} · Factores CD: {escape(str(sims.get('factores_cd','')))} · Total: {escape(str(sims.get('combinada','')))}</p></div>
<h2>5. Patrones próximos</h2>{table_html(top)}
<h2>6. Comparables internos</h2><p>Se muestran, en su caso, los puestos de la matriz interna con mayor similitud técnica, únicamente como referencia de coherencia comparativa.</p>{table_html(comp.head(10) if not comp.empty else comp)}
{impacto_html}
<h2>Advertencia jurídica</h2><p>El resultado tiene carácter técnico auxiliar y no sustituye el expediente administrativo de modificación de la RPT, que exigirá motivación, memoria económica, negociación cuando proceda, informes preceptivos y aprobación por el órgano competente.</p>
</div></body></html>
"""


def build_global_report(summary: pd.DataFrame, agregado: Dict[str, Any]) -> str:
    dist = distribucion_resultado_informe(summary) if not summary.empty else pd.DataFrame()
    por_grupo = resumen_por_grupo_informe(summary) if not summary.empty else pd.DataFrame()
    prioritarios = build_prioritarios_table(summary)
    impacto_prioritario = build_impacto_table(summary)
    sim_source = summary.copy()
    if not sim_source.empty:
        sim_source["conclusion_tecnica_final"] = sim_source.apply(resultado_informe, axis=1)
    sim_cols = [c for c in ["id_het", "denominacion_normalizada", "grupo_subgrupo", "cd_vigente", "patron_k1", "nombre_patron_k1", "cd_k1_orientativo", "similitud_total_k1", "similitud_funcional_k1", "similitud_factores_cd_k1", "conclusion_tecnica_final"] if c in sim_source.columns]
    similitudes = sim_source[sim_cols].rename(columns={"conclusion_tecnica_final": "Conclusión técnica"}) if sim_cols else pd.DataFrame()
    return f"""
<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Informe global HET-CD RPT</title>
<style>@page{{size:A4 portrait;margin:16mm 12mm}}body{{font-family:Arial,Helvetica,sans-serif;background:#eef1f3;color:#2f3941;margin:0;line-height:1.5}}.page{{max-width:1100px;margin:0 auto;background:white;min-height:100vh;padding:38px 46px;box-shadow:0 8px 30px rgba(0,0,0,.08)}}h1{{color:#0b101d;text-transform:uppercase}} h2{{color:#006089;border-bottom:2px solid #cfd6dc;padding-bottom:5px;margin-top:26px}}.box{{background:#f7f8fa;border:1px solid #cfd6dc;border-left:6px solid #006089;padding:12px 14px;margin:10px 0}}table{{border-collapse:collapse;width:100%;font-size:10.5px;margin-top:10px;page-break-inside:auto}} th,td{{border:1px solid #cfd6dc;padding:5px;vertical-align:top;word-break:break-word}} th{{background:#eef1f3}}</style>
</head><body><div class="page"><h1>Informe global de revisión técnica de CD de la RPT</h1>
<p>El informe resume la evaluación sistemática de los puestos tipo cargados en puestos_vector. La conclusión técnica se basa en el patrón K1 activo más próximo y en el contraste posterior con el rango legal del grupo/subgrupo.</p>
<h2>1. Parámetros técnicos del análisis</h2>{parametros_tecnicos_html(agregado=agregado)}
<h2>2. Resumen ejecutivo</h2>
<div class="box"><p><strong>Puestos analizados:</strong> {escape(str(agregado.get('puestos_analizados','')))}</p><p><strong>Impacto anual total:</strong> {escape(_fmt(agregado.get('impacto_anual_total',''), 'impacto_anual_total'))} €</p><p><strong>Impacto periodo:</strong> {escape(_fmt(agregado.get('impacto_periodo_total',''), 'impacto_periodo_total'))} €</p></div>
<h2>3. Distribución por conclusión técnica</h2><p>Los casos en los que el patrón K1 bruto supera el rango legal, pero el CD final admisible sigue siendo superior al CD vigente, se computan como adecuación al alza.</p>{table_html(dist)}
<h2>4. Resumen por grupo/subgrupo de puestos tipo analizados</h2><p>Este resumen agrupa los puestos tipo analizados por grupo/subgrupo. La columna de puestos tipo analizados no representa dotaciones reales de plantilla, sino el número de perfiles o puestos tipo incorporados al modelo. La estimación económica se calcula sobre las dotaciones efectivamente informadas en la aplicación para cada puesto tipo, con independencia de que dichas dotaciones estén ocupadas o vacantes en el momento del análisis. La columna de dotaciones afectadas permite comprobar cuántas dotaciones reales se han utilizado para calcular el impacto de los puestos tipo con adecuación al alza.</p>{table_html(por_grupo)}
<h2>5. Similitud técnica del conjunto de puestos</h2><p>Se muestra el patrón K1 y las similitudes funcional, de factores CD y combinada para facilitar la revisión del encaje técnico.</p>{table_html(similitudes, max_rows=300)}
<h2>6. Puestos de actuación prioritaria</h2>
<p>Se incluyen en este apartado los puestos en los que el análisis técnico evidencia una diferencia relevante entre el nivel de complemento de destino actualmente asignado y el nivel resultante de la comparación funcional con los puestos patrón, así como aquellos supuestos en los que concurren límites legales o incidencias técnicas que condicionan la propuesta de adecuación.</p>
<p>La conclusión técnica se formula en términos de procedencia o improcedencia de adecuación del nivel de complemento de destino, sin perjuicio de la tramitación administrativa que, en su caso, resulte exigible para la modificación de la Relación de Puestos de Trabajo.</p>
{table_html(prioritarios, max_rows=250)}
<h2>7. Estimación económica de las actuaciones propuestas</h2>
<p>La estimación económica se realiza exclusivamente respecto de aquellos puestos para los que existen datos suficientes de dotaciones y diferencia retributiva asociada al nivel de complemento de destino.</p>
{table_html(impacto_prioritario, max_rows=250)}
<p>La estimación económica se refiere a las dotaciones efectivamente incluidas en la aplicación para cada puesto tipo, con independencia de que dichas dotaciones se encuentren ocupadas o vacantes en el momento del análisis.</p>
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
            distribucion_resultado_informe(summary).to_excel(writer, sheet_name="distribucion_conclusion", index=False)
            resumen_por_grupo_informe(summary).to_excel(writer, sheet_name="resumen_grupo_conclusion", index=False)

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
