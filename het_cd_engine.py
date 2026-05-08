"""
Sistema de Clasificación de Puestos por Similitud Coseno Ponderada
==================================================================
Autor  : Senior Python Developer / Applied Data Scientist
Versión: 1.0.0
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# 49 verbos que forman el espacio dimensional
VERBOS_VALIDOS: List[str] = [
    "administrar",
    "anticipar",
    "representar",
    "definir",
    "aprobar",
    "coordinar",
    "decidir",
    "establecer",
    "evaluar",
    "fomentar",
    "comunicar",
    "planificar",
    "dirigir",
    "motivar",
    "sustentar",
    "colaborar",
    "controlar",
    "retroalimentar",
    "desarrollar",
    "programar",
    "verificar",
    "orientar",
    "proponer",
    "analizar",
    "asesorar",
    "documentar",
    "elaborar",
    "estudiar",
    "identificar",
    "investigar",
    "informar",
    "sistematizar",
    "organizar",
    "supervisar",
    "implementar",
    "facilitar",
    "promover",
    "apoyar",
    "asistir",
    "revisar",
    "optimizar",
    "ejecutar",
    "reportar",
    "participar",
    "aportar",
    "aprender",
    "atender",
    "capacitar",
    "cumplir",
]

assert len(VERBOS_VALIDOS) == 49, "El espacio dimensional debe tener exactamente 49 verbos."

VERBO_INDEX: Dict[str, int] = {v: i for i, v in enumerate(VERBOS_VALIDOS)}

# Categorías jerárquicas y sus pesos
CATEGORIA_PESO: Dict[str, float] = {
    "nuclear":    2.50,
    "relevante":  1.65,
    "apoyo":      1.15,
    "accesorio":  0.80,
}

CATEGORIAS_VALIDAS = set(CATEGORIA_PESO.keys())

# Parámetros globales configurables
ALPHA_DEFAULT: float = 0.7
TOPK_DEFAULT:  int   = 3
EPS:           float = 1e-6


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Verbo:
    """Representa un verbo con su categoría jerárquica."""
    nombre:    str
    categoria: str

    def __post_init__(self) -> None:
        if self.nombre not in VERBO_INDEX:
            raise ValueError(
                f"Verbo inválido: '{self.nombre}'. "
                f"Debe ser uno de los 49 verbos definidos."
            )
        if self.categoria not in CATEGORIAS_VALIDAS:
            raise ValueError(
                f"Categoría inválida: '{self.categoria}'. "
                f"Opciones: {sorted(CATEGORIAS_VALIDAS)}"
            )

    @property
    def peso_jerarquico(self) -> float:
        return CATEGORIA_PESO[self.categoria]

    @property
    def indice(self) -> int:
        return VERBO_INDEX[self.nombre]


@dataclass
class Patron:
    """
    Patrón de puesto de trabajo.

    Attributes
    ----------
    id      : Identificador único.
    nombre  : Nombre del puesto.
    cd      : Código de Denominación (entero o float).
    verbos  : Lista de Verbo que definen el patrón.
    """
    id:     str
    nombre: str
    cd:     float
    verbos: List[Verbo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.verbos:
            raise ValueError(f"El patrón '{self.id}' debe tener al menos un verbo.")
        nombres = [v.nombre for v in self.verbos]
        if len(nombres) != len(set(nombres)):
            raise ValueError(
                f"El patrón '{self.id}' contiene verbos duplicados."
            )

    def vector_binario(self) -> List[int]:
        """Vector binario de 49 posiciones."""
        vec = [0] * 49
        for v in self.verbos:
            vec[v.indice] = 1
        return vec


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def compute_df(patrones: List[Patron]) -> List[int]:
    """
    Calcula la frecuencia de documento (df) para cada verbo.

    df_i = número de patrones que contienen el verbo i.
    Garantiza df_i >= 1 (mínimo 1 para evitar log(0)).

    Returns
    -------
    Lista de 49 enteros con df por verbo.
    """
    if not patrones:
        raise ValueError("La lista de patrones no puede estar vacía.")

    df = [0] * 49
    for patron in patrones:
        vec = patron.vector_binario()
        for i, val in enumerate(vec):
            if val == 1:
                df[i] += 1

    # Garantizar df_i >= 1
    df = [max(d, 1) for d in df]
    return df


def compute_weights(
    patrones:  List[Patron],
    alpha:     float = ALPHA_DEFAULT,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcula los tres tipos de pesos.

    Pesos jerárquicos se derivan de la categoría asignada a cada verbo
    en CADA patrón. Si un verbo aparece con distintas categorías en
    distintos patrones, se usa el máximo peso (política conservadora).
    Para el usuario se aplica peso de categoría máxima observada.

    Returns
    -------
    w_comp : List[float] — peso jerárquico por verbo (49 valores)
    w_idf  : List[float] — peso IDF por verbo (49 valores)
    w      : List[float] — peso total = w_comp * w_idf (49 valores)
    """
    N = len(patrones)
    df = compute_df(patrones)

    # Peso jerárquico: máximo observado en todos los patrones
    w_comp = [0.80] * 49  # accesorio por defecto si no aparece en ningún patrón
    for patron in patrones:
        for verbo in patron.verbos:
            i = verbo.indice
            w_comp[i] = max(w_comp[i], verbo.peso_jerarquico)

    # Peso IDF
    w_idf = [(math.log(N / df[i]) + 1) ** alpha for i in range(49)]

    # Peso total
    w = [w_comp[i] * w_idf[i] for i in range(49)]

    return w_comp, w_idf, w


def cosine(u: List[float], p: List[float]) -> float:
    """
    Similitud coseno entre dos vectores ponderados.

    Raises
    ------
    ZeroDivisionError si la norma del vector usuario es 0.
    ValueError si los vectores tienen distinto tamaño.
    """
    if len(u) != len(p):
        raise ValueError(
            f"Vectores de distinto tamaño: {len(u)} vs {len(p)}"
        )

    dot    = sum(ui * pi for ui, pi in zip(u, p))
    norm_u = math.sqrt(sum(ui ** 2 for ui in u))
    norm_p = math.sqrt(sum(pi ** 2 for pi in p))

    if norm_u == 0.0:
        raise ZeroDivisionError(
            "El vector del usuario tiene norma 0. "
            "El usuario no ha seleccionado ningún verbo válido."
        )
    if norm_p == 0.0:
        # Patrón vacío en el espacio ponderado (no debería ocurrir con validaciones)
        return 0.0

    return dot / (norm_u * norm_p)


def classify(
    verbos_usuario:   List[Verbo],
    patrones:         List[Patron],
    alpha:            float = ALPHA_DEFAULT,
    topk:             int   = TOPK_DEFAULT,
) -> Dict:
    """
    Clasifica al usuario entre los patrones disponibles.

    Parameters
    ----------
    verbos_usuario : Verbos seleccionados por el usuario (con categorías).
    patrones       : Lista de patrones de referencia.
    alpha          : Exponente IDF.
    topk           : Número máximo de mejores coincidencias a devolver.

    Returns
    -------
    Diccionario con:
        cd_pred      : float  — CD discreto predicho.
        cd_cont      : float  — CD continuo (promedio ponderado Top-K).
        top_k        : list   — [(patron_id, nombre, cd, similitud), ...]
        explicacion  : dict   — verbos comunes, solo_usuario, solo_top1.
    """
    # ── Validaciones ───────────────────────────────────────────────────────
    if not patrones:
        raise ValueError("Se requiere al menos un patrón de referencia.")

    topk_efectivo = min(topk, len(patrones))

    # Validar verbos de usuario
    nombres_usuario = [v.nombre for v in verbos_usuario]
    if len(nombres_usuario) != len(set(nombres_usuario)):
        raise ValueError("El usuario tiene verbos duplicados.")

    # ── Pesos ──────────────────────────────────────────────────────────────
    _, _, w = compute_weights(patrones, alpha=alpha)

    # ── Vector usuario ─────────────────────────────────────────────────────
    s = [0] * 49
    for verbo in verbos_usuario:
        s[verbo.indice] = 1

    u = [w[i] * s[i] for i in range(49)]

    # ── Vectores de patrones ponderados ────────────────────────────────────
    patrones_vecs: List[Tuple[Patron, List[float]]] = []
    for patron in patrones:
        t = patron.vector_binario()
        p_vec = [w[i] * t[i] for i in range(49)]
        patrones_vecs.append((patron, p_vec))

    # ── Similitudes ────────────────────────────────────────────────────────
    similitudes: List[Tuple[float, Patron]] = []
    for patron, p_vec in patrones_vecs:
        sim = cosine(u, p_vec)
        similitudes.append((sim, patron))

    similitudes.sort(key=lambda x: x[0], reverse=True)
    top_k_raw = similitudes[:topk_efectivo]

    # ── CD continuo ────────────────────────────────────────────────────────
    suma_sim    = sum(sim for sim, _ in top_k_raw)
    if suma_sim == 0.0:
        # Todos los patrones tienen similitud 0; usar top-1 CD
        cd_cont = top_k_raw[0][1].cd
    else:
        cd_cont = sum(sim * patron.cd for sim, patron in top_k_raw) / suma_sim

    # ── CD discreto ────────────────────────────────────────────────────────
    cds_existentes = {patron.cd for patron in patrones}

    cd_pred: Optional[float] = None
    for cd in cds_existentes:
        if abs(cd_cont - cd) < EPS:
            cd_pred = cd
            break

    if cd_pred is None:
        # Top-1 del Top-K
        cd_pred = top_k_raw[0][1].cd

    # ── Top-K formateado ───────────────────────────────────────────────────
    top_k_out = [
        {
            "patron_id":  patron.id,
            "nombre":     patron.nombre,
            "cd":         patron.cd,
            "similitud":  round(sim, 6),
        }
        for sim, patron in top_k_raw
    ]

    # ── Explicación ────────────────────────────────────────────────────────
    top1_patron = top_k_raw[0][1]
    set_usuario = {v.nombre for v in verbos_usuario}
    set_top1    = {v.nombre for v in top1_patron.verbos}

    comunes         = sorted(set_usuario & set_top1)
    solo_usuario    = sorted(set_usuario - set_top1)
    solo_top1       = sorted(set_top1    - set_usuario)

    explicacion = {
        "top1_patron_id":   top1_patron.id,
        "top1_nombre":      top1_patron.nombre,
        "verbos_comunes":   comunes,
        "solo_en_usuario":  solo_usuario,
        "solo_en_top1":     solo_top1,
    }

    return {
        "cd_pred":     cd_pred,
        "cd_cont":     round(cd_cont, 6),
        "top_k":       top_k_out,
        "explicacion": explicacion,
    }


# ---------------------------------------------------------------------------
# Dataset de ejemplo
# ---------------------------------------------------------------------------

def build_dataset() -> List[Patron]:
    """
    Construye un dataset pequeño de 7 patrones de puestos.
    CD representa el nivel jerárquico (1=operativo … 5=directivo).
    """
    patrones = [
        Patron(
            id="P01", nombre="Director General", cd=5.0,
            verbos=[
                Verbo("planificar",   "nuclear"),
                Verbo("dirigir",      "nuclear"),
                Verbo("liderar",      "nuclear"),
                Verbo("delegar",      "relevante"),
                Verbo("aprobar",      "relevante"),
                Verbo("negociar",     "relevante"),
                Verbo("presupuestar", "apoyo"),
                Verbo("reportar",     "accesorio"),
            ],
        ),
        Patron(
            id="P02", nombre="Gerente de Área", cd=4.0,
            verbos=[
                Verbo("coordinar",   "nuclear"),
                Verbo("supervisar",  "nuclear"),
                Verbo("gestionar",   "nuclear"),
                Verbo("evaluar",     "relevante"),
                Verbo("planificar",  "relevante"),
                Verbo("comunicar",   "apoyo"),
                Verbo("reportar",    "apoyo"),
                Verbo("delegar",     "accesorio"),
            ],
        ),
        Patron(
            id="P03", nombre="Jefe de Departamento", cd=3.0,
            verbos=[
                Verbo("organizar",   "nuclear"),
                Verbo("controlar",   "nuclear"),
                Verbo("supervisar",  "relevante"),
                Verbo("capacitar",   "relevante"),
                Verbo("evaluar",     "relevante"),
                Verbo("documentar",  "apoyo"),
                Verbo("validar",     "apoyo"),
                Verbo("registrar",   "accesorio"),
            ],
        ),
        Patron(
            id="P04", nombre="Analista Senior", cd=2.0,
            verbos=[
                Verbo("analizar",     "nuclear"),
                Verbo("investigar",   "nuclear"),
                Verbo("diseñar",      "nuclear"),
                Verbo("desarrollar",  "relevante"),
                Verbo("elaborar",     "relevante"),
                Verbo("documentar",   "apoyo"),
                Verbo("proponer",     "apoyo"),
                Verbo("verificar",    "accesorio"),
            ],
        ),
        Patron(
            id="P05", nombre="Técnico Especialista", cd=2.0,
            verbos=[
                Verbo("implementar",  "nuclear"),
                Verbo("ejecutar",     "nuclear"),
                Verbo("operar",       "nuclear"),
                Verbo("mantener",     "relevante"),
                Verbo("diagnosticar", "relevante"),
                Verbo("instalar",     "apoyo"),
                Verbo("verificar",    "apoyo"),
                Verbo("registrar",    "accesorio"),
            ],
        ),
        Patron(
            id="P06", nombre="Asistente Administrativo", cd=1.0,
            verbos=[
                Verbo("procesar",    "nuclear"),
                Verbo("registrar",   "nuclear"),
                Verbo("archivar",    "nuclear"),
                Verbo("clasificar",  "relevante"),
                Verbo("atender",     "relevante"),
                Verbo("documentar",  "apoyo"),
                Verbo("comunicar",   "accesorio"),
            ],
        ),
        Patron(
            id="P07", nombre="Auditor Interno", cd=3.0,
            verbos=[
                Verbo("auditar",    "nuclear"),
                Verbo("revisar",    "nuclear"),
                Verbo("verificar",  "nuclear"),
                Verbo("analizar",   "relevante"),
                Verbo("evaluar",    "relevante"),
                Verbo("reportar",   "apoyo"),
                Verbo("certificar", "apoyo"),
                Verbo("documentar", "accesorio"),
            ],
        ),
    ]
    return patrones


# ---------------------------------------------------------------------------
# Tests unitarios
# ---------------------------------------------------------------------------

class TestComputeDf(unittest.TestCase):

    def setUp(self):
        self.patrones = build_dataset()

    def test_df_length(self):
        df = compute_df(self.patrones)
        self.assertEqual(len(df), 49)

    def test_df_minimum_one(self):
        df = compute_df(self.patrones)
        for val in df:
            self.assertGreaterEqual(val, 1)

    def test_df_planificar(self):
        # "planificar" aparece en P01 y P02
        df = compute_df(self.patrones)
        idx = VERBO_INDEX["planificar"]
        self.assertEqual(df[idx], 2)

    def test_empty_patrones_raises(self):
        with self.assertRaises(ValueError):
            compute_df([])


class TestComputeWeights(unittest.TestCase):

    def setUp(self):
        self.patrones = build_dataset()

    def test_weights_length(self):
        w_comp, w_idf, w = compute_weights(self.patrones)
        self.assertEqual(len(w_comp), 49)
        self.assertEqual(len(w_idf),  49)
        self.assertEqual(len(w),      49)

    def test_weights_positive(self):
        _, _, w = compute_weights(self.patrones)
        for val in w:
            self.assertGreater(val, 0)

    def test_alpha_effect(self):
        _, w_idf_07, _ = compute_weights(self.patrones, alpha=0.7)
        _, w_idf_10, _ = compute_weights(self.patrones, alpha=1.0)
        # Con alpha mayor, el IDF de términos raros debe ser mayor
        idx = VERBO_INDEX["liderar"]  # solo en P01
        self.assertGreater(w_idf_10[idx], w_idf_07[idx])


class TestCosine(unittest.TestCase):

    def test_identical_vectors(self):
        u = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine(u, u), 1.0, places=10)

    def test_orthogonal_vectors(self):
        u = [1.0, 0.0]
        p = [0.0, 1.0]
        self.assertAlmostEqual(cosine(u, p), 0.0, places=10)

    def test_zero_user_vector_raises(self):
        with self.assertRaises(ZeroDivisionError):
            cosine([0.0, 0.0], [1.0, 2.0])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine([1.0, 2.0], [1.0])

    def test_range(self):
        import random
        random.seed(42)
        u = [random.random() for _ in range(49)]
        p = [random.random() for _ in range(49)]
        sim = cosine(u, p)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim,  1.0)


class TestVerboDataclass(unittest.TestCase):

    def test_valid_verbo(self):
        v = Verbo("planificar", "nuclear")
        self.assertEqual(v.peso_jerarquico, 2.50)
        self.assertEqual(v.indice, VERBO_INDEX["planificar"])

    def test_invalid_nombre_raises(self):
        with self.assertRaises(ValueError):
            Verbo("volar", "nuclear")

    def test_invalid_categoria_raises(self):
        with self.assertRaises(ValueError):
            Verbo("planificar", "superimportante")


class TestPatronDataclass(unittest.TestCase):

    def test_valid_patron(self):
        p = Patron(
            id="T01", nombre="Test", cd=1.0,
            verbos=[Verbo("planificar", "nuclear")],
        )
        self.assertEqual(len(p.vector_binario()), 49)

    def test_empty_verbos_raises(self):
        with self.assertRaises(ValueError):
            Patron(id="T01", nombre="Test", cd=1.0, verbos=[])

    def test_duplicate_verbos_raises(self):
        with self.assertRaises(ValueError):
            Patron(
                id="T01", nombre="Test", cd=1.0,
                verbos=[
                    Verbo("planificar", "nuclear"),
                    Verbo("planificar", "relevante"),
                ],
            )

    def test_vector_binario_correct(self):
        v = Verbo("planificar", "nuclear")
        p = Patron(id="T01", nombre="Test", cd=1.0, verbos=[v])
        vec = p.vector_binario()
        self.assertEqual(vec[VERBO_INDEX["planificar"]], 1)
        self.assertEqual(sum(vec), 1)


class TestClassify(unittest.TestCase):

    def setUp(self):
        self.patrones = build_dataset()

    def test_output_keys(self):
        usuario = [
            Verbo("planificar",  "nuclear"),
            Verbo("dirigir",     "nuclear"),
            Verbo("liderar",     "nuclear"),
            Verbo("negociar",    "relevante"),
        ]
        result = classify(usuario, self.patrones)
        self.assertIn("cd_pred",     result)
        self.assertIn("cd_cont",     result)
        self.assertIn("top_k",       result)
        self.assertIn("explicacion", result)

    def test_cd_pred_in_existing_cds(self):
        usuario = [
            Verbo("analizar",   "nuclear"),
            Verbo("investigar", "nuclear"),
            Verbo("documentar", "apoyo"),
        ]
        result   = classify(usuario, self.patrones)
        cds_existentes = {p.cd for p in self.patrones}
        self.assertIn(result["cd_pred"], cds_existentes)

    def test_topk_length(self):
        usuario = [Verbo("planificar", "nuclear")]
        result  = classify(usuario, self.patrones, topk=3)
        self.assertLessEqual(len(result["top_k"]), 3)

    def test_top_k_sorted_descending(self):
        usuario = [
            Verbo("organizar",  "nuclear"),
            Verbo("controlar",  "nuclear"),
            Verbo("supervisar", "relevante"),
        ]
        result = classify(usuario, self.patrones)
        sims   = [item["similitud"] for item in result["top_k"]]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_explicacion_keys(self):
        usuario = [Verbo("planificar", "nuclear")]
        result  = classify(usuario, self.patrones)
        exp     = result["explicacion"]
        self.assertIn("verbos_comunes",  exp)
        self.assertIn("solo_en_usuario", exp)
        self.assertIn("solo_en_top1",    exp)

    def test_zero_user_vector_raises(self):
        # Usuario con verbos que NO están en el sistema — imposible por validación de Verbo,
        # pero si s es todo ceros en la ponderación no puede ocurrir con verbos válidos.
        # Simulamos directamente:
        with self.assertRaises((ZeroDivisionError, ValueError)):
            cosine([0.0] * 49, [1.0] * 49)

    def test_empty_patrones_raises(self):
        usuario = [Verbo("planificar", "nuclear")]
        with self.assertRaises(ValueError):
            classify(usuario, [])

    def test_topk_larger_than_n(self):
        # Solo 2 patrones, topk=3 → debe usar 2
        patrones_mini = build_dataset()[:2]
        usuario = [Verbo("planificar", "nuclear")]
        result  = classify(usuario, patrones_mini, topk=3)
        self.assertLessEqual(len(result["top_k"]), 2)


# ---------------------------------------------------------------------------
# Función principal / Demo
# ---------------------------------------------------------------------------

def _print_separator(char: str = "─", width: int = 65) -> None:
    print(char * width)


def _print_result(result: Dict, label: str = "") -> None:
    _print_separator()
    if label:
        print(f"  CASO: {label}")
        _print_separator()
    print(f"  CD Continuo  : {result['cd_cont']:.4f}")
    print(f"  CD Predicho  : {result['cd_pred']}")
    print()
    print("  TOP-K resultados:")
    for i, item in enumerate(result["top_k"], 1):
        print(
            f"    {i}. [{item['patron_id']}] {item['nombre']:<30} "
            f"CD={item['cd']}  sim={item['similitud']:.4f}"
        )
    print()
    exp = result["explicacion"]
    print(f"  Top-1 patrón : [{exp['top1_patron_id']}] {exp['top1_nombre']}")
    print(f"  Verbos comunes    ({len(exp['verbos_comunes']):2d}): "
          f"{', '.join(exp['verbos_comunes']) or '—'}")
    print(f"  Solo en usuario   ({len(exp['solo_en_usuario']):2d}): "
          f"{', '.join(exp['solo_en_usuario']) or '—'}")
    print(f"  Solo en Top-1     ({len(exp['solo_en_top1']):2d}): "
          f"{', '.join(exp['solo_en_top1']) or '—'}")
    _print_separator()


def main() -> None:
    print()
    print("=" * 65)
    print("  SISTEMA DE CLASIFICACIÓN DE PUESTOS")
    print("  Similitud Coseno Ponderada (IDF + Jerarquía)")
    print("=" * 65)

    patrones = build_dataset()

    print(f"\n  Dataset: {len(patrones)} patrones cargados.")
    print(f"  Dimensiones: {len(VERBOS_VALIDOS)} verbos")
    print(f"  Alpha IDF: {ALPHA_DEFAULT}  |  TopK: {TOPK_DEFAULT}")

    # ── Caso 1: Perfil directivo ───────────────────────────────────────────
    usuario_1 = [
        Verbo("planificar",   "nuclear"),
        Verbo("dirigir",      "nuclear"),
        Verbo("liderar",      "nuclear"),
        Verbo("delegar",      "relevante"),
        Verbo("negociar",     "relevante"),
        Verbo("presupuestar", "apoyo"),
    ]
    res1 = classify(usuario_1, patrones)
    _print_result(res1, label="Perfil Directivo")

    # ── Caso 2: Perfil analítico / técnico ────────────────────────────────
    usuario_2 = [
        Verbo("analizar",    "nuclear"),
        Verbo("investigar",  "nuclear"),
        Verbo("diseñar",     "relevante"),
        Verbo("desarrollar", "relevante"),
        Verbo("documentar",  "apoyo"),
        Verbo("verificar",   "accesorio"),
    ]
    res2 = classify(usuario_2, patrones)
    _print_result(res2, label="Perfil Analítico")

    # ── Caso 3: Perfil operativo ───────────────────────────────────────────
    usuario_3 = [
        Verbo("procesar",   "nuclear"),
        Verbo("registrar",  "nuclear"),
        Verbo("archivar",   "nuclear"),
        Verbo("clasificar", "relevante"),
        Verbo("atender",    "relevante"),
    ]
    res3 = classify(usuario_3, patrones)
    _print_result(res3, label="Perfil Operativo")

    # ── Caso 4: Perfil mixto / auditoria ──────────────────────────────────
    usuario_4 = [
        Verbo("auditar",    "nuclear"),
        Verbo("revisar",    "nuclear"),
        Verbo("analizar",   "relevante"),
        Verbo("evaluar",    "relevante"),
        Verbo("supervisar", "apoyo"),
        Verbo("reportar",   "accesorio"),
    ]
    res4 = classify(usuario_4, patrones)
    _print_result(res4, label="Perfil Auditoría / Control")

    # ── Pesos IDF (diagnóstico) ────────────────────────────────────────────
    print("\n  Diagnóstico IDF — verbos con mayor discriminación:\n")
    _, w_idf, w = compute_weights(patrones)
    ranked = sorted(
        [(VERBOS_VALIDOS[i], w_idf[i], w[i]) for i in range(49)],
        key=lambda x: x[1], reverse=True
    )
    print(f"  {'Verbo':<20} {'w_idf':>8}  {'w_total':>8}")
    _print_separator("-", 42)
    for nombre, widf, wtotal in ranked[:10]:
        print(f"  {nombre:<20} {widf:>8.4f}  {wtotal:>8.4f}")
    print()

    # ── Tests unitarios ────────────────────────────────────────────────────
    print("  Ejecutando tests unitarios...\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestComputeDf, TestComputeWeights, TestCosine,
        TestVerboDataclass, TestPatronDataclass, TestClassify,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Motor HET-CD v1 — evaluación técnica ampliada del Complemento de Destino
# ---------------------------------------------------------------------------
# Este bloque mantiene la compatibilidad con el motor clásico anterior y añade
# funciones puras para trabajar con matrices vectorizadas F1-F49 + CD1_1-CD6_6.
# La función principal es classify_het_cd().

from typing import Any

# Columnas funcionales F1-F49. Por convención, F1 corresponde al primer verbo
# de VERBOS_VALIDOS, F2 al segundo, etc.
F_COLUMNS: List[str] = [f"F{i}" for i in range(1, 50)]
F_COLUMN_BY_VERBO: Dict[str, str] = {
    verbo: f"F{i + 1}" for i, verbo in enumerate(VERBOS_VALIDOS)
}
VERBO_BY_F_COLUMN: Dict[str, str] = {
    f"F{i + 1}": verbo for i, verbo in enumerate(VERBOS_VALIDOS)
}

# Factores madre CD y subfactores esperados.
CD_FACTOR_PREFIXES: Dict[str, str] = {
    "CD1": "Especialización",
    "CD2": "Responsabilidad",
    "CD3": "Competencia",
    "CD4": "Mando",
    "CD5": "Complejidad funcional",
    "CD6": "Complejidad territorial/organizativa",
}

CD_SUBFACTOR_COLUMNS: Dict[str, List[str]] = {
    "CD1": [f"CD1_{i}" for i in range(1, 6)],
    "CD2": [f"CD2_{i}" for i in range(1, 8)],
    "CD3": [f"CD3_{i}" for i in range(1, 8)],
    "CD4": [f"CD4_{i}" for i in range(1, 8)],
    "CD5": [f"CD5_{i}" for i in range(1, 8)],
    "CD6": [f"CD6_{i}" for i in range(1, 7)],
}

CD_COLUMNS: List[str] = [
    col for cols in CD_SUBFACTOR_COLUMNS.values() for col in cols
]

DEFAULT_BLOCK_WEIGHTS: Dict[str, float] = {
    "funcional": 0.40,
    "cd": 0.60,
}

DEFAULT_CD_FACTOR_WEIGHTS: Dict[str, float] = {
    "CD1": 0.15,
    "CD2": 0.25,
    "CD3": 0.15,
    "CD4": 0.20,
    "CD5": 0.20,
    "CD6": 0.05,
}

DEFAULT_CD_SUBFACTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "CD1": {"CD1_1": 0.35, "CD1_2": 0.25, "CD1_3": 0.20, "CD1_4": 0.10, "CD1_5": 0.10},
    "CD2": {"CD2_1": 0.20, "CD2_2": 0.20, "CD2_3": 0.15, "CD2_4": 0.10, "CD2_5": 0.15, "CD2_6": 0.15, "CD2_7": 0.05},
    "CD3": {"CD3_1": 0.25, "CD3_2": 0.20, "CD3_3": 0.20, "CD3_4": 0.15, "CD3_5": 0.10, "CD3_6": 0.05, "CD3_7": 0.05},
    "CD4": {"CD4_1": 0.25, "CD4_2": 0.20, "CD4_3": 0.15, "CD4_4": 0.15, "CD4_5": 0.15, "CD4_6": 0.05, "CD4_7": 0.05},
    "CD5": {"CD5_1": 0.15, "CD5_2": 0.20, "CD5_3": 0.15, "CD5_4": 0.15, "CD5_5": 0.20, "CD5_6": 0.10, "CD5_7": 0.05},
    "CD6": {"CD6_1": 0.20, "CD6_2": 0.20, "CD6_3": 0.15, "CD6_4": 0.20, "CD6_5": 0.15, "CD6_6": 0.10},
}

RANGO_OK = "OK"
FUERA_RANGO_INFERIOR = "FUERA_RANGO_INFERIOR"
FUERA_RANGO_SUPERIOR = "FUERA_RANGO_SUPERIOR"
SIN_RANGO_DEFINIDO = "SIN_RANGO_DEFINIDO"
GRUPO_NO_VALIDO = "GRUPO_NO_VALIDO"


def _is_na_value(value: Any) -> bool:
    """Devuelve True si el valor debe tratarse como N.A./vacío."""
    if value is None:
        return True
    try:
        # Compatible con pandas/numpy sin obligar a importar pandas aquí.
        if value != value:  # NaN
            return True
    except Exception:
        pass
    txt = str(value).strip().lower()
    return txt in {"", "na", "n.a", "n.a.", "n/a", "nan", "none", "null", "no aplica", "no aplicable"}


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convierte valores a float admitiendo coma decimal y vacíos."""
    if _is_na_value(value):
        return default
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    num = _to_float(value, None)
    if num is None:
        return default
    try:
        return int(round(num))
    except Exception:
        return default


def _norm_col_name(col: Any) -> str:
    return str(col).strip()


def row_get(row: Any, key: str, default: Any = None) -> Any:
    """
    Acceso robusto a filas tipo dict, pandas.Series o similares.
    Intenta coincidencia exacta y, después, coincidencia case-insensitive.
    """
    if row is None:
        return default

    try:
        if key in row:
            return row[key]
    except Exception:
        pass

    try:
        keys = list(row.keys())
        key_low = key.strip().lower()
        for k in keys:
            if str(k).strip().lower() == key_low:
                return row[k]
    except Exception:
        pass

    return default


def dataframe_columns(df_like: Any) -> List[str]:
    """Devuelve nombres de columnas para pandas.DataFrame o lista de dicts."""
    if df_like is None:
        return []
    if hasattr(df_like, "columns"):
        return [str(c).strip() for c in list(df_like.columns)]
    if isinstance(df_like, list) and df_like:
        if isinstance(df_like[0], dict):
            return [str(c).strip() for c in df_like[0].keys()]
    return []


def iter_rows(df_like: Any) -> List[Any]:
    """Itera filas de pandas.DataFrame o lista de dicts devolviendo objetos fila."""
    if df_like is None:
        return []
    if hasattr(df_like, "iterrows"):
        return [row for _, row in df_like.iterrows()]
    if isinstance(df_like, list):
        return df_like
    return []


def detectar_columnas_vectoriales(columns: List[str]) -> Dict[str, List[str]]:
    """Detecta columnas funcionales y de subfactores CD disponibles."""
    cols = {_norm_col_name(c): c for c in columns}
    f_cols = [c for c in F_COLUMNS if c in cols]
    cd_cols = [c for c in CD_COLUMNS if c in cols]
    return {"F": f_cols, "CD": cd_cols, "VECTOR": f_cols + cd_cols}


def validar_dimensiones_vectoriales(puestos_vector: Any, patrones_vector: Any) -> Dict[str, Any]:
    """Comprueba que puestos_vector y patrones_vector comparten F1-F49 y CD1_1-CD6_6."""
    puestos_cols = set(dataframe_columns(puestos_vector))
    patrones_cols = set(dataframe_columns(patrones_vector))

    required = set(F_COLUMNS + CD_COLUMNS)
    missing_puestos = sorted(required - puestos_cols)
    missing_patrones = sorted(required - patrones_cols)
    common_vector = sorted(required & puestos_cols & patrones_cols, key=lambda x: (x[:2], x))

    ok = not missing_puestos and not missing_patrones
    return {
        "ok": ok,
        "missing_puestos": missing_puestos,
        "missing_patrones": missing_patrones,
        "common_vector_columns": common_vector,
    }


def normalizar_verbo_value(value: Any) -> float:
    """Normaliza verbos F1-F49: cualquier valor positivo se considera 1; el resto 0."""
    num = _to_float(value, 0.0)
    return 1.0 if num and num > 0 else 0.0


def normalizar_subfactor_cd(value: Any) -> Optional[float]:
    """
    Normaliza subfactores CD:
      1 -> 0.0
      2 -> 0.5
      3 -> 1.0
      N.A./vacío -> None, excluido del denominador.
    """
    if _is_na_value(value):
        return None
    num = _to_float(value, None)
    if num is None:
        return None
    if num <= 1:
        return 0.0
    if num == 2:
        return 0.5
    if num >= 3:
        return 1.0
    # Para valores intermedios accidentales, se acota a [0, 1].
    return max(0.0, min(1.0, (num - 1.0) / 2.0))


def cosine_optional(u: List[Optional[float]], p: List[Optional[float]], weights: Optional[List[float]] = None) -> float:
    """
    Coseno ponderado que ignora dimensiones None en cualquiera de los vectores.
    Si no queda ninguna dimensión comparable, devuelve 0.0.
    """
    if len(u) != len(p):
        raise ValueError(f"Vectores de distinto tamaño: {len(u)} vs {len(p)}")
    if weights is None:
        weights = [1.0] * len(u)
    if len(weights) != len(u):
        raise ValueError("La lista de pesos no coincide con la dimensión vectorial.")

    dot = 0.0
    norm_u = 0.0
    norm_p = 0.0
    comparable = 0
    for ui, pi, wi in zip(u, p, weights):
        if ui is None or pi is None:
            continue
        comparable += 1
        uw = float(ui) * float(wi)
        pw = float(pi) * float(wi)
        dot += uw * pw
        norm_u += uw ** 2
        norm_p += pw ** 2

    if comparable == 0 or norm_u == 0.0 or norm_p == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_u) * math.sqrt(norm_p))


def build_functional_vector_from_row(row: Any, f_cols: Optional[List[str]] = None) -> List[float]:
    """Construye vector F1-F49 en 0/1 desde una fila."""
    cols = f_cols or F_COLUMNS
    return [normalizar_verbo_value(row_get(row, c, 0)) for c in cols]


def build_cd_vector_from_row(row: Any, cd_cols: Optional[List[str]] = None) -> List[Optional[float]]:
    """Construye vector CD1_1-CD6_6 normalizado en 0/0.5/1/None."""
    cols = cd_cols or CD_COLUMNS
    return [normalizar_subfactor_cd(row_get(row, c, None)) for c in cols]


def _renormalize_weights(weights: Dict[str, float], active_keys: List[str]) -> Dict[str, float]:
    total = sum(float(weights.get(k, 0.0)) for k in active_keys)
    if total <= 0:
        if not active_keys:
            return {}
        equal = 1.0 / len(active_keys)
        return {k: equal for k in active_keys}
    return {k: float(weights.get(k, 0.0)) / total for k in active_keys}


def calcular_score_factor_madre(row: Any, factor: str, subfactor_weights: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[float]:
    """
    Calcula el score agregado de un factor madre CD1-CD6.
    Devuelve None si ningún subfactor aplica.
    """
    subfactor_weights = subfactor_weights or DEFAULT_CD_SUBFACTOR_WEIGHTS
    cols = CD_SUBFACTOR_COLUMNS.get(factor, [])
    active_values: Dict[str, float] = {}
    for col in cols:
        val = normalizar_subfactor_cd(row_get(row, col, None))
        if val is not None:
            active_values[col] = val

    if not active_values:
        return None

    weights = _renormalize_weights(subfactor_weights.get(factor, {}), list(active_values.keys()))
    return sum(active_values[col] * weights[col] for col in active_values)


def calcular_scores_factores_madre(row: Any, subfactor_weights: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Optional[float]]:
    return {
        factor: calcular_score_factor_madre(row, factor, subfactor_weights=subfactor_weights)
        for factor in CD_SUBFACTOR_COLUMNS.keys()
    }


def calcular_similitud_funcional_vector(puesto_row: Any, patron_row: Any, f_cols: Optional[List[str]] = None) -> float:
    """Similitud coseno simple sobre F1-F49."""
    cols = f_cols or F_COLUMNS
    u = build_functional_vector_from_row(puesto_row, cols)
    p = build_functional_vector_from_row(patron_row, cols)
    try:
        return cosine(u, p)
    except ZeroDivisionError:
        return 0.0


def calcular_similitud_factores_cd(
    puesto_row: Any,
    patron_row: Any,
    cd_factor_weights: Optional[Dict[str, float]] = None,
    cd_subfactor_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Calcula similitud de factores CD a partir de los scores agregados CD1-CD6.
    Si CD6 u otro factor no aplica en alguna parte, se excluye y se redistribuyen pesos.
    """
    cd_factor_weights = cd_factor_weights or DEFAULT_CD_FACTOR_WEIGHTS
    puesto_scores = calcular_scores_factores_madre(puesto_row, cd_subfactor_weights)
    patron_scores = calcular_scores_factores_madre(patron_row, cd_subfactor_weights)

    active_factors = [
        f for f in CD_SUBFACTOR_COLUMNS.keys()
        if puesto_scores.get(f) is not None and patron_scores.get(f) is not None
    ]
    if not active_factors:
        return 0.0

    weights_norm = _renormalize_weights(cd_factor_weights, active_factors)
    u = [puesto_scores[f] for f in active_factors]
    p = [patron_scores[f] for f in active_factors]
    w = [weights_norm[f] for f in active_factors]
    return cosine_optional(u, p, w)


def calcular_similitud_het_cd(
    puesto_row: Any,
    patron_row: Any,
    block_weights: Optional[Dict[str, float]] = None,
    cd_factor_weights: Optional[Dict[str, float]] = None,
    cd_subfactor_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Calcula similitud funcional, similitud CD y similitud total HET-CD."""
    block_weights = block_weights or DEFAULT_BLOCK_WEIGHTS
    sim_func = calcular_similitud_funcional_vector(puesto_row, patron_row)
    sim_cd = calcular_similitud_factores_cd(
        puesto_row,
        patron_row,
        cd_factor_weights=cd_factor_weights,
        cd_subfactor_weights=cd_subfactor_weights,
    )
    total = (
        float(block_weights.get("funcional", 0.40)) * sim_func
        + float(block_weights.get("cd", 0.60)) * sim_cd
    )
    return {
        "funcional": round(sim_func, 6),
        "factores_cd": round(sim_cd, 6),
        "total": round(total, 6),
    }


def validar_rango_cd(grupo_subgrupo: Any, cd_vigente: Any, rangos_cd: Any) -> Dict[str, Any]:
    """Valida el CD vigente contra la tabla de rangos por grupo/subgrupo."""
    grupo = str(grupo_subgrupo).strip().upper()
    cd = _to_int(cd_vigente, None)

    if not grupo:
        return {"estado": GRUPO_NO_VALIDO, "cd_min": None, "cd_max": None, "mensaje": "Grupo/subgrupo vacío o no válido."}
    if cd is None:
        return {"estado": SIN_RANGO_DEFINIDO, "cd_min": None, "cd_max": None, "mensaje": "CD vigente vacío o no numérico."}

    rows = iter_rows(rangos_cd)
    for row in rows:
        row_grupo = str(
            row_get(row, "grupo_subgrupo", row_get(row, "grupo", row_get(row, "subgrupo", "")))
        ).strip().upper()
        if row_grupo == grupo:
            cd_min = _to_int(row_get(row, "cd_min", row_get(row, "min", row_get(row, "nivel_min", None))), None)
            cd_max = _to_int(row_get(row, "cd_max", row_get(row, "max", row_get(row, "nivel_max", None))), None)
            if cd_min is None or cd_max is None:
                return {"estado": SIN_RANGO_DEFINIDO, "cd_min": cd_min, "cd_max": cd_max, "mensaje": f"Rango incompleto para {grupo}."}
            if cd < cd_min:
                return {"estado": FUERA_RANGO_INFERIOR, "cd_min": cd_min, "cd_max": cd_max, "mensaje": f"El CD vigente {cd} está por debajo del mínimo {cd_min} para {grupo}."}
            if cd > cd_max:
                return {"estado": FUERA_RANGO_SUPERIOR, "cd_min": cd_min, "cd_max": cd_max, "mensaje": f"El CD vigente {cd} supera el máximo {cd_max} para {grupo}."}
            return {"estado": RANGO_OK, "cd_min": cd_min, "cd_max": cd_max, "mensaje": f"El CD vigente {cd} se encuentra dentro del rango legal aplicable a {grupo}."}

    return {"estado": SIN_RANGO_DEFINIDO, "cd_min": None, "cd_max": None, "mensaje": f"No existe rango definido para el grupo/subgrupo {grupo}."}


def ajustar_cd_a_rango(cd_recomendado: Any, validacion_rango: Dict[str, Any]) -> Optional[int]:
    """Ajusta el CD recomendado a los límites normativos disponibles."""
    cd = _to_int(cd_recomendado, None)
    if cd is None:
        return None
    cd_min = validacion_rango.get("cd_min")
    cd_max = validacion_rango.get("cd_max")
    if cd_min is not None:
        cd = max(cd, int(cd_min))
    if cd_max is not None:
        cd = min(cd, int(cd_max))
    return cd



def patron_activo_para_calculo(patron_row: Any) -> bool:
    """Indica si un patrón debe intervenir en K1 y similitudes.

    Si la columna activo_calculo no existe o está vacía, se mantiene compatibilidad
    con versiones anteriores y el patrón se considera activo. Si existe, solo se
    aceptan valores afirmativos (SÍ/SI/S/TRUE/1/YES/Y).
    """
    valor = row_get(patron_row, "activo_calculo", None)
    if valor is None:
        return True
    texto = str(valor).strip().upper()
    if texto == "":
        return True
    return texto in {"SÍ", "SI", "S", "TRUE", "1", "YES", "Y"}


def contar_patrones_inactivos(patrones_vector: Any) -> int:
    """Cuenta patrones desactivados por activo_calculo = NO, si la columna existe."""
    total = 0
    for patron_row in iter_rows(patrones_vector):
        if not patron_activo_para_calculo(patron_row):
            total += 1
    return total

def obtener_top_k_patrones(
    puesto_row: Any,
    patrones_vector: Any,
    topk: int = 5,
    block_weights: Optional[Dict[str, float]] = None,
    cd_factor_weights: Optional[Dict[str, float]] = None,
    cd_subfactor_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Devuelve los patrones más similares al puesto evaluado."""
    resultados: List[Dict[str, Any]] = []
    for patron_row in iter_rows(patrones_vector):
        if not patron_activo_para_calculo(patron_row):
            continue
        sims = calcular_similitud_het_cd(
            puesto_row,
            patron_row,
            block_weights=block_weights,
            cd_factor_weights=cd_factor_weights,
            cd_subfactor_weights=cd_subfactor_weights,
        )
        cd_ref = _to_int(row_get(patron_row, "cd_referencia", row_get(patron_row, "cd", None)), None)
        resultados.append({
            "patron_id": row_get(patron_row, "patron_id", row_get(patron_row, "id", "")),
            "nombre_patron": row_get(patron_row, "nombre_patron", row_get(patron_row, "nombre", "")),
            "cd_referencia": cd_ref,
            "similitud_funcional": sims["funcional"],
            "similitud_cd": sims["factores_cd"],
            "similitud_total": sims["total"],
        })

    resultados.sort(key=lambda x: x["similitud_total"], reverse=True)
    return resultados[:max(1, int(topk))]


def recomendar_cd_desde_top_k(top_k_patrones: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula el CD recomendado a partir del patrón más próximo.

    Decisión v2:
    - El CD técnico recomendado se toma del Top-1 para mantener coherencia explicativa:
      si el patrón más similar es CD 30, la recomendación principal debe ser CD 30.
    - Se conserva, solo como dato auxiliar, el CD continuo Top-K calculado por media
      ponderada de similitudes. Ese valor sirve para detectar zonas límite o dispersión,
      pero no desplaza automáticamente al patrón más próximo.
    """
    validos = [p for p in top_k_patrones if p.get("cd_referencia") is not None]
    if not validos:
        return {
            "cd_continuo": None,
            "cd_recomendado": None,
            "cd_top1": None,
            "metodo": "SIN_PATRONES_VALIDOS",
        }

    top1 = validos[0]
    cd_top1 = int(top1["cd_referencia"])

    suma_sim = sum(max(0.0, float(p.get("similitud_total", 0.0))) for p in validos)
    if suma_sim <= 0:
        cd_cont = float(cd_top1)
    else:
        cd_cont = sum(
            float(p["cd_referencia"]) * max(0.0, float(p.get("similitud_total", 0.0)))
            for p in validos
        ) / suma_sim

    return {
        "cd_continuo": round(cd_cont, 6),
        "cd_recomendado": cd_top1,
        "cd_top1": cd_top1,
        "patron_top1": top1.get("patron_id"),
        "similitud_top1": top1.get("similitud_total"),
        "metodo": "TOP1_PATRON_MAS_PROXIMO",
    }


def etiqueta_riesgo_arrastre(similitud: float) -> str:
    if similitud >= 0.90:
        return "MUY_ALTO"
    if similitud >= 0.85:
        return "ALTO"
    if similitud >= 0.80:
        return "MEDIO"
    return "BAJO"


def calcular_comparables_internos(
    puesto_row: Any,
    puestos_vector: Any,
    topk: int = 10,
    exclude_same_id: bool = True,
    block_weights: Optional[Dict[str, float]] = None,
    cd_factor_weights: Optional[Dict[str, float]] = None,
    cd_subfactor_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Calcula puestos reales más similares para estimar efecto arrastre."""
    id_base = str(row_get(puesto_row, "id_het", "")).strip()
    comparables: List[Dict[str, Any]] = []

    for other in iter_rows(puestos_vector):
        other_id = str(row_get(other, "id_het", "")).strip()
        if exclude_same_id and id_base and other_id == id_base:
            continue
        sims = calcular_similitud_het_cd(
            puesto_row,
            other,
            block_weights=block_weights,
            cd_factor_weights=cd_factor_weights,
            cd_subfactor_weights=cd_subfactor_weights,
        )
        sim_total = sims["total"]
        comparables.append({
            "id_het": other_id,
            "denominacion_normalizada": row_get(other, "denominacion_normalizada", row_get(other, "denominacion", "")),
            "grupo_subgrupo": row_get(other, "grupo_subgrupo", ""),
            "cd_vigente": _to_int(row_get(other, "cd_vigente", None), None),
            "similitud_funcional": sims["funcional"],
            "similitud_cd": sims["factores_cd"],
            "similitud_total": sim_total,
            "riesgo": etiqueta_riesgo_arrastre(sim_total),
        })

    comparables.sort(key=lambda x: x["similitud_total"], reverse=True)
    return comparables[:max(1, int(topk))]


def calcular_efecto_arrastre(comparables: List[Dict[str, Any]], cd_tecnico_ajustado: Optional[int]) -> Dict[str, Any]:
    """Resume el riesgo de arrastre a partir de los comparables internos."""
    muy_alto = [c for c in comparables if c.get("similitud_total", 0) >= 0.90]
    alto = [c for c in comparables if c.get("similitud_total", 0) >= 0.85]
    medio = [c for c in comparables if c.get("similitud_total", 0) >= 0.80]

    if muy_alto:
        riesgo = "MUY_ALTO"
    elif alto:
        riesgo = "ALTO"
    elif medio:
        riesgo = "MEDIO"
    else:
        riesgo = "BAJO"

    diferenciales: List[int] = []
    if cd_tecnico_ajustado is not None:
        for c in medio:
            cd = c.get("cd_vigente")
            if cd is not None:
                diferenciales.append(int(cd_tecnico_ajustado) - int(cd))

    diferencial_medio = None
    if diferenciales:
        diferencial_medio = round(sum(diferenciales) / len(diferenciales), 3)

    return {
        "riesgo_arrastre": riesgo,
        "puestos_muy_similares": len(muy_alto),
        "puestos_similares_alto": len(alto),
        "puestos_similares_medio": len(medio),
        "diferencial_medio_cd": diferencial_medio,
        "observacion": "Existen puestos con similitud relevante que conviene revisar conjuntamente." if riesgo in {"MUY_ALTO", "ALTO", "MEDIO"} else "No se aprecian comparables internos con similitud suficiente para activar una alerta de arrastre.",
    }


def determinar_resultado_preliminar(
    validacion_rango: Dict[str, Any],
    diferencial_cd: Optional[int],
    efecto_arrastre: Optional[Dict[str, Any]] = None,
) -> str:
    """Etiqueta de resultado preliminar."""
    if validacion_rango.get("estado") != RANGO_OK:
        return "INCIDENCIA_NORMATIVA"
    if diferencial_cd is None:
        return "SIN_DATOS_SUFICIENTES"
    if efecto_arrastre and efecto_arrastre.get("riesgo_arrastre") in {"MUY_ALTO", "ALTO"} and diferencial_cd != 0:
        return "REVISION_AGRUPADA"
    if differential_is_zero(diferencial_cd):
        return "MANTENER"
    if diferencial_cd > 0:
        return "ANALIZAR_AL_ALZA"
    if diferencial_cd < 0:
        return "ANALIZAR_A_LA_BAJA"
    return "SIN_DATOS_SUFICIENTES"


def differential_is_zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except Exception:
        return False


def extraer_factores_dominantes(row: Any, top_n: int = 2) -> List[str]:
    """Identifica factores madre con mayor puntuación agregada en el puesto."""
    scores = calcular_scores_factores_madre(row)
    valid_scores = [(factor, score) for factor, score in scores.items() if score is not None]
    valid_scores.sort(key=lambda x: x[1], reverse=True)
    return [CD_FACTOR_PREFIXES.get(f, f) for f, _ in valid_scores[:top_n]]


def classify_het_cd(
    puesto_row: Any,
    patrones_vector: Any,
    puestos_vector: Any,
    rangos_cd: Any,
    pesos_modelo: Any = None,
    diccionario_columnas: Any = None,
    topk: int = 5,
    calcular_arrastre: bool = True,
) -> Dict[str, Any]:
    """
    Clasifica/evalúa un puesto tipo con el modelo HET-CD.

    Parameters
    ----------
    puesto_row : fila del puesto evaluado, procedente de puestos_vector.
    patrones_vector : matriz de patrones ampliados con F1-F49 + CD1_1-CD6_6.
    puestos_vector : matriz de puestos reales para comparables internos.
    rangos_cd : tabla con grupo_subgrupo, cd_min y cd_max.
    pesos_modelo : reservado para parametrización desde Excel. En esta v1 se usan
        pesos por defecto si no se transforma previamente a diccionarios.
    diccionario_columnas : reservado para validaciones/explicaciones.
    topk : número de patrones/comparables a devolver.
    calcular_arrastre : activa comparables internos.

    Returns
    -------
    Dict trazable con identificación, validación normativa, similitudes,
    top-k de patrones, resultado de CD, comparables, arrastre y explicación.
    """
    # En esta primera versión, los pesos se mantienen por defecto. La lectura fina
    # desde pesos_modelo se incorporará en la capa de carga de app.py o en una
    # función parse_pesos_modelo() posterior.
    block_weights = DEFAULT_BLOCK_WEIGHTS
    cd_factor_weights = DEFAULT_CD_FACTOR_WEIGHTS
    cd_subfactor_weights = DEFAULT_CD_SUBFACTOR_WEIGHTS

    dimension_check = validar_dimensiones_vectoriales(puestos_vector, patrones_vector)
    advertencias: List[str] = [
        "El resultado tiene carácter técnico auxiliar.",
        "La modificación del CD exige expediente de modificación de RPT, motivación, informes y aprobación por el órgano competente.",
    ]
    if not dimension_check["ok"]:
        advertencias.append("Existen columnas vectoriales ausentes en puestos_vector o patrones_vector; revise la configuración.")

    patrones_excluidos_calculo = contar_patrones_inactivos(patrones_vector)
    if patrones_excluidos_calculo:
        advertencias.append(
            f"Se han excluido {patrones_excluidos_calculo} patrones no activos o no validados del cálculo de similitud."
        )

    id_het = row_get(puesto_row, "id_het", "")
    denominacion = row_get(puesto_row, "denominacion_normalizada", row_get(puesto_row, "denominacion", ""))
    grupo = row_get(puesto_row, "grupo_subgrupo", "")
    cd_vigente = _to_int(row_get(puesto_row, "cd_vigente", None), None)

    validacion = validar_rango_cd(grupo, cd_vigente, rangos_cd)
    if validacion.get("estado") != RANGO_OK:
        advertencias.append(validacion.get("mensaje", "Incidencia normativa en la validación de rango."))

    top_k_patrones = obtener_top_k_patrones(
        puesto_row,
        patrones_vector,
        topk=topk,
        block_weights=block_weights,
        cd_factor_weights=cd_factor_weights,
        cd_subfactor_weights=cd_subfactor_weights,
    )

    recomendacion = recomendar_cd_desde_top_k(top_k_patrones)
    cd_recomendado = recomendacion.get("cd_recomendado")
    # Salvaguarda: el CD orientativo siempre debe coincidir con el CD del patrón K1.
    # Si un resultado heredado de sesión o una versión previa trae otro valor, se corrige aquí.
    if top_k_patrones and top_k_patrones[0].get("cd_referencia") is not None:
        cd_k1 = _to_int(top_k_patrones[0].get("cd_referencia"), None)
        if cd_k1 is not None and cd_recomendado != cd_k1:
            cd_recomendado = cd_k1
            recomendacion["cd_recomendado"] = cd_k1
            recomendacion["cd_top1"] = cd_k1
            recomendacion["metodo"] = "TOP1_PATRON_MAS_PROXIMO_CORREGIDO"
            advertencias.append("Se ha corregido internamente el CD orientativo para hacerlo coincidir con el patrón K1 más similar.")
    # Criterio v1.0.3:
    # El grupo/subgrupo NO interpreta ni corrige la recomendación técnica.
    # Solo sirve para validar rangos legales. La recomendación se toma del patrón K1
    # activo más similar. Si el CD recomendado excede el rango legal configurado,
    # se informa como advertencia, pero no se recorta automáticamente.
    cd_ajustado = cd_recomendado
    cd_recomendado_rango = ajustar_cd_a_rango(cd_recomendado, validacion)
    if cd_recomendado is not None and cd_recomendado_rango is not None and int(cd_recomendado) != int(cd_recomendado_rango):
        advertencias.append(
            f"El CD recomendado por K1 ({cd_recomendado}) queda fuera del rango legal configurado "
            f"para {grupo} ({validacion.get('cd_min')} - {validacion.get('cd_max')}). "
            "El motor no lo ajusta: lo informa para revisión jurídica/técnica."
        )
    diferencial_cd = None
    if cd_ajustado is not None and cd_vigente is not None:
        diferencial_cd = int(cd_ajustado) - int(cd_vigente)

    # Similitudes resumen: se toma el top-1 como referencia principal.
    if top_k_patrones:
        similitudes = {
            "funcional": top_k_patrones[0]["similitud_funcional"],
            "factores_cd": top_k_patrones[0]["similitud_cd"],
            "combinada": top_k_patrones[0]["similitud_total"],
        }
        sim_top1 = float(top_k_patrones[0].get("similitud_total", 0.0) or 0.0)
        if sim_top1 < 0.80:
            advertencias.append(
                "La similitud del patrón más próximo es inferior a 0,80; conviene revisar la vectorización funcional y los factores CD antes de usar el resultado como base de propuesta."
            )
    else:
        similitudes = {"funcional": 0.0, "factores_cd": 0.0, "combinada": 0.0}

    comparables: List[Dict[str, Any]] = []
    efecto_arrastre: Dict[str, Any] = {
        "riesgo_arrastre": "NO_CALCULADO",
        "puestos_muy_similares": 0,
        "puestos_similares_alto": 0,
        "puestos_similares_medio": 0,
        "diferencial_medio_cd": None,
        "observacion": "El cálculo de arrastre no se ha activado.",
    }
    if calcular_arrastre:
        comparables = calcular_comparables_internos(
            puesto_row,
            puestos_vector,
            topk=max(10, topk),
            block_weights=block_weights,
            cd_factor_weights=cd_factor_weights,
            cd_subfactor_weights=cd_subfactor_weights,
        )
        efecto_arrastre = calcular_efecto_arrastre(comparables, cd_ajustado)

    resultado_preliminar = determinar_resultado_preliminar(validacion, diferencial_cd, efecto_arrastre)

    factores_dominantes = extraer_factores_dominantes(puesto_row, top_n=3)
    motivos = []
    if top_k_patrones:
        motivos.append(
            f"El patrón de referencia más próximo es '{top_k_patrones[0].get('nombre_patron')}' "
            f"con similitud total {top_k_patrones[0].get('similitud_total')}."
        )
    if factores_dominantes:
        motivos.append("Factores dominantes detectados: " + ", ".join(factores_dominantes) + ".")
    if diferencial_cd is not None:
        motivos.append(f"El diferencial entre CD técnico ajustado y CD vigente es {diferencial_cd}.")

    vector_entrada = {col: row_get(puesto_row, col, None) for col in (F_COLUMNS + CD_COLUMNS)}

    return {
        "identificacion": {
            "id_het": id_het,
            "denominacion_normalizada": denominacion,
            "grupo_subgrupo": grupo,
            "cd_vigente": cd_vigente,
        },
        "vector_entrada": vector_entrada,
        "validacion_rango": validacion,
        "dimension_check": dimension_check,
        "similitudes": similitudes,
        "top_k_patrones": top_k_patrones,
        "resultado_cd": {
            "cd_continuo": recomendacion.get("cd_continuo"),
            "cd_tecnico_recomendado": cd_recomendado,
            "cd_tecnico_ajustado": cd_ajustado,
            "diferencial_cd": diferencial_cd,
            "resultado_preliminar": resultado_preliminar,
            "metodo_recomendacion": recomendacion.get("metodo"),
            "patron_top1": recomendacion.get("patron_top1"),
            "similitud_top1": recomendacion.get("similitud_top1"),
            "cd_recomendado_ajustado_a_rango_legal": cd_recomendado_rango,
            "nota_rango": (
                "El CD recomendado por K1 queda fuera del rango legal configurado; se informa sin ajustar automáticamente."
                if cd_recomendado is not None and cd_recomendado_rango is not None and int(cd_recomendado) != int(cd_recomendado_rango)
                else "El CD recomendado por K1 se mantiene dentro del rango legal configurado."
            ),
        },
        "comparables_internos": comparables,
        "efecto_arrastre": efecto_arrastre,
        "advertencias": advertencias,
        "explicacion": {
            "resumen": "Evaluación HET-CD calculada mediante similitud combinada de verbos funcionales y factores ampliados de Complemento de Destino.",
            "factores_dominantes": factores_dominantes,
            "motivos": motivos,
        },
    }


# ---------------------------------------------------------------------------
# Motor HET-CD v11: escala 0-3, impacto económico y análisis masivo RPT
# ---------------------------------------------------------------------------

# La versión v10 heredaba una normalización 1/2/3. En v11 la escala anclada es 0/1/2/3.
def normalizar_subfactor_cd(value: Any) -> Optional[float]:
    """
    Normaliza subfactores CD conforme a la escala metodológica v1.4+:
      0 -> 0.000
      1 -> 0.333
      2 -> 0.667
      3 -> 1.000
      N.A./vacío -> None, excluido del denominador.

    Nota: 0 no equivale a N.A. El 0 significa que el subfactor se ha valorado
    y no concurre; N.A. significa que el subfactor no es aplicable estructuralmente.
    """
    if _is_na_value(value):
        return None
    num = _to_float(value, None)
    if num is None:
        return None
    if num <= 0:
        return 0.0
    if num >= 3:
        return 1.0
    return max(0.0, min(1.0, num / 3.0))


def _lookup_cd_value(row: Any, keys: List[str], default: Any = None) -> Any:
    """Busca una columna admitiendo varias denominaciones habituales."""
    for k in keys:
        v = row_get(row, k, None)
        if v is not None and not _is_na_value(v):
            return v
    return default


def obtener_importe_cd_anual(importes_cd: Any, grupo_subgrupo: Any, nivel_cd: Any, anio: int = 2026) -> Optional[float]:
    """Devuelve el importe anual de CD para grupo/subgrupo y nivel desde importes_cd_2026."""
    if importes_cd is None:
        return None
    grupo = str(grupo_subgrupo or "").strip().upper()
    nivel = _to_int(nivel_cd, None)
    if not grupo or nivel is None:
        return None
    for row in iter_rows(importes_cd):
        row_grupo = str(_lookup_cd_value(row, ["grupo_subgrupo", "grupo", "subgrupo", "GR", "GR2"], "")).strip().upper()
        row_nivel = _to_int(_lookup_cd_value(row, ["nivel_cd", "nivel", "cd", "CD", "nivel_complemento_destino"], None), None)
        row_anio = _to_int(_lookup_cd_value(row, ["año", "anio", "ano", "year"], anio), anio)
        if row_grupo == grupo and row_nivel == nivel and row_anio == anio:
            val = _lookup_cd_value(row, ["cd_anual", "importe_anual", "anual", "año_cd", "ano_cd"], None)
            return _to_float(val, None)
    return None


def calcular_impacto_cd(
    grupo_subgrupo: Any,
    cd_vigente: Any,
    cd_destino: Any,
    importes_cd: Any,
    dotaciones: Any = 1,
    meses: Any = 12,
    anio: int = 2026,
) -> Dict[str, Any]:
    """Calcula el impacto económico del cambio de CD usando importes anuales oficiales."""
    cd_v = _to_int(cd_vigente, None)
    cd_d = _to_int(cd_destino, None)
    dots = _to_float(dotaciones, 1.0) or 0.0
    meses_num = _to_float(meses, 12.0) or 12.0
    grupo = str(grupo_subgrupo or "").strip().upper()

    if cd_v is None or cd_d is None or not grupo:
        return {"aplica": False, "motivo": "Grupo/subgrupo o niveles CD insuficientes para calcular impacto."}

    importe_origen = obtener_importe_cd_anual(importes_cd, grupo, cd_v, anio=anio)
    importe_destino = obtener_importe_cd_anual(importes_cd, grupo, cd_d, anio=anio)
    if importe_origen is None or importe_destino is None:
        return {
            "aplica": False,
            "motivo": "No se encontraron importes CD para el grupo/subgrupo y niveles indicados.",
            "grupo_subgrupo": grupo,
            "cd_origen": cd_v,
            "cd_destino": cd_d,
        }

    diferencial = float(importe_destino) - float(importe_origen)
    # Para la HET-CD se informa el diferencial unitario, pero el impacto presupuestario
    # agregado se activa únicamente cuando existe incremento de CD. Las bajadas no se
    # tratan como ahorro automático.
    diferencial_coste = max(diferencial, 0.0)
    impacto_anual = diferencial_coste * dots
    impacto_periodo = impacto_anual * (meses_num / 12.0)
    return {
        "aplica": diferencial > 0,
        "anio": anio,
        "grupo_subgrupo": grupo,
        "cd_origen": cd_v,
        "cd_destino": cd_d,
        "importe_anual_origen": round(float(importe_origen), 2),
        "importe_anual_destino": round(float(importe_destino), 2),
        "diferencial_anual_por_dotacion": round(diferencial, 2),
        "dotaciones": dots,
        "meses": meses_num,
        "impacto_anual_total": round(impacto_anual, 2),
        "impacto_periodo": round(impacto_periodo, 2),
        "motivo": "Impacto calculado exclusivamente sobre complemento de destino.",
    }


def generar_texto_recomendacion(result: Dict[str, Any]) -> Dict[str, str]:
    """Genera una recomendación administrativa legible a partir del resultado HET-CD."""
    ident = result.get("identificacion", {}) or {}
    res = result.get("resultado_cd", {}) or {}
    val = result.get("validacion_rango", {}) or {}
    arr = result.get("efecto_arrastre", {}) or {}
    dif = res.get("diferencial_cd")
    cd_vig = ident.get("cd_vigente")
    cd_k1 = res.get("cd_tecnico_recomendado")
    cd_final = res.get("cd_tecnico_ajustado")
    estado = val.get("estado")
    riesgo = arr.get("riesgo_arrastre", "NO_CALCULADO")

    try:
        dif_i = int(dif)
    except Exception:
        dif_i = None

    titulo = "Sin datos suficientes"
    texto = "No se dispone de datos suficientes para formular una recomendación técnica conclusiva."

    if estado != RANGO_OK:
        titulo = "Incidencia normativa previa"
        texto = (
            "Antes de valorar una modificación técnica del CD debe revisarse la incidencia detectada en el rango legal "
            "del grupo/subgrupo o en el CD vigente configurado."
        )
    elif dif_i == 0:
        titulo = "Mantener CD"
        texto = (
            f"El CD vigente ({cd_vig}) resulta coherente con el patrón técnico más próximo. "
            "Con los datos valorados, no se aprecia necesidad técnica individual de modificar el nivel de complemento de destino."
        )
    elif dif_i and dif_i > 0:
        titulo = "Analizar al alza"
        texto = (
            f"El puesto presenta mayor proximidad técnica con patrones de referencia de CD {cd_k1}. "
            f"La recomendación técnica queda en CD {cd_final}, frente al CD vigente {cd_vig}. Procede analizar la revisión al alza, "
            "sin perjuicio de la tramitación del expediente de modificación de RPT y de la revisión de puestos comparables."
        )
    elif dif_i and dif_i < 0:
        titulo = "Revisar coherencia a la baja"
        texto = (
            f"El puesto se aproxima a patrones de referencia de CD inferior al vigente. No se propone una reducción automática, "
            "sino revisar la coherencia interna del puesto, su configuración funcional y su comparación con puestos análogos."
        )

    if str(cd_k1) != str(cd_final):
        texto += (
            f" El patrón K1 apunta a CD {cd_k1}; el rango legal configurado para el grupo/subgrupo es "
            f"{val.get('cd_min')} - {val.get('cd_max')}. Este dato se informa como control de legalidad, sin recortar automáticamente la recomendación técnica."
        )

    if riesgo in {"MUY_ALTO", "ALTO"} and dif_i is not None and dif_i > 0:
        texto += " El riesgo de arrastre interno aconseja abordar la revisión de forma agrupada con los puestos de alta similitud."
    elif riesgo in {"MUY_ALTO", "ALTO"} and dif_i == 0:
        texto += " Los puestos de alta similitud se muestran como comparables internos de coherencia, no como arrastre activo, al no existir propuesta de modificación del CD."

    return {"titulo": titulo, "texto": texto}


def flatten_result_for_batch(result: Dict[str, Any], impacto: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convierte un resultado HET-CD en una fila plana para Excel/resumen."""
    ident = result.get("identificacion", {}) or {}
    res = result.get("resultado_cd", {}) or {}
    sims = result.get("similitudes", {}) or {}
    arr = result.get("efecto_arrastre", {}) or {}
    rec = result.get("recomendacion_actuacion", generar_texto_recomendacion(result))
    top = result.get("top_k_patrones", []) or []
    k1 = top[0] if top else {}
    impacto = impacto or result.get("impacto_economico_auto", {}) or {}
    return {
        "id_het": ident.get("id_het"),
        "denominacion_normalizada": ident.get("denominacion_normalizada"),
        "grupo_subgrupo": ident.get("grupo_subgrupo"),
        "cd_vigente": ident.get("cd_vigente"),
        "cd_k1_orientativo": res.get("cd_tecnico_recomendado"),
        "cd_final_ajustado": res.get("cd_tecnico_ajustado"),
        "diferencial_cd": res.get("diferencial_cd"),
        "resultado_preliminar": res.get("resultado_preliminar"),
        "recomendacion": rec.get("titulo"),
        "texto_recomendacion": rec.get("texto"),
        "patron_k1": k1.get("patron_id"),
        "nombre_patron_k1": k1.get("nombre_patron"),
        "similitud_funcional_k1": sims.get("funcional"),
        "similitud_factores_cd_k1": sims.get("factores_cd"),
        "similitud_total_k1": sims.get("combinada"),
        "riesgo_arrastre": arr.get("riesgo_arrastre"),
        "puestos_muy_similares": arr.get("puestos_muy_similares"),
        "puestos_similares_alto": arr.get("puestos_similares_alto"),
        "puestos_similares_medio": arr.get("puestos_similares_medio"),
        "impacto_anual_total": impacto.get("impacto_anual_total"),
        "impacto_periodo": impacto.get("impacto_periodo"),
        "diferencial_anual_por_dotacion": impacto.get("diferencial_anual_por_dotacion"),
        "dotaciones": impacto.get("dotaciones"),
        "estado_rango": (result.get("validacion_rango", {}) or {}).get("estado"),
    }


def analizar_rpt_completa(
    puestos_vector: Any,
    patrones_vector: Any,
    rangos_cd: Any,
    importes_cd_2026: Any = None,
    pesos_modelo: Any = None,
    topk: int = 5,
    meses: int = 12,
    anio: int = 2026,
) -> Dict[str, Any]:
    """Ejecuta evaluación HET-CD sobre todos los puestos de puestos_vector."""
    resultados: List[Dict[str, Any]] = []
    resumen_rows: List[Dict[str, Any]] = []
    rows = iter_rows(puestos_vector)
    for row in rows:
        result = classify_het_cd(
            puesto_row=row,
            patrones_vector=patrones_vector,
            puestos_vector=puestos_vector,
            rangos_cd=rangos_cd,
            pesos_modelo=pesos_modelo,
            topk=topk,
            calcular_arrastre=True,
        )
        ident = result.get("identificacion", {}) or {}
        res = result.get("resultado_cd", {}) or {}
        dotaciones = _lookup_cd_value(row, ["dotaciones", "DOTACIONES", "num_dotaciones", "n_dotaciones"], 1)
        impacto = calcular_impacto_cd(
            ident.get("grupo_subgrupo"),
            ident.get("cd_vigente"),
            res.get("cd_tecnico_ajustado"),
            importes_cd_2026,
            dotaciones=dotaciones,
            meses=meses,
            anio=anio,
        ) if importes_cd_2026 is not None else {"aplica": False, "motivo": "No se ha cargado hoja importes_cd_2026."}
        result["impacto_economico_auto"] = impacto
        result["recomendacion_actuacion"] = generar_texto_recomendacion(result)
        resultados.append(result)
        resumen_rows.append(flatten_result_for_batch(result, impacto))

    # Resúmenes agregados simples.
    total_impacto_anual = round(sum(_to_float(r.get("impacto_anual_total"), 0.0) or 0.0 for r in resumen_rows), 2)
    total_impacto_periodo = round(sum(_to_float(r.get("impacto_periodo"), 0.0) or 0.0 for r in resumen_rows), 2)
    conteo_resultados: Dict[str, int] = {}
    conteo_grupos: Dict[str, int] = {}
    for r in resumen_rows:
        conteo_resultados[str(r.get("resultado_preliminar"))] = conteo_resultados.get(str(r.get("resultado_preliminar")), 0) + 1
        conteo_grupos[str(r.get("grupo_subgrupo"))] = conteo_grupos.get(str(r.get("grupo_subgrupo")), 0) + 1

    return {
        "resultados": resultados,
        "resumen_rows": resumen_rows,
        "agregado": {
            "puestos_analizados": len(resumen_rows),
            "conteo_resultados": conteo_resultados,
            "conteo_grupos": conteo_grupos,
            "impacto_anual_total": total_impacto_anual,
            "impacto_periodo_total": total_impacto_periodo,
            "anio_importes": anio,
            "meses": meses,
        },
    }
