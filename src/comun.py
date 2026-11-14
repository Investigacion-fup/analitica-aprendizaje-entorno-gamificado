"""Utilidades compartidas del proyecto Hoja 6 (Big Data educativo + optimización).

Contiene rutas, constantes y la construcción de variables (features) a partir
de OULAD para cualquier "día de corte" del curso. El día de corte simula el
momento en que el sistema observa al estudiante: solo se usan datos con
fecha <= corte, evitando fuga de información del futuro.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_OULAD = RAIZ / "oulad"
DIR_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_CONFIG = RAIZ / "config"
DIR_SALIDAS = RAIZ / "outputs"
DIR_FIGURAS = DIR_SALIDAS / "figuras"
DIR_DASHBOARD = RAIZ / "dashboard"

for _d in (DIR_PROC, DIR_MODELOS, DIR_SALIDAS, DIR_FIGURAS, DIR_DASHBOARD):
    _d.mkdir(parents=True, exist_ok=True)

# Días de corte (instantáneas) en los que el sistema evalúa a cada estudiante.
CORTES = [30, 60, 90, 120, 150]
CORTE_PRINCIPAL = 90
# Presentación reservada como conjunto de prueba (validación temporal).
PRESENTACION_PRUEBA = "2014J"
SEMILLA = 42

TIPOS_ACTIVIDAD = ["oucontent", "quiz", "forumng", "resource", "subpage", "url", "homepage"]

ORD_EDUCACION = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
ORD_EDAD = {"0-35": 0, "35-55": 1, "55<=": 2}
CLAVE = ["code_module", "code_presentation", "id_student"]


def imd_a_numero(s: pd.Series) -> pd.Series:
    """'20-30%' -> 25 ; '?' -> NaN (índice de privación socioeconómica)."""
    s = s.astype("string").str.replace("%", "", regex=False)
    lim = s.str.split("-", expand=True)
    return (pd.to_numeric(lim[0], errors="coerce") + pd.to_numeric(lim[1], errors="coerce")) / 2


def cargar_procesados():
    """Carga las tablas limpias generadas por 01_preparar_datos.py."""
    estudiantes = pd.read_pickle(DIR_PROC / "estudiantes.pkl")
    clics_dia = pd.read_pickle(DIR_PROC / "clics_diarios.pkl")
    entregas = pd.read_pickle(DIR_PROC / "entregas.pkl")
    return estudiantes, clics_dia, entregas


def _pendiente(matriz: np.ndarray) -> np.ndarray:
    """Pendiente de una regresión lineal fila a fila (tendencia semanal)."""
    n = matriz.shape[1]
    x = np.arange(n) - (n - 1) / 2
    return (matriz * x).sum(axis=1) / (x ** 2).sum()


def _racha_max_ceros(matriz: np.ndarray) -> np.ndarray:
    racha = np.zeros(matriz.shape[0], dtype=int)
    actual = np.zeros(matriz.shape[0], dtype=int)
    for j in range(matriz.shape[1]):
        actual = np.where(matriz[:, j] == 0, actual + 1, 0)
        racha = np.maximum(racha, actual)
    return racha


def construir_features(estudiantes: pd.DataFrame, clics_dia: pd.DataFrame,
                       entregas: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Variables de un estudiante observadas hasta el día `corte` del curso.

    Se excluye a quienes ya se dieron de baja antes del corte (ya no son
    "rescatables" por el sistema).
    """
    base = estudiantes[
        estudiantes["date_unregistration"].isna() | (estudiantes["date_unregistration"] > corte)
    ].copy()
    base = base[base["module_presentation_length"] > corte]

    # --- Interacción con la plataforma (clics y tiempo) ---------------------
    c = clics_dia[clics_dia["date"] <= corte]
    tot = c.groupby(CLAVE, observed=True).agg(
        clics_total=("sum_click", "sum"),
        dias_activos=("date", "nunique"),
        ultimo_dia_activo=("date", "max"),
        primer_dia_activo=("date", "min"),
    )
    tot["clics_pre_inicio"] = c[c["date"] < 0].groupby(CLAVE, observed=True)["sum_click"].sum()

    por_tipo = c.pivot_table(index=CLAVE, columns="tipo", values="sum_click",
                             aggfunc="sum", fill_value=0, observed=True)
    por_tipo = por_tipo.reindex(columns=TIPOS_ACTIVIDAD + ["otros"], fill_value=0)
    por_tipo.columns = [f"clics_{t}" for t in por_tipo.columns]

    c_sem = c[c["date"] >= 0].assign(semana=lambda d: d["date"] // 7)
    n_sem = corte // 7 + 1
    semanal = c_sem.pivot_table(index=CLAVE, columns="semana", values="sum_click",
                                aggfunc="sum", fill_value=0, observed=True)
    semanal = semanal.reindex(columns=range(n_sem), fill_value=0)

    df = base.set_index(CLAVE)
    df = df.join(tot).join(por_tipo)
    sem = semanal.reindex(df.index, fill_value=0).to_numpy(dtype=float)
    log_sem = np.log1p(sem)
    df["tendencia_semanal"] = _pendiente(log_sem)
    df["semanas_inactivas"] = (sem == 0).sum(axis=1)
    df["racha_inactiva_max"] = _racha_max_ceros(sem)
    k = min(4, n_sem // 2)
    ini, fin = sem[:, :k].sum(axis=1), sem[:, -k:].sum(axis=1)
    df["caida_actividad"] = (fin + 1) / (ini + 1)  # <1 => la actividad decae
    df["cv_semanal"] = sem.std(axis=1) / (sem.mean(axis=1) + 1)

    cols_clic = [col for col in df.columns if col.startswith("clics_")] + ["dias_activos"]
    df[cols_clic] = df[cols_clic].fillna(0)
    df["dias_sin_actividad"] = corte - df["ultimo_dia_activo"].fillna(-30)
    df["clics_por_dia_activo"] = df["clics_total"] / df["dias_activos"].clip(lower=1)
    df["prop_dias_activos"] = df["dias_activos"] / (corte + 1)

    # Normalización relativa al curso (cada módulo tiene distinto volumen de material)
    grp = df.groupby(level=["code_module", "code_presentation"], observed=True)
    df["clics_rel_curso"] = df["clics_total"] / grp["clics_total"].transform("median").clip(lower=1)
    df["dias_rel_curso"] = df["dias_activos"] / grp["dias_activos"].transform("median").clip(lower=1)

    # --- Evaluaciones (rendimiento y puntualidad) --------------------------
    e = entregas[(entregas["fecha_limite"] <= corte)]
    e = e.merge(base[CLAVE], on=CLAVE)
    e_ent = e[e["entregada"] & (e["date_submitted"] <= corte)]
    ev = e.groupby(CLAVE).agg(n_eval_esperadas=("id_assessment", "count"))
    ev = ev.join(e_ent.groupby(CLAVE).agg(
        n_eval_entregadas=("id_assessment", "count"),
        puntaje_medio=("score", "mean"),
        puntaje_min=("score", "min"),
        puntaje_ultimo=("score", "last"),
        n_tardias=("tardia", "sum"),
        retraso_medio=("retraso", "mean"),
        n_reprobadas=("reprobada", "sum"),
    ))
    pond = e_ent.assign(p=e_ent["score"] * e_ent["weight"]).groupby(CLAVE)[["p", "weight"]].sum()
    ev["puntaje_ponderado"] = pond["p"] / pond["weight"].replace(0, np.nan)
    df = df.join(ev)
    df["n_eval_esperadas"] = df["n_eval_esperadas"].fillna(0)
    df["n_eval_entregadas"] = df["n_eval_entregadas"].fillna(0)
    df["tasa_entrega"] = np.where(df["n_eval_esperadas"] > 0,
                                  df["n_eval_entregadas"] / df["n_eval_esperadas"].clip(lower=1), 1.0)
    df["n_omitidas"] = df["n_eval_esperadas"] - df["n_eval_entregadas"]
    for col in ("n_tardias", "n_reprobadas", "retraso_medio"):
        df[col] = df[col].fillna(0)
    df["sin_evaluaciones"] = df["puntaje_medio"].isna().astype(int)
    # Sin entregas todavía: se imputa con un valor neutro y se marca con el indicador anterior
    for col in ("puntaje_medio", "puntaje_min", "puntaje_ultimo", "puntaje_ponderado"):
        df[col] = df[col].fillna(-1)

    df["dia_corte"] = corte
    return df.reset_index()


COLUMNAS_MODELO = [
    "dia_corte",
    # interacción
    "clics_total", "dias_activos", "clics_por_dia_activo", "prop_dias_activos",
    "clics_rel_curso", "dias_rel_curso", "clics_pre_inicio", "dias_sin_actividad",
    "tendencia_semanal", "semanas_inactivas", "racha_inactiva_max", "caida_actividad", "cv_semanal",
    *[f"clics_{t}" for t in TIPOS_ACTIVIDAD + ["otros"]],
    # evaluaciones
    "n_eval_esperadas", "n_eval_entregadas", "tasa_entrega", "n_omitidas", "puntaje_medio",
    "puntaje_min", "puntaje_ultimo", "puntaje_ponderado", "n_tardias", "retraso_medio",
    "n_reprobadas", "sin_evaluaciones",
    # perfil
    "educacion", "edad", "imd", "num_of_prev_attempts", "studied_credits", "discapacidad",
    "genero_m", "date_registration", "stem",
]


def diagnosticar_perfil(df: pd.DataFrame) -> pd.Series:
    """Asigna un perfil pedagógico interpretable a partir de clics, tiempo y notas.

    Se aplica después del modelo: la probabilidad de riesgo dice *cuánto* riesgo
    hay; el perfil dice *por qué*, que es lo que necesita el motor de rutas.
    """
    p = df["prob_riesgo"]
    nota = df["puntaje_medio"].where(df["sin_evaluaciones"] == 0)
    desconexion = (df["dias_sin_actividad"] >= 14) | (df["dias_rel_curso"] < 0.35)
    frustracion = (
        (df["clics_rel_curso"] >= 0.8) & (nota < 60)
    ) | ((df["caida_actividad"] < 0.5) & (df["n_reprobadas"] > 0))
    procrastinacion = (df["n_omitidas"] > 0) | (df["n_tardias"] >= 1) | (df["cv_semanal"] > 1.6)
    bajo_rend = nota < 60
    avanzado = (nota >= 80) & (df["tasa_entrega"] >= 1) & (p < 0.3)

    # Las máscaras se aplican de menor a mayor prioridad: la última que se
    # cumple define el perfil (desconexión es la causa más urgente).
    perfil = pd.Series("En progreso", index=df.index)
    perfil[avanzado.fillna(False)] = "Avanzado"
    riesgo = p >= 0.5
    perfil[riesgo] = "Riesgo general"
    perfil[riesgo & bajo_rend.fillna(False)] = "Bajo rendimiento"
    perfil[riesgo & procrastinacion] = "Procrastinación"
    perfil[riesgo & frustracion.fillna(False)] = "Frustración"
    perfil[riesgo & desconexion] = "Desconexión"
    return perfil


def nivel_riesgo(prob: pd.Series) -> pd.Series:
    return pd.cut(prob, [-0.01, 0.3, 0.5, 0.7, 1.01],
                  labels=["Bajo", "Moderado", "Alto", "Crítico"]).astype(str)
