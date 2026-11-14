"""Tarea 1.1 - Selección y limpieza del dataset masivo (OULAD).

Lee las 7 tablas de OULAD (~10,6 millones de registros de clics), las limpia,
las integra y genera:
  data/processed/estudiantes.pkl     perfil + matrícula + resultado por estudiante-curso
  data/processed/clics_diarios.pkl   clics agregados por estudiante, día y tipo de recurso
  data/processed/entregas.pkl        evaluaciones esperadas vs. entregadas
  data/processed/dataset_estructurado.csv  dataset analítico (features al día de corte principal)
  outputs/reporte_calidad_datos.json
"""
import json
import time

import pandas as pd

from comun import (CLAVE, CORTE_PRINCIPAL, DIR_OULAD, DIR_PROC, DIR_SALIDAS, ORD_EDAD,
                   ORD_EDUCACION, TIPOS_ACTIVIDAD, construir_features, imd_a_numero)

t0 = time.time()
calidad = {}


def leer(nombre, **kw):
    df = pd.read_csv(DIR_OULAD / nombre, na_values=["?"], **kw)
    calidad[nombre] = {"filas": int(len(df)), "nulos": {k: int(v) for k, v in df.isna().sum().items() if v}}
    return df


# --- 1. Lectura ---------------------------------------------------------------
cursos = leer("courses.csv")
evaluaciones = leer("assessments.csv")
vle = leer("vle.csv")
info = leer("studentInfo.csv")
registro = leer("studentRegistration.csv")
eval_est = leer("studentAssessment.csv")
print("Leyendo studentVle.csv (~450 MB)...")
clics = leer("studentVle.csv", dtype={"code_module": "category", "code_presentation": "category",
                                      "id_student": "int32", "id_site": "int32",
                                      "date": "int16", "sum_click": "int32"})
print(f"  {len(clics):,} registros de interacción leídos en {time.time() - t0:.0f}s")

# --- 2. Limpieza de estudiantes ------------------------------------------------
dup = int(info.duplicated(CLAVE).sum())
info = info.drop_duplicates(CLAVE)
est = (info.merge(registro, on=CLAVE, how="left")
           .merge(cursos, on=["code_module", "code_presentation"], how="left"))
# Fechas de registro faltantes -> mediana del curso
est["date_registration"] = est["date_registration"].fillna(
    est.groupby(["code_module", "code_presentation"])["date_registration"].transform("median"))
est["educacion"] = est["highest_education"].map(ORD_EDUCACION)
est["edad"] = est["age_band"].map(ORD_EDAD)
est["imd"] = imd_a_numero(est["imd_band"])
est["imd"] = est["imd"].fillna(est.groupby("region")["imd"].transform("median"))
est["discapacidad"] = (est["disability"] == "Y").astype(int)
est["genero_m"] = (est["gender"] == "M").astype(int)
est["stem"] = est["code_module"].isin(["CCC", "DDD", "EEE", "FFF"]).astype(int)
est["riesgo"] = est["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
calidad["limpieza_estudiantes"] = {
    "duplicados_eliminados": dup,
    "registros_finales": int(len(est)),
    "imd_imputados_por_region": int(info["imd_band"].isna().sum()),
    "tasa_riesgo_global": round(float(est["riesgo"].mean()), 4),
}

# --- 3. Limpieza y agregación de clics (Big Data -> tabla manejable) --------------
antes = len(clics)
clics = clics[clics["sum_click"] > 0]
tipos = vle.set_index("id_site")["activity_type"]
clics["tipo"] = clics["id_site"].map(tipos)
clics["tipo"] = clics["tipo"].where(clics["tipo"].isin(TIPOS_ACTIVIDAD), "otros").astype("category")
clics_dia = (clics.groupby(CLAVE + ["date", "tipo"], observed=True)["sum_click"].sum()
                  .reset_index())
clics_dia["id_student"] = clics_dia["id_student"].astype("int32")
clics_dia["sum_click"] = clics_dia["sum_click"].astype("int32")
calidad["agregacion_clics"] = {
    "registros_originales": int(antes),
    "registros_agregados": int(len(clics_dia)),
    "reduccion_pct": round(100 * (1 - len(clics_dia) / antes), 1),
    "clics_totales": int(clics_dia["sum_click"].sum()),
    "estudiantes_con_actividad": int(clics_dia.groupby(CLAVE, observed=True).ngroups),
}
del clics

# --- 4. Entregas: evaluaciones esperadas (no examen) vs. reales ---------------
ev = evaluaciones[evaluaciones["assessment_type"] != "Exam"].dropna(subset=["date"])
esperadas = est[CLAVE].merge(ev, on=["code_module", "code_presentation"])
entregas = esperadas.merge(eval_est, on=["id_assessment", "id_student"], how="left")
entregas = entregas.rename(columns={"date": "fecha_limite"})
entregas["entregada"] = entregas["date_submitted"].notna()
entregas["score"] = entregas["score"].clip(0, 100)
entregas.loc[entregas["entregada"] & entregas["score"].isna(), "score"] = 0
entregas["retraso"] = (entregas["date_submitted"] - entregas["fecha_limite"]).clip(lower=0)
entregas["tardia"] = (entregas["retraso"] > 0).astype(int)
entregas["reprobada"] = (entregas["score"] < 40).astype(int)
entregas = entregas.sort_values(CLAVE + ["fecha_limite"]).reset_index(drop=True)
calidad["entregas"] = {
    "evaluaciones_esperadas": int(len(entregas)),
    "entregadas": int(entregas["entregada"].sum()),
    "tardias": int(entregas["tardia"].sum()),
}

# --- 5. Guardado -------------------------------------------------------------
est.to_pickle(DIR_PROC / "estudiantes.pkl")
clics_dia.to_pickle(DIR_PROC / "clics_diarios.pkl")
entregas.to_pickle(DIR_PROC / "entregas.pkl")

dataset = construir_features(est, clics_dia, entregas, CORTE_PRINCIPAL)
dataset.to_csv(DIR_PROC / "dataset_estructurado.csv", index=False)
calidad["dataset_estructurado"] = {
    "dia_corte": CORTE_PRINCIPAL,
    "filas": int(len(dataset)),
    "columnas": int(dataset.shape[1]),
    "excluidos_por_baja_previa": int(len(est) - len(dataset)),
}
calidad["tiempo_segundos"] = round(time.time() - t0, 1)
(DIR_SALIDAS / "reporte_calidad_datos.json").write_text(
    json.dumps(calidad, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in calidad.items() if not k.endswith(".csv")}, indent=2,
                 ensure_ascii=False))
