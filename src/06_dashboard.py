"""Tarea 3.2 - Datos del Dashboard de monitoreo docente.

Produce dos salidas a partir del monitoreo semanal + ajuste de dificultad:
  dashboard/datos_bi/*.csv          modelo estrella listo para Power BI / Looker Studio / Tableau
  dashboard/dashboard_docente.html  tablero interactivo autónomo (abre en cualquier navegador)
"""
import json

import pandas as pd

from comun import CLAVE, DIR_CONFIG, DIR_DASHBOARD, DIR_PROC

DIR_BI = DIR_DASHBOARD / "datos_bi"
DIR_BI.mkdir(exist_ok=True)

mon = pd.read_pickle(DIR_PROC / "monitoreo_semanal.pkl")
tray = pd.read_pickle(DIR_PROC / "trayectorias.pkl")
est = pd.read_pickle(DIR_PROC / "estudiantes.pkl")
catalogo = json.loads((DIR_CONFIG / "misiones_io.json").read_text(encoding="utf-8"))

cols_fabrica = ["velocidad_maquinas", "tasa_llegada_pedidos", "prob_falla_maquina",
                "tiempo_limite_turno", "variabilidad_demanda", "nivel_pistas"]
hecho = mon.merge(
    tray[CLAVE + ["semana", "desempeno", "nivel"] + cols_fabrica].rename(columns={"nivel": "nivel_dificultad"}),
    on=CLAVE + ["semana"], how="left")
hecho["ruta"] = hecho["ruta"].map(" > ".join)
hecho["necesita_ayuda"] = (hecho["prob_riesgo"] >= 0.5).astype(int)

# --- Modelo estrella para herramientas BI --------------------------------------
cols_hecho = CLAVE + ["semana", "dia_corte", "prob_riesgo", "nivel_riesgo", "necesita_ayuda", "perfil",
                      "mision_principal", "ruta", "tema_debil", "tema_actual", "formato_deficit",
                      "desempeno", "nivel_dificultad", *cols_fabrica, "clics_total", "dias_sin_actividad",
                      "puntaje_medio", "tasa_entrega", "motivo"]
hecho[cols_hecho].to_csv(DIR_BI / "hecho_monitoreo_semanal.csv", index=False, encoding="utf-8-sig")
est_cohorte = est[est["code_presentation"].isin(hecho["code_presentation"].unique())]
est_cohorte[CLAVE + ["gender", "region", "highest_education", "imd_band", "age_band",
                     "num_of_prev_attempts", "studied_credits", "disability", "final_result"]].to_csv(
    DIR_BI / "dim_estudiante.csv", index=False, encoding="utf-8-sig")
misiones = pd.DataFrame(catalogo["misiones"]).drop(columns="perfiles")
especiales = pd.DataFrame([{"id": v["id"], "titulo": v["titulo"], "descripcion": v["descripcion"],
                            "tema": "Transversal", "nivel": 0, "formato": "especial", "duracion_min": 5}
                           for v in catalogo["misiones_especiales"].values()])
pd.concat([misiones, especiales]).to_csv(DIR_BI / "dim_mision.csv", index=False, encoding="utf-8-sig")
print(f"BI: {len(hecho):,} filas de hechos semanales")

# --- Datos compactos (columnares) para el tablero HTML --------------------------
cursos = sorted(hecho["code_module"].unique())
perfiles = ["Desconexión", "Frustración", "Procrastinación", "Bajo rendimiento", "Riesgo general",
            "En progreso", "Avanzado"]
temas = catalogo["temas"]
rutas_unicas = sorted(hecho["ruta"].unique())
ids = hecho[CLAVE].drop_duplicates().reset_index(drop=True)
ids["k"] = range(len(ids))
hecho = hecho.merge(ids, on=CLAVE)
hecho = hecho.sort_values(["k", "semana"])

datos = {
    "presentacion": hecho["code_presentation"].iat[0],
    "semanas": [int(hecho["semana"].min()), int(hecho["semana"].max())],
    "cursos": cursos, "perfiles": perfiles, "temas": temas, "rutas": rutas_unicas,
    "formatos": {"oucontent": "lectura interactiva", "quiz": "práctica evaluada",
                 "forumng": "trabajo colaborativo", "resource": "recursos de apoyo"},
    "misiones": {m["id"]: m["titulo"] for m in catalogo["misiones"]}
                | {v["id"]: v["titulo"] for v in catalogo["misiones_especiales"].values()},
    "est_id": ids["id_student"].tolist(),
    "est_curso": ids["code_module"].map(cursos.index).tolist(),
    "fil": {
        "k": hecho["k"].tolist(),
        "w": hecho["semana"].tolist(),
        "p": (hecho["prob_riesgo"] * 100).round().astype(int).tolist(),
        "pf": hecho["perfil"].map(perfiles.index).tolist(),
        "r": hecho["ruta"].map({r: i for i, r in enumerate(rutas_unicas)}).tolist(),
        "td": hecho["tema_debil"].map(temas.index).tolist(),
        "fd": hecho["formato_deficit"].tolist(),
        "nd": (hecho["nivel_dificultad"].fillna(3) * 10).round().astype(int).tolist(),
        "d": (hecho["desempeno"].fillna(0) * 100).round().astype(int).tolist(),
        "vm": hecho["velocidad_maquinas"].round(2).tolist(),
        "tp": hecho["tasa_llegada_pedidos"].round(1).tolist(),
        "pi": hecho["nivel_pistas"].fillna(0).astype(int).tolist(),
        "ds": hecho["dias_sin_actividad"].astype(int).tolist(),
        "nt": hecho["puntaje_medio"].round().astype(int).tolist(),
    },
}
# formatos como índices para reducir tamaño
fmts = list(datos["formatos"])
datos["fil"]["fd"] = [fmts.index(f) for f in datos["fil"]["fd"]]
datos["formatos"] = [datos["formatos"][f] for f in fmts]

plantilla = (DIR_DASHBOARD / "plantilla_dashboard.html").read_text(encoding="utf-8")
html = plantilla.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
(DIR_DASHBOARD / "dashboard_docente.html").write_text(html, encoding="utf-8")
print(f"Dashboard HTML: {len(html) / 1e6:.1f} MB, {len(ids):,} estudiantes, {len(rutas_unicas)} rutas distintas")
