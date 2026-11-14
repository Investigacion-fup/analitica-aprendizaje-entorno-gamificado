"""Monitoreo semanal de la cohorte de prueba (simulación de operación en tiempo real).

Cada semana del curso el sistema "despierta", recalcula las variables con la
información disponible hasta ese día, estima el riesgo con el modelo ya
entrenado, diagnostica el perfil y reasigna la ruta de misiones. Es la fuente
de datos del dashboard docente y del ajuste de dificultad.

Genera data/processed/monitoreo_semanal.pkl
"""
import importlib
import time

import joblib
import pandas as pd

from comun import (DIR_MODELOS, DIR_PROC, PRESENTACION_PRUEBA, cargar_procesados,
                   construir_features, diagnosticar_perfil, nivel_riesgo)

motor = importlib.import_module("03_motor_recomendacion")
SEMANA_INICIO, SEMANA_FIN = 4, 21  # el primer diagnóstico se emite al cumplir 4 semanas

t0 = time.time()
estudiantes, clics_dia, entregas = cargar_procesados()
estudiantes = estudiantes[estudiantes["code_presentation"] == PRESENTACION_PRUEBA]
art = joblib.load(DIR_MODELOS / "modelo_riesgo_rf.joblib")
pares = motor.referencia_pares(pd.read_pickle(DIR_PROC / "instantaneas.pkl"))

semanas = []
for w in range(SEMANA_INICIO, SEMANA_FIN + 1):
    t = time.time()
    df = construir_features(estudiantes, clics_dia, entregas, corte=7 * w + 6)
    df["prob_riesgo"] = art["modelo"].predict_proba(df[art["columnas"]])[:, 1]
    df["nivel_riesgo"] = nivel_riesgo(df["prob_riesgo"])
    df["perfil"] = diagnosticar_perfil(df)
    df = df.join(motor.recomendar(df, entregas, pares))
    df["semana"] = w
    semanas.append(df)
    print(f"  semana {w:2d}: {len(df):,} estudiantes activos, "
          f"{(df['prob_riesgo'] >= 0.5).mean():.0%} en riesgo ({time.time() - t:.1f}s)")

monitoreo = pd.concat(semanas, ignore_index=True)
monitoreo.to_pickle(DIR_PROC / "monitoreo_semanal.pkl")
print(f"Monitoreo de {SEMANA_FIN - SEMANA_INICIO + 1} semanas en {time.time() - t0:.0f}s")
