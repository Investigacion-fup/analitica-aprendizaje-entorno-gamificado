"""Tarea 2.1 - Clasificador de riesgo (diagnóstico) con Random Forest.

Entrena un único modelo sobre "instantáneas" del curso (días 30, 60, 90, 120, 150),
de modo que el mismo modelo pueda evaluar a un estudiante en cualquier semana
(monitoreo continuo). Validación temporal: se entrena con 2013B, 2013J y 2014B
y se prueba con la presentación más reciente (2014J), nunca vista.

Genera:
  data/processed/instantaneas.pkl   features + predicción + perfil para todos los cortes
  models/modelo_riesgo_rf.joblib
  outputs/metricas_modelo.json, outputs/importancia_variables.csv
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_score

from comun import (COLUMNAS_MODELO, CORTES, DIR_MODELOS, DIR_PROC, DIR_SALIDAS,
                   PRESENTACION_PRUEBA, SEMILLA, cargar_procesados, construir_features,
                   diagnosticar_perfil, nivel_riesgo)

t0 = time.time()
estudiantes, clics_dia, entregas = cargar_procesados()

inst = pd.concat([construir_features(estudiantes, clics_dia, entregas, c) for c in CORTES],
                 ignore_index=True)
print(f"Instantáneas: {len(inst):,} filas ({time.time() - t0:.0f}s)")

prueba = inst["code_presentation"] == PRESENTACION_PRUEBA
X, y = inst[COLUMNAS_MODELO], inst["riesgo"]

modelo = RandomForestClassifier(
    n_estimators=300, max_depth=18, min_samples_leaf=5, max_features="sqrt",
    class_weight="balanced", n_jobs=-1, random_state=SEMILLA)

# Validación cruzada agrupada por curso (evita que un mismo curso esté en ambos lados)
grupos = inst.loc[~prueba, "code_module"] + inst.loc[~prueba, "code_presentation"]
cv_auc = cross_val_score(modelo, X[~prueba], y[~prueba], groups=grupos,
                         cv=GroupKFold(n_splits=5), scoring="roc_auc", n_jobs=1)
print(f"AUC validación cruzada (entrenamiento): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

t_fit = time.time()
modelo.fit(X[~prueba], y[~prueba])
t_fit = time.time() - t_fit

t_pred = time.time()
inst["prob_riesgo"] = modelo.predict_proba(X)[:, 1]
t_pred = (time.time() - t_pred) / len(inst) * 1000
inst["pred_riesgo"] = (inst["prob_riesgo"] >= 0.5).astype(int)
inst["nivel_riesgo"] = nivel_riesgo(inst["prob_riesgo"])
inst["perfil"] = diagnosticar_perfil(inst)
inst["conjunto"] = np.where(prueba, "prueba", "entrenamiento")


def metricas(d):
    return {
        "n": int(len(d)),
        "tasa_riesgo_real": round(float(d["riesgo"].mean()), 4),
        "accuracy": round(accuracy_score(d["riesgo"], d["pred_riesgo"]), 4),
        "precision": round(precision_score(d["riesgo"], d["pred_riesgo"]), 4),
        "recall": round(recall_score(d["riesgo"], d["pred_riesgo"]), 4),
        "f1": round(f1_score(d["riesgo"], d["pred_riesgo"]), 4),
        "auc": round(roc_auc_score(d["riesgo"], d["prob_riesgo"]), 4),
    }


test = inst[prueba]
resultado = {
    "modelo": "RandomForestClassifier",
    "hiperparametros": {k: v for k, v in modelo.get_params().items()
                        if k in ("n_estimators", "max_depth", "min_samples_leaf",
                                 "max_features", "class_weight")},
    "n_variables": len(COLUMNAS_MODELO),
    "entrenamiento": {"filas": int((~prueba).sum()), "presentaciones": sorted(
        inst.loc[~prueba, "code_presentation"].unique().tolist())},
    "prueba": {"filas": int(prueba.sum()), "presentacion": PRESENTACION_PRUEBA},
    "cv_auc_media": round(float(cv_auc.mean()), 4),
    "cv_auc_desv": round(float(cv_auc.std()), 4),
    "prueba_global": metricas(test),
    "prueba_por_corte": {int(c): metricas(g) for c, g in test.groupby("dia_corte")},
    "eficiencia": {
        "segundos_entrenamiento": round(t_fit, 1),
        "ms_prediccion_por_estudiante": round(t_pred, 4),
    },
}
print(json.dumps(resultado["prueba_por_corte"], indent=1))

importancia = (pd.DataFrame({"variable": COLUMNAS_MODELO, "importancia": modelo.feature_importances_})
                 .sort_values("importancia", ascending=False))
importancia.to_csv(DIR_SALIDAS / "importancia_variables.csv", index=False)
(DIR_SALIDAS / "metricas_modelo.json").write_text(
    json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
joblib.dump({"modelo": modelo, "columnas": COLUMNAS_MODELO}, DIR_MODELOS / "modelo_riesgo_rf.joblib",
            compress=3)
inst.to_pickle(DIR_PROC / "instantaneas.pkl")
print(importancia.head(10).to_string(index=False))
print(f"Listo en {time.time() - t0:.0f}s")
