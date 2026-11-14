"""Tarea 2.2 - Motor de recomendación (Ruta Adaptativa).

Sistema de recomendación híbrido que asigna automáticamente "misiones de refuerzo"
de Investigación de Operaciones según el perfil detectado:

  1. Basado en conocimiento: el perfil diagnosticado (Frustración, Desconexión...)
     y el nivel de riesgo fijan el tipo y nivel de misión adecuados.
  2. Basado en contenido: el tema IO más débil (la evaluación con peor nota) y el
     tema que se cursa en ese momento orientan el contenido de la misión.
  3. Colaborativo por pares: se compara la mezcla de recursos que usa el
     estudiante con la de sus pares exitosos del mismo curso; el formato con
     mayor déficit (lectura, práctica, foro, recurso) se prioriza.

Puntaje de cada misión m para el estudiante e:
  S(e,m) = 3·perfil + 2·tema_débil + 1·tema_actual + 1,5·formato_déficit + 1·ajuste_nivel

Genera data/processed/rutas.pkl y outputs/rutas_adaptativas.csv
"""
import json
import time

import numpy as np
import pandas as pd

from comun import CLAVE, CORTE_PRINCIPAL, DIR_CONFIG, DIR_PROC, DIR_SALIDAS

PESOS = {"perfil": 3.0, "tema_debil": 2.0, "tema_actual": 1.0, "formato": 1.5, "nivel": 1.0}
FORMATOS = {
    "oucontent": ["clics_oucontent"],
    "quiz": ["clics_quiz"],
    "forumng": ["clics_forumng"],
    "resource": ["clics_resource", "clics_url", "clics_subpage"],
}
NOMBRE_FORMATO = {"oucontent": "lectura interactiva", "quiz": "práctica evaluada",
                  "forumng": "trabajo colaborativo", "resource": "recursos de apoyo"}
PERFILES_RIESGO = {"Desconexión", "Frustración", "Procrastinación", "Bajo rendimiento",
                   "Riesgo general"}

catalogo = json.loads((DIR_CONFIG / "misiones_io.json").read_text(encoding="utf-8"))
TEMAS = catalogo["temas"]
MISIONES = pd.DataFrame(catalogo["misiones"])
ESPECIALES = catalogo["misiones_especiales"]


def proporciones_formato(df: pd.DataFrame) -> pd.DataFrame:
    tot = pd.DataFrame({f: df[cols].sum(axis=1) for f, cols in FORMATOS.items()})
    return tot.div(tot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def referencia_pares(df: pd.DataFrame) -> pd.DataFrame:
    """Mezcla media de formatos de los estudiantes exitosos (entrenamiento) por curso y corte."""
    exitosos = df[(df["conjunto"] == "entrenamiento") & (df["riesgo"] == 0)]
    prop = proporciones_formato(exitosos)
    prop[["code_module", "dia_corte"]] = exitosos[["code_module", "dia_corte"]]
    return prop.groupby(["code_module", "dia_corte"]).mean()


def tema_debil(df: pd.DataFrame, entregas: pd.DataFrame) -> pd.Series:
    """Tema IO asociado a la evaluación con peor resultado hasta el corte (omitida = 0)."""
    salida = pd.Series(np.nan, index=df.index)
    for corte, g in df.groupby("dia_corte"):
        e = entregas[entregas["fecha_limite"] <= corte].copy()
        e["nota"] = np.where(e["entregada"] & (e["date_submitted"] <= corte), e["score"], 0)
        peor = e.loc[e.groupby(CLAVE)["nota"].idxmin(), CLAVE + ["fecha_limite"]]
        m = g[CLAVE + ["module_presentation_length"]].reset_index().merge(peor, on=CLAVE)
        idx = (m["fecha_limite"] / (m["module_presentation_length"] / len(TEMAS))).astype(int)
        salida.loc[m["index"]] = idx.clip(0, len(TEMAS) - 1).to_numpy()
    return salida


def recomendar(df: pd.DataFrame, entregas: pd.DataFrame, pares: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    tema_act = (df["dia_corte"] / (df["module_presentation_length"] / len(TEMAS))).astype(int)
    tema_act = tema_act.clip(0, len(TEMAS) - 1).to_numpy()
    t_debil = tema_debil(df, entregas)
    t_debil = t_debil.fillna(pd.Series(tema_act, index=df.index)).astype(int).to_numpy()

    prop = proporciones_formato(df)
    # Para cortes semanales se usa la referencia de pares del corte de entrenamiento más cercano
    cortes_ref = np.array(sorted(pares.index.get_level_values("dia_corte").unique()))
    corte_ref = cortes_ref[np.abs(df["dia_corte"].to_numpy()[:, None] - cortes_ref).argmin(axis=1)]
    ref = pares.reindex(pd.MultiIndex.from_arrays([df["code_module"], corte_ref])).to_numpy()
    deficit = np.nan_to_num(ref - prop.to_numpy())
    formato_def = np.array(list(FORMATOS))[deficit.argmax(axis=1)]

    nivel_obj = np.where(df["perfil"].isin(PERFILES_RIESGO), 1,
                         np.where(df["perfil"] == "Avanzado", 3, 2))

    # Matriz de puntajes estudiantes x misiones
    tema_m = MISIONES["tema"].map(TEMAS.index).to_numpy()
    perfil_ok = np.array([[p in ps for ps in MISIONES["perfiles"]] for p in df["perfil"]])
    s = (PESOS["perfil"] * perfil_ok
         + PESOS["tema_debil"] * (tema_m[None, :] == t_debil[:, None])
         + PESOS["tema_actual"] * (tema_m[None, :] == tema_act[:, None])
         + PESOS["formato"] * (MISIONES["formato"].to_numpy()[None, :] == formato_def[:, None])
         + PESOS["nivel"] * (1 - np.abs(MISIONES["nivel"].to_numpy()[None, :] - nivel_obj[:, None]) / 2))
    # Nunca saltar dos niveles: un reto (3) no se asigna a quien necesita refuerzo (1) ni viceversa
    s[np.abs(MISIONES["nivel"].to_numpy()[None, :] - nivel_obj[:, None]) == 2] -= 10

    orden = np.argsort(-s, axis=1, kind="stable")
    ids = MISIONES["id"].to_numpy()
    rutas = []
    for i in range(n):
        elegidas, temas_usados = [], {}
        for j in orden[i]:
            t = tema_m[j]
            if temas_usados.get(t, 0) < 2:
                elegidas.append(j)
                temas_usados[t] = temas_usados.get(t, 0) + 1
            if len(elegidas) == 3:
                break
        elegidas.sort(key=lambda j: MISIONES["nivel"].iat[j])
        rutas.append([ids[j] for j in elegidas])

    rec = pd.DataFrame(index=df.index)
    especial = df["perfil"].map({k: v["id"] for k, v in ESPECIALES.items()})
    rec["ruta"] = [([e] if isinstance(e, str) else []) + r for e, r in zip(especial, rutas)]
    rec["mision_principal"] = [r[0] for r in rutas]
    rec["puntaje_mision"] = s[np.arange(n), orden[:, 0]].round(2)
    rec["tema_debil"] = [TEMAS[t] for t in t_debil]
    rec["tema_actual"] = [TEMAS[t] for t in tema_act]
    rec["formato_deficit"] = formato_def
    rec["nivel_objetivo"] = nivel_obj
    rec["motivo"] = (
        "Perfil " + df["perfil"] + " (riesgo " + (df["prob_riesgo"] * 100).round().astype(int).astype(str)
        + "%). Tema a reforzar: " + rec["tema_debil"] + ". Formato poco usado frente a pares exitosos: "
        + rec["formato_deficit"].map(NOMBRE_FORMATO) + "."
    )
    return rec


if __name__ == "__main__":
    t0 = time.time()
    inst = pd.read_pickle(DIR_PROC / "instantaneas.pkl")
    entregas = pd.read_pickle(DIR_PROC / "entregas.pkl")
    pares = referencia_pares(inst)
    rec = recomendar(inst, entregas, pares)
    rutas = inst.join(rec)
    rutas.to_pickle(DIR_PROC / "rutas.pkl")
    pares.to_csv(DIR_SALIDAS / "referencia_pares_exitosos.csv")

    titulos = MISIONES.set_index("id")["titulo"].to_dict()
    titulos.update({v["id"]: v["titulo"] for v in ESPECIALES.values()})
    vista = rutas[rutas["dia_corte"] == CORTE_PRINCIPAL].copy()
    vista["ruta_adaptativa"] = vista["ruta"].map(lambda r: " → ".join(f"{m} {titulos[m]}" for m in r))
    cols = CLAVE + ["dia_corte", "prob_riesgo", "nivel_riesgo", "perfil", "tema_debil",
                    "formato_deficit", "mision_principal", "ruta_adaptativa", "motivo"]
    vista[cols].sort_values("prob_riesgo", ascending=False).to_csv(
        DIR_SALIDAS / "rutas_adaptativas.csv", index=False, encoding="utf-8-sig")

    print(vista["perfil"].value_counts().to_string())
    print(vista["mision_principal"].value_counts().head(10).to_string())
    print(vista[["perfil", "ruta_adaptativa"]].drop_duplicates("perfil").to_string(index=False))
    print(f"Rutas generadas para {len(rutas):,} instantáneas en {time.time() - t0:.0f}s")
