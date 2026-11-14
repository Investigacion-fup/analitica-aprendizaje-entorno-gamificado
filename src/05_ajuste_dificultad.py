"""Tarea 3.1 - Lógica de ajuste dinámico de dificultad.

Las reglas viven en config/reglas_dificultad.json (no en el código), así el
docente o el equipo de diseño del juego pueden calibrarlas sin programar.

Uso en el juego (un estudiante, una semana):
    ajuste = importlib.import_module("05_ajuste_dificultad")
    estado = ajuste.ajustar(estado_anterior, desempeno=0.32, perfil="Frustración")

Uso como script (simula semana a semana la cohorte de prueba 2014J):
    python 05_ajuste_dificultad.py
    python 05_ajuste_dificultad.py --estudiante 24734 --modulo AAA
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from comun import CLAVE, DIR_CONFIG, DIR_PROC, DIR_SALIDAS, PRESENTACION_PRUEBA

REGLAS = json.loads((DIR_CONFIG / "reglas_dificultad.json").read_text(encoding="utf-8"))
SEMANAS = 22  # ~150 días, hasta el último corte del modelo


# --- Reglas (funciones puras, reutilizables por el motor del juego) ---------------
def siguiente_nivel(nivel, d_suavizado, reglas=REGLAS):
    """Controlador proporcional con zona muerta (zona de flujo) y paso máximo."""
    zmin, zmax = reglas["zona_flujo"]["min"], reglas["zona_flujo"]["max"]
    delta = np.where(d_suavizado < zmin, -(zmin - d_suavizado),
                     np.where(d_suavizado > zmax, d_suavizado - zmax, 0.0)) * reglas["ganancia"]
    paso = reglas["paso_max_semana"]
    delta = np.clip(delta, -paso, paso)
    return np.clip(nivel + delta, reglas["nivel"]["min"], reglas["nivel"]["max"])


def configurar_fabrica(nivel: float, perfil: str, reglas=REGLAS) -> dict:
    """Traduce el nivel (1-5) y el perfil a las variables concretas de la fábrica."""
    t = (nivel - reglas["nivel"]["min"]) / (reglas["nivel"]["max"] - reglas["nivel"]["min"])
    cfg = {}
    for var, r in reglas["variables_fabrica"].items():
        v = r["nivel_1"] + t * (r["nivel_5"] - r["nivel_1"])
        cfg[var] = round(v) if r.get("entero") else round(v, 3)
    cfg.update({"modo_reenganche": False, "checkpoints_entrega": False,
                "eventos_sorpresa": False, "multiplicador_recompensa": 1.0})
    for var, op in reglas["modificadores_perfil"].get(perfil, {}).items():
        if "multiplicar" in op:
            cfg[var] = round(cfg[var] * op["multiplicar"], 3)
        elif "sumar" in op:
            cfg[var] = cfg[var] + op["sumar"]
        elif "fijar" in op:
            cfg[var] = op["fijar"]
    for var, (lo, hi) in reglas["limites"].items():
        cfg[var] = min(max(cfg[var], lo), hi)
    return cfg


def ajustar(estado: dict | None, desempeno: float, perfil: str, reglas=REGLAS) -> dict:
    """Un paso del ajuste dinámico para un estudiante (API para el juego)."""
    if estado is None:
        estado = {"nivel": reglas["nivel"]["inicial"], "desempeno_suavizado": desempeno}
    a = reglas["suavizado_alpha"]
    d = a * desempeno + (1 - a) * estado["desempeno_suavizado"]
    nivel = float(siguiente_nivel(estado["nivel"], d, reglas))
    return {"nivel": round(nivel, 3), "desempeno_suavizado": round(d, 4),
            "fabrica": configurar_fabrica(nivel, perfil, reglas)}


# --- Simulación semanal sobre la cohorte de prueba ---------------------------------
def desempeno_semanal(est, clics_dia, entregas, reglas=REGLAS):
    """Matrices (estudiantes x semanas) de nota, actividad, puntualidad y desempeño D."""
    idx = pd.MultiIndex.from_frame(est[CLAVE])
    c = clics_dia[clics_dia["date"].between(0, SEMANAS * 7 - 1)]
    c = c.merge(est[CLAVE], on=CLAVE).assign(semana=lambda d: d["date"] // 7)
    sem = (c.pivot_table(index=CLAVE, columns="semana", values="sum_click", aggfunc="sum",
                         fill_value=0, observed=True)
            .reindex(index=idx, columns=range(SEMANAS), fill_value=0))
    # Referencia: mediana de clics de los compañeros activos del mismo curso esa semana
    ref = sem.where(sem > 0).groupby(level="code_module").transform("median").fillna(1)
    tope = reglas["desempeno"]["actividad_tope"]
    actividad = (sem / ref).clip(upper=tope) / tope

    e = entregas.merge(est[CLAVE], on=CLAVE)
    nota = np.full(sem.shape, np.nan)
    puntual = np.ones(sem.shape)
    for w in range(SEMANAS):
        dia = 7 * w + 6
        hechas = e[e["entregada"] & (e["date_submitted"] <= dia)]
        ultima = hechas.sort_values("date_submitted").groupby(CLAVE)["score"].last()
        nota[:, w] = ultima.reindex(idx).to_numpy() / 100
        vencidas = e[e["fecha_limite"] <= dia]
        a_tiempo = (vencidas["entregada"] & (vencidas["date_submitted"] <= vencidas["fecha_limite"]))
        p = a_tiempo.groupby([vencidas[k] for k in CLAVE]).mean()
        puntual[:, w] = p.reindex(idx).fillna(1).to_numpy()
    nota = np.nan_to_num(nota, nan=reglas["desempeno"]["nota_sin_evaluaciones"])

    pesos = reglas["desempeno"]["pesos"]
    d = pesos["nota"] * nota + pesos["actividad"] * actividad.to_numpy() + pesos["puntualidad"] * puntual
    return nota, actividad.to_numpy(), puntual, d


def perfil_semanal(est, monitoreo):
    """Perfil vigente cada semana según el monitoreo semanal (antes de la semana 4: sin diagnóstico)."""
    r = monitoreo.set_index(CLAVE + ["semana"])["perfil"]
    out = np.full((len(est), SEMANAS), "En progreso", dtype=object)
    claves = pd.MultiIndex.from_frame(est[CLAVE])
    for w in range(SEMANAS):
        if w in r.index.levels[-1]:
            p = r.xs(w, level="semana").reindex(claves)
            # si ya no hay diagnóstico (p.ej. se dio de baja) se mantiene el de la semana anterior
            out[:, w] = np.where(p.isna(), out[:, w - 1], p.to_numpy())
        elif w:
            out[:, w] = out[:, w - 1]
    return out


def simular(est, clics_dia, entregas, monitoreo, reglas=REGLAS):
    nota, actividad, puntual, d = desempeno_semanal(est, clics_dia, entregas, reglas)
    perfiles = perfil_semanal(est, monitoreo)
    a = reglas["suavizado_alpha"]
    nivel = np.full(len(est), float(reglas["nivel"]["inicial"]))
    d_s = d[:, 0].copy()
    filas = []
    for w in range(SEMANAS):
        d_s = a * d[:, w] + (1 - a) * d_s if w else d_s
        nivel_prev = nivel
        nivel = siguiente_nivel(nivel, d_s, reglas)
        filas.append(pd.DataFrame({
            **{k: est[k].to_numpy() for k in CLAVE},
            "semana": w, "nota": nota[:, w].round(3), "actividad": actividad[:, w].round(3),
            "puntualidad": puntual[:, w].round(3), "desempeno": d[:, w].round(3),
            "desempeno_suavizado": d_s.round(3), "nivel": nivel.round(3),
            "cambio_nivel": (nivel - nivel_prev).round(3), "perfil": perfiles[:, w],
        }))
    tray = pd.concat(filas, ignore_index=True)
    # Se corta la trayectoria cuando el estudiante se da de baja
    baja = tray.merge(est[CLAVE + ["date_unregistration"]], on=CLAVE)["date_unregistration"]
    tray = tray[~(baja <= tray["semana"] * 7).to_numpy()].reset_index(drop=True)

    cfg = pd.DataFrame([configurar_fabrica(n, p, reglas) for n, p in zip(tray["nivel"], tray["perfil"])])
    return pd.concat([tray, cfg], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estudiante", type=int)
    ap.add_argument("--modulo")
    args = ap.parse_args()

    t0 = time.time()
    est = pd.read_pickle(DIR_PROC / "estudiantes.pkl")
    est = est[est["code_presentation"] == PRESENTACION_PRUEBA].reset_index(drop=True)
    if args.estudiante:
        est = est[est["id_student"] == args.estudiante]
        if args.modulo:
            est = est[est["code_module"] == args.modulo]
        est = est.reset_index(drop=True)
        if est.empty:
            raise SystemExit(f"Estudiante {args.estudiante} no está en la cohorte {PRESENTACION_PRUEBA}")
    clics_dia = pd.read_pickle(DIR_PROC / "clics_diarios.pkl")
    entregas = pd.read_pickle(DIR_PROC / "entregas.pkl")
    monitoreo = pd.read_pickle(DIR_PROC / "monitoreo_semanal.pkl")

    tray = simular(est, clics_dia, entregas, monitoreo)
    if args.estudiante:
        cols = ["code_module", "semana", "desempeno", "nivel", "perfil", "velocidad_maquinas",
                "tasa_llegada_pedidos", "nivel_pistas", "tiempo_limite_turno"]
        print(tray[cols].to_string(index=False))
    else:
        tray.to_pickle(DIR_PROC / "trayectorias.pkl")
        tray.to_csv(DIR_SALIDAS / "trayectorias_dificultad.csv", index=False, encoding="utf-8-sig")
        ultima = tray.sort_values("semana").groupby(CLAVE).tail(1)
        cfg_cols = list(REGLAS["variables_fabrica"]) + ["modo_reenganche", "checkpoints_entrega",
                                                         "eventos_sorpresa", "multiplicador_recompensa"]
        config = {f"{r['code_module']}-{r['code_presentation']}-{r['id_student']}": {
            "semana": int(r["semana"]), "nivel": float(r["nivel"]), "perfil": r["perfil"],
            "fabrica": {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cfg_cols}}
            for r in ultima.to_dict("records")}
        (DIR_SALIDAS / "config_fabrica_estudiantes.json").write_text(
            json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
        print(tray.groupby("semana")[["desempeno", "nivel"]].mean().round(2).T.to_string())
        print(f"{len(config):,} configuraciones de fábrica, {len(tray):,} pasos simulados "
              f"en {time.time() - t0:.0f}s")
