"""Ejecuta el flujo completo de análisis en orden y registra el tiempo de cada etapa.

Uso:  python run_all.py          (todo)
      python run_all.py 04       (desde la etapa 04)
Requiere los archivos CSV de OULAD en la carpeta oulad/ (ver oulad/LEEME.txt).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent / "src"
ETAPAS = [
    "01_preparar_datos.py",
    "02_modelo_riesgo.py",
    "03_motor_recomendacion.py",
    "04_monitoreo_semanal.py",
    "05_ajuste_dificultad.py",
    "06_dashboard.py",
    "07_validacion.py",
    "07b_crear_notebook.py",
]

desde = sys.argv[1] if len(sys.argv) > 1 else "01"
tiempos_path = Path(__file__).parent / "outputs" / "tiempos_pipeline.json"
tiempos = json.loads(tiempos_path.read_text()) if tiempos_path.exists() else {}
for etapa in ETAPAS:
    if etapa[:2] < desde:
        continue
    print(f"\n=== {etapa} ===", flush=True)
    t = time.time()
    subprocess.run([sys.executable, etapa], cwd=SRC, check=True)
    tiempos[etapa] = round(time.time() - t, 1)
    tiempos_path.parent.mkdir(exist_ok=True)
    tiempos_path.write_text(json.dumps(tiempos, indent=2))
print("\nTiempos (s):", json.dumps(tiempos, indent=2))
