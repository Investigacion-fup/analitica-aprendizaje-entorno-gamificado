"""Convierte 07_validacion.py (celdas '# %%') en notebooks/validacion_sistema.ipynb."""
import json
import re
import subprocess
import sys

from comun import RAIZ

fuente = (RAIZ / "src" / "07_validacion.py").read_text(encoding="utf-8")
bloques = re.split(r"^# %%(.*)$", fuente, flags=re.M)[1:]
celdas = [{
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": ["import sys, platform, datetime\n",
               "import pandas, sklearn\n",
               "sys.path.insert(0, '../src')  # para importar comun.py\n",
               "print('Ejecutado el:', datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))\n",
               "print('Python', platform.python_version(), '| pandas', pandas.__version__,\n",
               "      '| scikit-learn', sklearn.__version__, '|', platform.system(), platform.release())\n"],
}]
for marca, cuerpo in zip(bloques[::2], bloques[1::2]):
    cuerpo = cuerpo.strip("\n")
    if "[markdown]" in marca:
        texto = "\n".join(re.sub(r"^# ?", "", l) for l in cuerpo.splitlines())
        celdas.append({"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)})
    else:
        cuerpo = cuerpo.replace('matplotlib.use("Agg")\n', "")
        celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": cuerpo.splitlines(keepends=True)})
nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
for i, c in enumerate(celdas):
    c["id"] = f"celda-{i:02d}"
destino = RAIZ / "notebooks"
destino.mkdir(exist_ok=True)
(destino / "validacion_sistema.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook con {len(celdas)} celdas en notebooks/validacion_sistema.ipynb")

# Si Jupyter está instalado, se ejecuta el notebook (queda con sus resultados) y se exporta a HTML como evidencia
try:
    for args in (["--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=900"],
                 ["--to", "html"]):
        subprocess.run([sys.executable, "-m", "nbconvert", *args, "validacion_sistema.ipynb"],
                       cwd=destino, check=True, capture_output=True)
    print("Notebook ejecutado y exportado: notebooks/validacion_sistema.html")
except Exception as e:  # noqa: BLE001
    print(f"Notebook no ejecutado ({type(e).__name__}); instalar con: pip install notebook nbconvert ipykernel")
