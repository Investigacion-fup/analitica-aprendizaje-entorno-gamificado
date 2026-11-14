# Detección temprana del riesgo académico y adaptación dinámica de un entorno gamificado mediante analítica del aprendizaje

Código, configuración y resultados del artículo:

> [Autores anonimizados para revisión] (2026). *Detección temprana del riesgo académico y adaptación dinámica de un entorno
> gamificado mediante analítica del aprendizaje*. Pixel-Bit. Revista de Medios y Educación. [DOI del artículo cuando se publique]

El sistema procesa los registros de interacción del Open University Learning Analytics Dataset (OULAD) para:

1. **Diagnosticar** semanalmente el riesgo de reprobación o abandono con un Random Forest, sin usar información futura.
2. **Explicar** la causa del riesgo con perfiles interpretables (desconexión, frustración, procrastinación, bajo rendimiento).
3. **Recomendar** rutas de misiones de refuerzo de Investigación de Operaciones con un recomendador híbrido.
4. **Ajustar** la dificultad de un simulador gamificado de fábrica con reglas configurables (JSON).
5. **Visualizar** los resultados en un dashboard docente.

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `src/` | Código del flujo de análisis (etapas 01 a 07) |
| `config/` | Catálogo de misiones de IO (`misiones_io.json`) y reglas de ajuste de dificultad (`reglas_dificultad.json`) |
| `sql/` | Consulta equivalente para construir las variables en BigQuery (opcional) |
| `notebooks/` | Notebook de validación ejecutado (`.ipynb`) y su versión en HTML |
| `outputs/` | Métricas, reportes, rutas asignadas, trayectorias de dificultad y figuras |
| `models/` | Modelo Random Forest entrenado (`joblib`) |
| `dashboard/` | Tablas para Power BI / Looker Studio (`datos_bi/`), tablero HTML y captura del dashboard en Power BI |
| `oulad/` | Carpeta donde deben copiarse los datos de OULAD (ver `oulad/LEEME.txt`) |
| `run_all.py` | Ejecuta todo el flujo en orden |

### Etapas del código

| Script | Qué hace |
|---|---|
| `01_preparar_datos.py` | Lee y limpia las 7 tablas de OULAD y agrega los clics por estudiante, día y tipo de recurso |
| `02_modelo_riesgo.py` | Construye las variables por día de corte y entrena y valida el Random Forest (cohorte de prueba 2014J) |
| `03_motor_recomendacion.py` | Asigna las rutas de misiones de IO |
| `04_monitoreo_semanal.py` | Simula la operación semana a semana sobre la cohorte de prueba |
| `05_ajuste_dificultad.py` | Calcula el nivel de dificultad y la configuración de la fábrica por estudiante y semana |
| `06_dashboard.py` | Exporta las tablas del dashboard y genera el tablero HTML |
| `07_validacion.py` | Calcula las métricas de precisión, anticipación, coherencia, equidad y eficiencia |
| `07b_crear_notebook.py` | Genera y ejecuta el notebook de validación |
| `comun.py` | Rutas, constantes y construcción de variables (compartido) |

## Cómo reproducir los resultados

1. Instalar Python 3.11 o superior y las dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Descargar OULAD y copiar los 7 archivos CSV en `oulad/` (instrucciones en `oulad/LEEME.txt`).
3. Ejecutar:
   ```
   python run_all.py
   ```
   El flujo completo tarda alrededor de 6 minutos en un computador portátil y regenera `outputs/`, `models/`,
   `dashboard/` y `notebooks/`. Los datos intermedios se guardan en `data/processed/`.

Para consultar el ajuste de dificultad de un estudiante concreto:
```
cd src
python 05_ajuste_dificultad.py --estudiante 24734 --modulo AAA
```

## Resultados principales (cohorte de prueba 2014J)

| Indicador | Valor |
|---|---|
| AUC del diagnóstico al día 90 | .837 (exactitud 77.2%) |
| AUC al día 150 | .904 |
| Estudiantes que abandonaron y recibieron una alerta previa | 83.9% (mediana de 68.5 días de anticipación) |
| AUC del nivel de dificultad final como indicador de éxito | .810 |

El detalle está en `outputs/reporte_precision_eficiencia.md` y en `notebooks/validacion_sistema.html`.

## Datos

Los datos de entrada provienen del **Open University Learning Analytics Dataset** (licencia CC BY 4.0):
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*,
170171. https://doi.org/10.1038/sdata.2017.171

## Licencia

Código, configuración, resultados y figuras: **CC BY 4.0** (ver `LICENSE.txt`).

## Declaración de uso de inteligencia artificial

En el desarrollo de este código se utilizó Claude (Anthropic) como herramienta de apoyo, principalmente en la
programación del flujo de procesamiento y modelado de datos. Las autoras verificaron el código y los resultados.
