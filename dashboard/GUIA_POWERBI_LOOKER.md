# Guía: Dashboard docente en Power BI / Looker Studio / Tableau

El tablero listo para usar es `dashboard_docente.html` (ábrelo con doble clic en cualquier navegador).
Si el semillero debe entregarlo en Power BI, Looker Studio o Tableau, usa los CSV de `datos_bi/`.

## Tablas (modelo estrella)

| Archivo | Tipo | Clave | Contenido |
|---|---|---|---|
| `hecho_monitoreo_semanal.csv` | Hechos | code_module + code_presentation + id_student + semana | Riesgo, perfil, ruta, misión principal, dificultad y variables de la fábrica cada semana |
| `dim_estudiante.csv` | Dimensión | code_module + code_presentation + id_student | Datos demográficos y resultado final |
| `dim_mision.csv` | Dimensión | id | Catálogo de misiones de IO (tema, nivel, formato) |

Relaciones: `hecho[mision_principal] → dim_mision[id]` y `hecho[clave estudiante] → dim_estudiante`.
En Power BI crea una columna `clave = [code_module] & "-" & [code_presentation] & "-" & [id_student]` en ambas tablas.

## Páginas sugeridas

1. **Panorama semanal**: segmentador de `semana` (control deslizante) y de `code_module`; tarjetas con las medidas de abajo; gráfico de líneas `% en riesgo` y `nivel_dificultad` promedio por semana.
2. **¿Quién necesita ayuda?**: tabla filtrada a `necesita_ayuda = 1`, ordenada por `prob_riesgo`, con `perfil`, `dias_sin_actividad`, `puntaje_medio`, `ruta` y `motivo`. Formato condicional por `nivel_riesgo`.
3. **Rutas de la IA**: barras de `mision_principal` por `perfil`; matriz `tema_debil × formato_deficit`.
4. **Fábrica adaptativa**: dispersión `desempeno` vs `nivel_dificultad`; promedios de `velocidad_maquinas`, `tasa_llegada_pedidos`, `nivel_pistas` por perfil.

## Medidas DAX

```DAX
Estudiantes activos = DISTINCTCOUNT(hecho_monitoreo_semanal[clave])
Necesitan ayuda = CALCULATE([Estudiantes activos], hecho_monitoreo_semanal[necesita_ayuda] = 1)
% en riesgo = DIVIDE([Necesitan ayuda], [Estudiantes activos])
Riesgo crítico = CALCULATE([Estudiantes activos], hecho_monitoreo_semanal[nivel_riesgo] = "Crítico")
Dificultad media = AVERAGE(hecho_monitoreo_semanal[nivel_dificultad])
```

## Looker Studio

Sube los CSV a Google Sheets o a BigQuery (ver `sql/bigquery_features.sql`), crea una fuente de datos combinada
uniendo por `code_module`, `code_presentation` e `id_student`, y replica las páginas anteriores. Para el campo
calculado de % en riesgo: `SUM(necesita_ayuda) / COUNT_DISTINCT(id_student)`.
