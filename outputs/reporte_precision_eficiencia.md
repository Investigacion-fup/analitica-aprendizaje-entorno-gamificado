# Reporte de precisión y eficiencia del sistema

Cohorte de prueba: **2014J** (no usada en el entrenamiento).

## Diagnóstico de riesgo (Random Forest)
| Día de corte | n | Accuracy | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| 30 | 9,203 | 0.691 | 0.610 | 0.669 | 0.638 | 0.757 |
| 60 | 8,818 | 0.748 | 0.669 | 0.670 | 0.669 | 0.813 |
| 90 | 8,533 | 0.772 | 0.684 | 0.682 | 0.683 | 0.837 |
| 120 | 8,260 | 0.809 | 0.720 | 0.715 | 0.718 | 0.873 |
| 150 | 8,010 | 0.845 | 0.761 | 0.747 | 0.754 | 0.904 |

AUC en validación cruzada (entrenamiento): 0.859.

Calibración de niveles (día 90), tasa real de fracaso/abandono: Bajo: 11%, Moderado: 28%, Alto: 48%, Crítico: 87%

## Detección anticipada del abandono
- 83.9% de los retirados evaluables recibió una alerta antes de darse de baja.
- Mediana de anticipación: 68 días.
- 74.5% fue alertado con 2 semanas o más de margen.

## Recomendaciones (rutas adaptativas)
- Perfiles de riesgo con misión de refuerzo (nivel 1): 83.4%
- Avanzados con misión de reto (nivel 3): 100.0%
- Rutas que incluyen el tema más débil: 100.0%
- Estudiantes en riesgo que aumentaron el formato recomendado: tasa de éxito 53.9% vs 32.5% (diferencia ajustada por riesgo +10.0%, p = 1.52e-16).

## Ajuste dinámico de dificultad
- AUC del nivel final como indicador de éxito: 0.810
- Nivel final medio: exitosos 3.83 vs en riesgo 2.50
- Cambio medio por semana: 0.056 niveles; reversiones de dirección por estudiante: 0.15
- Semanas dentro de la zona de flujo: 53.5%

## Eficiencia
- 10,655,280 registros de clics procesados en 12 s.
- Entrenamiento: 13 s; predicción: 0.008 ms por estudiante.

Figuras en `outputs/figuras/`.
