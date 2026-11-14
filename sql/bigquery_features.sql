-- Tarea 1.1 (alternativa en la nube) - Ingeniería de variables de OULAD en BigQuery.
-- Equivale a la parte de interacción y evaluación de src/comun.py::construir_features
-- para un día de corte. Pasos previos:
--   bq mk oulad
--   bq load --autodetect --skip_leading_rows=1 --null_marker='?' oulad.studentVle oulad/studentVle.csv
--   (repetir para studentInfo, studentRegistration, studentAssessment, assessments, vle, courses)

DECLARE corte INT64 DEFAULT 90;

WITH activos AS (
  SELECT i.code_module, i.code_presentation, i.id_student, i.final_result,
         IF(i.final_result IN ('Withdrawn', 'Fail'), 1, 0) AS riesgo
  FROM oulad.studentInfo i
  JOIN oulad.studentRegistration r USING (code_module, code_presentation, id_student)
  WHERE r.date_unregistration IS NULL OR r.date_unregistration > corte
),
clics AS (
  SELECT s.code_module, s.code_presentation, s.id_student,
         SUM(s.sum_click)                                         AS clics_total,
         COUNT(DISTINCT s.date)                                   AS dias_activos,
         corte - MAX(s.date)                                      AS dias_sin_actividad,
         SUM(IF(v.activity_type = 'oucontent', s.sum_click, 0))   AS clics_oucontent,
         SUM(IF(v.activity_type = 'quiz', s.sum_click, 0))        AS clics_quiz,
         SUM(IF(v.activity_type = 'forumng', s.sum_click, 0))     AS clics_forumng,
         SUM(IF(v.activity_type IN ('resource', 'url', 'subpage'), s.sum_click, 0)) AS clics_recursos
  FROM oulad.studentVle s
  JOIN oulad.vle v USING (id_site, code_module, code_presentation)
  WHERE s.date <= corte
  GROUP BY 1, 2, 3
),
evaluaciones AS (
  SELECT a.code_module, a.code_presentation, act.id_student,
         COUNT(*)                                              AS n_eval_esperadas,
         COUNTIF(sa.date_submitted <= corte)                   AS n_eval_entregadas,
         AVG(IF(sa.date_submitted <= corte, sa.score, NULL))   AS puntaje_medio,
         COUNTIF(sa.date_submitted > a.date)                   AS n_tardias
  FROM oulad.assessments a
  JOIN activos act USING (code_module, code_presentation)
  LEFT JOIN oulad.studentAssessment sa
    ON sa.id_assessment = a.id_assessment AND sa.id_student = act.id_student
  WHERE a.assessment_type != 'Exam' AND a.date <= corte
  GROUP BY 1, 2, 3
)
SELECT act.*, c.* EXCEPT (code_module, code_presentation, id_student),
       e.* EXCEPT (code_module, code_presentation, id_student),
       SAFE_DIVIDE(c.clics_total,
                   APPROX_QUANTILES(c.clics_total, 2)[OFFSET(1)]
                     OVER (PARTITION BY act.code_module, act.code_presentation)) AS clics_rel_curso,
       corte AS dia_corte
FROM activos act
LEFT JOIN clics c USING (code_module, code_presentation, id_student)
LEFT JOIN evaluaciones e USING (code_module, code_presentation, id_student);
