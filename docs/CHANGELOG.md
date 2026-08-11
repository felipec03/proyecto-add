# Changelog — Historial de prompts

Registro de prompts realizados durante el desarrollo del proyecto, reconstruido a partir del historial de Git y contexto de conversaciones.  
Las fechas corresponden a los timestamps de los commits.

---

## Sesión 1 — 2026-08-07 — Inicialización del proyecto

| # | Prompt | Resultado / Archivos generados |
|---|---|---|
| 1 | *"Crear el repositorio e inicializar el proyecto"* | Commit `d7e6082`: README.md inicial |
| 2 | *"Agregar pipeline completo de análisis (EDA, preprocesamiento, PCA, UMAP, OPTICS), datos, gráficos y planificación"* | Commit `8b04782`: 35 archivos — `analisis_consolidado.py`, `proyecto-add-1-2026.ipynb`, datos, 16 gráficos PNG, scripts auxiliares, planificación y PDFs de referencia |

**Archivos creados**: pipeline Python, notebook 48 celdas, datasets, gráficos, scripts EDA/preprocesamiento/rebuild.

---

## Sesión 2 — 2026-08-08 / 2026-08-09 — Presentación LaTeX

| # | Prompt | Resultado / Archivos generados |
|---|---|---|
| 3 | *"Crear presentación beamer en LaTeX con estructura KDD, mostrando todos los gráficos generados (EDA, PCA, UMAP, OPTICS, evaluación)"* | Commit `8801001`: `presentacion/presentacion.tex` (421 líneas, 7 secciones) + `presentacion.pdf` compilado |

**Nota del commit**: *"falta k-means entero csm, mas encima esta entero malo el ppt, reformular el ipynb. mencionar kdd explicitamente en el informe"* — el usuario identificó que K-Means no estaba implementado y que el notebook necesitaba reformularse.

---

## Sesión 3 — 2026-08-11 — Revisión de estado y planificación

| # | Prompt | Resultado / Archivos generados |
|---|---|---|
| 4 | *"Revisa el estado del proyecto y elabora un TODO respecto a lo que falte en términos de: notebook alineado con KDD, informe en formato LaTeX, presentación en formato LaTeX. Contrasta con Proyecto-PostParo.pdf y Planificacion_Proyecto_ADD_DEFINITIVA.pdf"* | Exploración exhaustiva del proyecto; diagnóstico de completitud por etapa KDD y por categoría de entregable |
| 5 | *"Deja el output en TODO.md y déjalo en docs, igualmente reordena el README para que sea descriptivo y pushea en main"* | Commit `6fff155`: `docs/TODO.md` (priorizado en 3 categorías), `README.md` reescrito con estructura y metodología KDD |
| 6 | *"También, en docs tengo que dejar un changelog con los prompts que se han hecho históricamente en todas las conversaciones"* | `docs/CHANGELOG.md` (este archivo) |

---

## Pendientes identificados (ver `docs/TODO.md`)

1. Ejecutar el notebook completo
2. Implementar K-Means (EX1 y EX3 del plan experimental)
3. Crear informe LaTeX desde cero
4. Reemplazar placeholders en la presentación
5. Completar celdas markdown con interpretación académica
