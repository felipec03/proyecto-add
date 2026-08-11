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

## Sesión 4 — 2026-08-11 — Integración de K-Means y actualización de entregables

| # | Prompt | Resultado / Archivos generados |
|---|---|---|
| 7 | *"Tomando en cuenta el changelog y el TODO anda rellenando lo que haga falta, deja las celdas de explicación para hacerlas nosotros, lo importante es integrar K-Means y lo que falte en función de más tarde hacer el informe y la respectiva presentación"* | K-Means implementado (EX1 = PCA + K-Means, EX3 = UMAP + K-Means) en notebook y pipeline; notebook ejecutado completo (56 celdas); presentación actualizada y recompilada |

**Cambios realizados**:

- **Notebook** (`proyecto-add-1-2026.ipynb` + `_ejecutado.ipynb`): sección K-Means con método del codo (inercia por k), Silhouette/Davies-Bouldin/Calinski-Harabasz/ARI/NMI/Dunn por k, gráfico Silhouette del k óptimo, clusters finales de EX1 y EX3, interpretación de centroides (perfil clínico por cluster + cruce con severidad), métricas adicionales para OPTICS y tabla comparativa final K-Means vs OPTICS. Se corrigió `matplotlib.use('Agg')` → `%matplotlib inline` (las figuras ahora sí se muestran inline tras la celda 18). Referencias actualizadas (Lloyd, MacQueen, Hartigan-Wong, Rousseeuw, Calinski-Harabasz, Davies-Bouldin, Dunn, Hubert-Arabie, Vinh).
- **Pipeline** (`analisis_consolidado.py`): sección 7 K-Means espejo del notebook, reproducible con `python analisis_consolidado.py`.
- **Plots**: 6 nuevos — `4_mineria/kmeans_codo_pca.png`, `kmeans_codo_umap.png`, `kmeans_silhouette.png`, `kmeans_pca.png`, `kmeans_umap.png`, `5_evaluacion/clustering_metrics.png`.
- **Presentación** (`presentacion/presentacion.tex`): tabla OPTICS con valores reales reemplazada por tabla EX1-EX4 (Silhouette, DB, CH, ARI, NMI); 5 slides K-Means nuevos; gráfico comparativo normalizado; corregidos valores desactualizados (outliers 35→19, filas 268→284, Kaiser 7/59.1%→10/76.8%, 80% 12→11 componentes, varianza 2D 27%→29.5%, loadings PC1/PC2, correlaciones con target). PDF recompilado.

**Resultados clave obtenidos** (para informe y presentación):

- K-Means PCA (k=4): Silhouette 0.447, DB 0.788, CH 338.2, ARI 0.177, NMI 0.219. Cluster 1: 92% pacientes sanos (jóvenes, thalach alto).
- K-Means UMAP (k=2): Silhouette 0.641, DB 0.378, CH 440.8, ARI -0.086 — grupos compactos pero sin correspondencia con niveles de severidad.
- OPTICS PCA: 20 clusters, 39.1% ruido, Silhouette 0.546, CH 506.9. OPTICS UMAP: 27 clusters, 16.5% ruido, Silhouette 0.567, CH 2194.1.
- ARI/NMI bajos en todos los modelos: los clusters reflejan perfiles clínicos, no las etiquetas de severidad 0-4.

---

## Pendientes identificados (ver `docs/TODO.md`)

1. Escribir el informe LaTeX (estructura en `informe/informe.md`)
2. Completar celdas markdown de explicación/interpretación en el notebook (datos de ejecución ya disponibles en las celdas 49-53)
3. Opcionales: t-SNE, PCA 3D, pulido final de la presentación
