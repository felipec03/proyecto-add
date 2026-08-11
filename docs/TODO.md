# TODO — Proyecto Análisis de Datos

Entrega: **13 de agosto 2026, 12:00 hrs**

---

## A. Notebook alineado con KDD

| Prioridad | Tarea | Estado |
|---|---|---|
| CRÍTICA | Ejecutar el notebook completo (56 celdas sin errores, salidas en `proyecto-add-1-2026_ejecutado.ipynb`) | Listo |
| CRÍTICA | Implementar K-Means completo: método del codo, silhouette por k, interpretación de centroides (EX1 y EX3 del plan experimental) | Listo |
| ALTA | Agregar métricas Davies-Bouldin y Calinski-Harabasz (requisito Parte 5) | Listo |
| ALTA | Completar celdas markdown pendientes: reemplazar "Yo creo que...", "???", "pq lo vamos a dejar vivir..." por texto académico | Pendiente (celdas de explicación, las escriben los autores) |
| ALTA | Completar interpretación de clustering con valores reales (datos ya generados en celdas 49-53; celdas marcadas con "[completar]") | Pendiente (celdas de explicación, las escriben los autores) |
| MEDIA | Agregar t-SNE como tercera técnica de reducción (opcional) | Pendiente |
| MEDIA | Agregar PCA 3D (opcional) | Pendiente |
| BAJA | Agregar ARI, NMI, Dunn index | Listo (incluidos en K-Means y en métricas extra de OPTICS) |

## B. Informe en formato LaTeX

| Prioridad | Tarea | Estado |
|---|---|---|
| CRÍTICA | Crear archivo `.tex` del informe escrito (no existe) | Pendiente |
| ALTA | Redactar introducción (contexto del dataset, objetivo, hipótesis exploratoria) | Pendiente |
| ALTA | Documentar metodología KDD aplicada (7 etapas) | Pendiente |
| ALTA | Incluir todos los gráficos generados (`plots/`) en el informe | Pendiente |
| ALTA | Redactar discusión: PCA vs UMAP, K-Means vs OPTICS, limitaciones | Pendiente |
| ALTA | Redactar conclusiones | Pendiente |
| ALTA | Compilar y verificar el PDF del informe | Pendiente |

## C. Presentación en formato LaTeX

| Prioridad | Tarea | Estado |
|---|---|---|
| CRÍTICA | Reemplazar los `--` en la tabla de resultados de clustering por valores reales | Listo (tabla EX1-EX4 con Silhouette, DB, CH, ARI, NMI) |
| ALTA | Agregar slides de K-Means (método del codo, silhouette, centroides) | Listo (5 slides nuevos en Sección 3) |
| ALTA | Agregar métricas Davies-Bouldin y Calinski-Harabasz a slides de evaluación | Listo (tabla comparativa + gráfico normalizado) |
| ALTA | Re-compilar `presentacion.tex` para generar PDF actualizado | Listo (`presentacion/presentacion.pdf`) |
| BAJA | Pulir formato, ortografía, consistencia visual | Pendiente |
| BAJA | Crear changelog histórico de prompts | Listo (`docs/CHANGELOG.md`) |

---

## Resumen

- **Completo**: Pipeline de datos, 22 gráficos PNG generados (incluye 6 nuevos de K-Means), notebook de 56 celdas ejecutado sin errores, K-Means con codo/silhouette/centroides (EX1 y EX3), métricas DB/CH/ARI/NMI/Dunn, presentación LaTeX con valores reales, planificación documentada.
- **Faltante (a cargo de los autores)**: Informe LaTeX, celdas markdown de explicación/interpretación, opcionales t-SNE y PCA 3D.
