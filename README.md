# Proyecto Análisis de Datos — 1-2026

Proyecto de la asignatura **Análisis de Datos**, semestre 1-2026.  
Aplicación de la metodología **KDD** (Knowledge Discovery in Databases) sobre el dataset **Heart Disease** del repositorio UCI Machine Learning.

## Objetivo

Aplicar técnicas de reducción dimensional (PCA, UMAP), clustering (K-Means, OPTICS) y evaluación de modelos sobre datos clínicos de enfermedades cardíacas, documentando cada etapa del proceso KDD.

## Estructura del proyecto

```
proyecto-add/
├── proyecto-add-1-2026.ipynb       # Notebook principal (KDD)
├── analisis_consolidado.py         # Pipeline completo (EDA a clustering)
├── data/                           # Datasets (raw + preprocesado)
├── plots/                          # Gráficos organizados por etapa KDD
│   ├── 1_seleccion_eda/
│   ├── 2_preprocesamiento/
│   ├── 3_transformacion/
│   ├── 4_mineria/
│   └── 5_evaluacion/
├── scripts/                        # Scripts auxiliares (EDA, preprocesamiento)
├── presentacion/                   # Presentación LaTeX (beamer)
├── docs/                           # Documentación y planificación
└── Planificacion/                  # Versiones históricas de planificación
```

## Metodología KDD

1. Comprensión del dominio y objetivos
2. Selección de datos (UCI Heart Disease)
3. Preprocesamiento y limpieza (nulos, outliers, one-hot encoding, escalado)
4. Transformación y reducción dimensional (PCA, UMAP)
5. Data mining — Clustering (K-Means, OPTICS)
6. Evaluación (Silhouette, Davies-Bouldin, Calinski-Harabasz)
7. Interpretación y comunicación de resultados

## Requisitos

```
numpy, pandas, matplotlib, seaborn, scikit-learn, umap-learn, scipy
```
