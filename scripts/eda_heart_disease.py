import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PLOTS = BASE / 'plots'

# ------------------------------------------------------------
# 1. Carga del dataset
# ------------------------------------------------------------
heart_disease = fetch_ucirepo(id=45)
hd_features = heart_disease.data.features
hd_target = heart_disease.data.targets

print("=== METADATA ===")
print(heart_disease.metadata)
print("\n=== TARGET INFO ===")
print(hd_target.head())
print(hd_target.value_counts())

# Unir features y target para analisis completo
df = pd.concat([hd_features, hd_target], axis=1)

# ------------------------------------------------------------
# 2. Estadística descriptiva
# ------------------------------------------------------------
print("\n========================================")
print("  ESTADÍSTICA DESCRIPTIVA")
print("========================================\n")

# Seleccion de variables criticas
variables_clave = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak']

print("--- Variables clave ---")
print(df[variables_clave].describe().round(2))

# Media, mediana, desviacion estandar individuales
print("\n--- Media, Mediana y Desviacion Estandar ---")
for col in variables_clave:
    media = df[col].mean()
    mediana = df[col].median()
    std = df[col].std()
    print(f"{col:>10s} -> Media: {media:8.2f} | Mediana: {mediana:8.2f} | Std: {std:8.2f}")

# ------------------------------------------------------------
# 3. Histogramas
# ------------------------------------------------------------
print("\n========================================")
print("  GENERANDO HISTOGRAMAS...")
print("========================================\n")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(variables_clave):
    axes[i].hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    axes[i].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Media={df[col].mean():.1f}')
    axes[i].set_title(f'Distribución de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')
    axes[i].legend()

# Quitar ultimo subplot vacio si hay 5 variables
if len(variables_clave) < len(axes):
    for j in range(len(variables_clave), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'histogramas.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 4. Boxplots
# ------------------------------------------------------------
print("\n========================================")
print("  GENERANDO BOXPLOTS...")
print("========================================\n")

# Boxplots individuales para variables con potenciales outliers
vars_boxplot = ['chol', 'oldpeak', 'trestbps', 'thalach']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(vars_boxplot):
    axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.6))
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_ylabel(col)
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'boxplots.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 5. Boxplot combinado (todas las variables en una misma figura)
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
df_melt = df[variables_clave].melt(var_name='Variable', value_name='Valor')
sns.boxplot(x='Variable', y='Valor', data=df_melt, hue='Variable', legend=False, palette='Set2')
plt.title('Boxplots Combinados - Variables Críticas')   
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'boxplots_combinados.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 6. Matriz de correlación
# ------------------------------------------------------------
print("\n========================================")
print("  MATRIZ DE CORRELACIÓN")
print("========================================\n")

corr_matrix = df.corr(numeric_only=True)
print(corr_matrix.round(3))

# ------------------------------------------------------------
# 7. Heatmap de correlación
# ------------------------------------------------------------
print("\n========================================")
print("  GENERANDO HEATMAP DE CORRELACIÓN...")
print("========================================\n")

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Correlación'})
plt.title('Heatmap de Correlación - Heart Disease Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'heatmap_correlacion.png', dpi=150)
plt.show()

print("\n=== EXPLORACIÓN DE DATOS COMPLETA ===")
