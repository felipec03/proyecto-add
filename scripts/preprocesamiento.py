import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PLOTS = BASE / 'plots'
DATA = BASE / 'data'

# ============================================================
# 0. CARGA DE DATOS
# ============================================================
heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.features.copy()
target = heart_disease.data.targets.copy()

# Unir para tener nombre de target claro
target = target.rename(columns={'num': 'target'})
df_original = pd.concat([df, target], axis=1)

print(f"Shape original: {df_original.shape}")
print(f"Target distribucion:\n{df_original['target'].value_counts().sort_index()}\n")

# ============================================================
# 1. TRATAMIENTO DE NULOS
# ============================================================
print("=" * 50)
print("  1. TRATAMIENTO DE NULOS")
print("=" * 50)

nulos_antes = df_original.isnull().sum()
print(f"\nNulos por columna (antes):\n{nulos_antes[nulos_antes > 0]}\n")

# ca: variable discreta (0-3). Imputar con la mediana
mediana_ca = df_original['ca'].median()
df_original['ca'] = df_original['ca'].fillna(mediana_ca)
print(f"-> ca: {int(df_original['ca'].isnull().sum())} nulos imputados con mediana ({mediana_ca})")

# thal: categorica (3,6,7). Imputar con la moda
moda_thal = df_original['thal'].mode()[0]
df_original['thal'] = df_original['thal'].fillna(moda_thal)
print(f"-> thal: {int(df_original['thal'].isnull().sum())} nulos imputados con moda ({int(moda_thal)})")

print(f"\nNulos restantes: {df_original.isnull().sum().sum()}")
print(f"Shape post-imputacion: {df_original.shape}")

# ============================================================
# 2. ELIMINACION DE OUTLIERS (IQR)
# ============================================================
print("\n" + "=" * 50)
print("  2. ELIMINACION DE OUTLIERS (metodo IQR)")
print("=" * 50)

# Variables continuas propensas a outliers
cols_outlier = ['chol', 'trestbps', 'oldpeak', 'thalach', 'age']

outliers_por_columna = {}
total_outliers = set()

for col in cols_outlier:
    Q1 = df_original[col].quantile(0.25)
    Q3 = df_original[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    mascara_outlier = (df_original[col] < limite_inf) | (df_original[col] > limite_sup)
    indices_outlier = df_original.index[mascara_outlier].tolist()
    outliers_por_columna[col] = len(indices_outlier)
    total_outliers.update(indices_outlier)

    print(f"  {col:>10s}: {len(indices_outlier):3d} outliers  "
          f"(limites: [{limite_inf:.1f}, {limite_sup:.1f}])")

print(f"\n  Total combinado de filas con al menos un outlier: {len(total_outliers)}")

df_clean = df_original.drop(index=total_outliers).reset_index(drop=True)
print(f"  Filas eliminadas: {len(total_outliers)}")
print(f"  Shape post-outliers: {df_clean.shape}")
print(f"  Target distribucion post-outliers:\n{df_clean['target'].value_counts().sort_index()}\n")

# Guardar target aparte antes de transformaciones
y = df_clean['target']
X = df_clean.drop(columns=['target'])

# ============================================================
# 3. CODIFICACION DE VARIABLES CATEGORICAS
# ============================================================
print("=" * 50)
print("  3. CODIFICACION DE VARIABLES CATEGORICAS")
print("=" * 50)

# Variables binarias: mantener como 0/1  (no necesitan one-hot)
binarias = ['sex', 'fbs', 'exang']

# Variables multi-categoria (nominales u ordinales): one-hot encoding
categoricas_multiclase = ['cp', 'restecg', 'slope', 'thal']

# Variables continuas
continuas = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

print(f"  Binarias (sin cambios):     {binarias}")
print(f"  Multi-categoria (one-hot):  {categoricas_multiclase}")
print(f"  Continuas (solo escalar):   {continuas}")

# One-hot encoding sobre las multi-categoria
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_array = encoder.fit_transform(X[categoricas_multiclase])
encoded_cols = encoder.get_feature_names_out(categoricas_multiclase)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)

print(f"\n  Columnas generadas por one-hot: {list(df_encoded.columns)}")

# Ensamblar dataframe final (sin escalar aun): continuas + binarias + one-hot
df_pre_escalado = pd.concat([
    X[continuas],
    X[binarias],
    df_encoded
], axis=1)

print(f"  Shape pre-escalado: {df_pre_escalado.shape}")

# ============================================================
# 4. ESCALAMIENTO / ESTANDARIZACION
# ============================================================
print("\n" + "=" * 50)
print("  4. ESCALAMIENTO (StandardScaler)")
print("=" * 50)

scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df_pre_escalado)
df_scaled = pd.DataFrame(df_scaled_array, columns=df_pre_escalado.columns)

print(f"  Media por columna (debe ~0):\n{df_scaled.mean().round(4)}\n")
print(f"  Std por columna (debe =1):\n{df_scaled.std().round(4)}\n")

# ============================================================
# 5. RESUMEN FINAL
# ============================================================
print("=" * 50)
print("  RESUMEN DEL PREPROCESAMIENTO")
print("=" * 50)
print(f"  Shape inicial:             {df_original.shape}")
print(f"  Nulos imputados:           6 (ca=4, thal=2)")
print(f"  Outliers eliminados:       {len(total_outliers)} filas")
print(f"  Shape final (escalado):    {df_scaled.shape}")
print(f"  Columnas finales:          {list(df_scaled.columns)}")

# Guardar datasets procesados
df_scaled.to_csv(DATA / 'heart_disease_preprocesado.csv', index=False)
y.to_csv(DATA / 'heart_disease_target.csv', index=False)

print("\n  Archivos guardados:")
print("    - heart_disease_preprocesado.csv   (features escaladas)")
print("    - heart_disease_target.csv         (target)")

# ============================================================
# 6. VISUALIZACION: comparativa antes/despues de outliers
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
cols_vis = ['chol', 'oldpeak', 'trestbps', 'thalach', 'age']

for i, col in enumerate(cols_vis):
    ax = axes[i // 3][i % 3]
    data_antes = df_original[col]
    data_despues = X[col]
    bp = ax.boxplot([data_antes, data_despues], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_xticklabels(['Antes', 'Después'])
    ax.set_title(f'{col}')
    ax.set_ylabel(col)
    ax.grid(axis='y', alpha=0.3)

# Ocultar subplot vacio
fig.delaxes(axes[1][2])

plt.suptitle('Comparativa de Outliers: Antes vs Después del Tratamiento IQR',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS / '2_preprocesamiento' / 'outliers_antes_despues.png', dpi=150)

print("\n=== PREPROCESAMIENTO COMPLETO ===")
