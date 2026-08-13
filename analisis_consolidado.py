import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from pathlib import Path

BASE = Path(__file__).resolve().parent
PLOTS = BASE / 'plots'

# ============================================================
# 0. CARGA DEL DATASET
# ============================================================
print("=" * 60)
print("  0. CARGA DEL DATASET (UCI Heart Disease)")
print("=" * 60)

heart_disease = fetch_ucirepo(id=45)
hd_features = heart_disease.data.features
hd_target = heart_disease.data.targets
df = pd.concat([hd_features, hd_target], axis=1)

print(f"Shape: {df.shape}")
print(f"Target:\n{hd_target['num'].value_counts().sort_index()}\n")

# ============================================================
# 1. EXPLORACION INICIAL (EDA)
# ============================================================
print("=" * 60)
print("  1. EXPLORACION INICIAL (EDA)")
print("=" * 60)

variables_clave = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak']

print("--- Estadistica descriptiva (media, mediana, std) ---")
for col in variables_clave:
    print(f"{col:>10s} -> Media: {df[col].mean():8.2f} | "
          f"Mediana: {df[col].median():8.2f} | Std: {df[col].std():8.2f}")

# Histogramas
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(variables_clave):
    axes[i].hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    axes[i].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=1.5,
                    label=f'Media={df[col].mean():.1f}')
    axes[i].set_title(f'Distribucion de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')
    axes[i].legend()
for j in range(len(variables_clave), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'histogramas.png', dpi=300)
plt.close(fig)
print("[OK] histogramas.png")

# Boxplots individuales
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
plt.savefig(PLOTS / '1_seleccion_eda' / 'boxplots.png', dpi=300)
plt.close(fig)
print("[OK] boxplots.png")

# Boxplot combinado
plt.figure(figsize=(10, 6))
df_melt = df[variables_clave].melt(var_name='Variable', value_name='Valor')
sns.boxplot(x='Variable', y='Valor', data=df_melt, hue='Variable', legend=False, palette='Set2')
plt.title('Boxplots Combinados - Variables Criticas')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'boxplots_combinados.png', dpi=300)
plt.close()
print("[OK] boxplots_combinados.png")

# Correlacion
corr_matrix = df.corr(numeric_only=True)
print("\n--- Correlaciones significativas (|r| >= 0.3) ---")
print(corr_matrix.round(3).mask(corr_matrix.abs().lt(0.3)))

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Correlacion'})
plt.title('Heatmap de Correlacion - Heart Disease Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS / '1_seleccion_eda' / 'heatmap_correlacion.png', dpi=300)
plt.close()
print("[OK] heatmap_correlacion.png\n")

# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================
print("=" * 60)
print("  2. PREPROCESAMIENTO")
print("=" * 60)

# 2.1 Nulos
nulos_antes = df.isnull().sum()
print(f"\nNulos por columna (antes):\n{nulos_antes[nulos_antes > 0]}\n")

mediana_ca = df['ca'].median()
print(f"-> ca: {int(df['ca'].isnull().sum())} nulos rellenados con mediana ({mediana_ca})")
df['ca'] = df['ca'].fillna(mediana_ca)

moda_thal = df['thal'].mode()[0]
print(f"-> thal: {int(df['thal'].isnull().sum())} nulos rellenados con moda ({int(moda_thal)})")
df['thal'] = df['thal'].fillna(moda_thal)

# 2.2 Outliers (IQR)
cols_outlier = ['chol', 'trestbps', 'oldpeak', 'thalach', 'age']
outliers_por_columna = {}
total_outliers = set()
for col in cols_outlier:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    mascara = (df[col] < limite_inf) | (df[col] > limite_sup)
    indices = df.index[mascara].tolist()
    outliers_por_columna[col] = len(indices)
    total_outliers.update(indices)
    print(f"  {col:>10s}: {len(indices):3d} outliers  "
          f"(limites: [{limite_inf:.1f}, {limite_sup:.1f}])")

df_clean = df.drop(index=total_outliers).reset_index(drop=True)
print(f"\n  Total de filas con outliers: {len(total_outliers)}")
print(f"  Shape post-outliers: {df_clean.shape}")

y = df_clean['num']
X = df_clean.drop(columns=['num'])

# 2.3 Codificacion categorica
binarias = ['sex', 'fbs', 'exang']
categoricas_multiclase = ['cp', 'restecg', 'slope', 'thal']
continuas = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_array = encoder.fit_transform(X[categoricas_multiclase])
encoded_cols = encoder.get_feature_names_out(categoricas_multiclase)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)

X_pre_escalado = pd.concat([X[continuas], X[binarias], df_encoded], axis=1)
print(f"\n  Columnas one-hot: {list(df_encoded.columns)}")
print(f"  Shape pre-escalado: {X_pre_escalado.shape}")

# 2.4 Escalamiento
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_pre_escalado), columns=X_pre_escalado.columns)
print("  Media por columna (debe ~0):")
print(X_scaled.mean().round(4))
print("\n  Std por columna (debe ~1):")
print(X_scaled.std().round(4))

# ============================================================
# 3. REDUCCION DE DIMENSIONALIDAD: PCA
# ============================================================
print("\n" + "=" * 60)
print("  3. PCA")
print("=" * 60)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
varianza_individual = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_individual)
autovalores = pca.explained_variance_

print("=== VARIANZA EXPLICADA POR COMPONENTE ===")
for i, (v_ind, v_acum, eig) in enumerate(zip(varianza_individual, varianza_acumulada, autovalores)):
    print(f"PC{i+1:>3d}: {v_ind:.4f} ({v_ind*100:5.1f}%%)  |  acum: {v_acum:.4f} ({v_acum*100:5.1f}%%)  |  eig: {eig:.3f}")

n_kaiser = sum(autovalores > 1)
print(f"\nComponentes con eigenvalue > 1 (Kaiser): {n_kaiser}")
print(f"  Varianza acumulada con {n_kaiser} componentes: {varianza_acumulada[n_kaiser-1]*100:.1f}%")
n_80 = np.argmax(varianza_acumulada >= 0.80) + 1
print(f"Componentes para >= 80%: {n_80}")
print(f"  Interpretacion: se necesitan {n_80}/{X_scaled.shape[1]} componentes "
      f"-> estructura mayoritariamente NO lineal.")

n_comps = len(varianza_individual)
comps = range(1, n_comps + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(comps, autovalores, 'o-', color='steelblue', linewidth=2, markersize=8)
ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Kaiser (eig=1)')
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Autovalor')
ax1.set_title('Scree Plot')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(comps)
ax2.bar(comps, varianza_individual * 100, color='steelblue', alpha=0.7, label='Individual')
ax2.plot(comps, varianza_acumulada * 100, 'o-', color='darkorange', linewidth=2, markersize=6, label='Acumulada')
ax2.axhline(y=80, color='green', linestyle='--', linewidth=1.5, label='Umbral 80%')
ax2.set_xlabel('Componente Principal')
ax2.set_ylabel('Varianza Explicada (%)')
ax2.set_title('Varianza por Componente')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(comps)
plt.tight_layout()
plt.savefig(PLOTS / '3_transformacion' / 'pca_varianza.png', dpi=300)
plt.close(fig)
print("[OK] pca_varianza.png")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_comps)],
    index=X_scaled.columns
)
print("\n=== LOADINGS (primeros 5 PCs) ===")
print(loadings.iloc[:, :5].round(3))

plt.figure(figsize=(10, 14))
sns.heatmap(loadings.iloc[:, :5], annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Peso'})
plt.title('Loadings: Contribucion de Variables a PCs')
plt.tight_layout()
plt.savefig(PLOTS / '3_transformacion' / 'pca_loadings.png', dpi=300)
plt.close()
print("[OK] pca_loadings.png")

X_pca_2d = X_pca[:, :2]
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                      c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)
plt.xlabel(f'PC1 ({varianza_individual[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({varianza_individual[1]*100:.1f}%)')
plt.title('Proyeccion PCA 2D - Heart Disease')
plt.colorbar(scatter, label='Severidad (0-4)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / '3_transformacion' / 'pca_proyeccion_2d.png', dpi=300)
plt.close()
print("[OK] pca_proyeccion_2d.png")
print(f"Varianza explicada en 2D: {varianza_acumulada[1]*100:.1f}%\n")

# ============================================================
# 4. REDUCCION DE DIMENSIONALIDAD: UMAP
# ============================================================
print("=" * 60)
print("  4. UMAP")
print("=" * 60)

umap_ok = False
try:
    import umap
    umap_ok = True
    print("UMAP import OK")
except ImportError as e:
    print(f"ERROR UMAP: {e}")
    print("Instala con: pip install umap-learn")

if umap_ok:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
    X_umap = reducer.fit_transform(X_scaled)
    print(f"Shape UMAP: {X_umap.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                          c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel(f'PC1 ({varianza_individual[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({varianza_individual[1]*100:.1f}%)')
    axes[0].set_title('PCA')
    axes[0].grid(alpha=0.3)
    sc2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1],
                          c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)
    axes[1].set_xlabel('UMAP Dim 1')
    axes[1].set_ylabel('UMAP Dim 2')
    axes[1].set_title('UMAP (n_neighbors=15, min_dist=0.1)')
    axes[1].grid(alpha=0.3)
    fig.colorbar(sc2, ax=axes, label='Severidad (0-4)', shrink=0.6)
    plt.suptitle('PCA vs UMAP - Heart Disease', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS / '3_transformacion' / 'pca_vs_umap.png', dpi=300)
    plt.close(fig)
    print("[OK] pca_vs_umap.png")
else:
    print("UMAP no disponible. Se omiten las secciones que dependen de el.")

# ============================================================
# 5. COMPARACION OBJETIVA PCA vs UMAP
# ============================================================
print("\n" + "=" * 60)
print("  5. COMPARACION PCA vs UMAP")
print("=" * 60)


def jaccard_knn(X_high, X_low, k=10):
    nn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k).fit(X_low)
    neigh_high = nn_high.kneighbors(return_distance=False)
    neigh_low = nn_low.kneighbors(return_distance=False)
    jaccards = []
    for i in range(len(X_high)):
        inter = len(set(neigh_high[i]) & set(neigh_low[i]))
        union = len(set(neigh_high[i]) | set(neigh_low[i]))
        jaccards.append(inter / union if union > 0 else 0)
    return np.mean(jaccards)


def sp_rho_dist(X_high, X_low):
    rho, _ = spearmanr(pdist(X_high), pdist(X_low))
    return rho


def sil_kmeans(X_low, k=2):
    labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X_low)
    return silhouette_score(X_low, labels)


print('Metricas PCA...')
m_pca = {
    'Trustworthiness': trustworthiness(X_scaled.values, X_pca_2d, n_neighbors=5),
    'Continuity':      trustworthiness(X_pca_2d, X_scaled.values, n_neighbors=5),
    'Jaccard k-NN':   jaccard_knn(X_scaled.values, X_pca_2d),
    'Spearman rho':   sp_rho_dist(X_scaled.values, X_pca_2d),
    'Silhouette':     sil_kmeans(X_pca_2d),
}

m_umap = {k: 0 for k in m_pca}
if umap_ok:
    try:
        print('Metricas UMAP...')
        m_umap = {
            'Trustworthiness': trustworthiness(X_scaled.values, X_umap, n_neighbors=5),
            'Continuity':      trustworthiness(X_umap, X_scaled.values, n_neighbors=5),
            'Jaccard k-NN':   jaccard_knn(X_scaled.values, X_umap),
            'Spearman rho':   sp_rho_dist(X_scaled.values, X_umap),
            'Silhouette':     sil_kmeans(X_umap),
        }
    except Exception as e:
        print(f'Error UMAP: {e}')
else:
    print('UMAP no disponible.')

df_m = pd.DataFrame({'PCA': m_pca, 'UMAP': m_umap})
df_m['Delta'] = df_m['UMAP'] - df_m['PCA']
df_m['Ganador'] = df_m[['PCA', 'UMAP']].idxmax(axis=1)

print('\n=== COMPARACION PCA vs UMAP ===')
print(df_m.round(4))
print(f'\nUMAP gana en {sum(df_m["Ganador"] == "UMAP")}/{len(m_pca)} metricas.')

labels = list(m_pca.keys())
pca_vals = [m_pca[k] for k in labels]
umap_vals = [m_umap[k] for k in labels]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width/2, pca_vals, width, label='PCA', color='steelblue', alpha=0.85)
b2 = ax.bar(x + width/2, umap_vals, width, label='UMAP', color='darkorange', alpha=0.85)
ax.set_ylabel('Score')
ax.set_title('PCA vs UMAP: Metricas de Preservacion Topologica', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(pca_vals), max(umap_vals)) * 1.15)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS / '5_evaluacion' / 'pca_vs_umap_metrics.png', dpi=300)
plt.close(fig)
print("[OK] pca_vs_umap_metrics.png")

# ============================================================
# 6. CLUSTERING BASADO EN DENSIDAD: OPTICS
# (reemplaza a DBSCAN: no requiere eps, densidad variable)
# 2 tandas: EX2/EX4 con min_samples=5, EX2.1/EX4.1 con min_samples=15
# ============================================================
print("\n" + "=" * 60)
print("  6. CLUSTERING OPTICS - 2 TANDAS")
print("     Tanda 1: EX2 (PCA) y EX4 (UMAP) con min_samples=5")
print("     Tanda 2: EX2.1 (PCA) y EX4.1 (UMAP) con min_samples=15")
print("=" * 60)


def run_optics(X, nombre, min_samples=5, xi=0.05, max_eps=np.inf):
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps,
                    cluster_method='xi', xi=xi)
    optics.fit(X)
    labels = optics.labels_

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    pct_noise = 100 * n_noise / len(labels)
    mask = labels != -1
    sil = (silhouette_score(X[mask], labels[mask])
           if n_clusters >= 2 and mask.sum() >= 2 else np.nan)

    resumen = {
        'Clusters': n_clusters,
        'Ruido (%)': round(pct_noise, 1),
        'Silhouette': round(float(sil), 3) if np.isfinite(sil) else np.nan,
    }
    if n_clusters > 0:
        tam = pd.Series(labels).value_counts().drop(-1, errors='ignore')
        resumen['Tam min'] = int(tam.min())
        resumen['Tam max'] = int(tam.max())

    print(f'=== OPTICS {nombre} ===')
    print(f'  Hiperparametros: min_samples={min_samples}, xi={xi}')
    print(f'  Clusters detectados: {n_clusters}')
    print(f'  Ruido (label -1): {n_noise} ({pct_noise:.1f}%)')
    print(f'  Silhouette (sin ruido): {resumen.get("Silhouette")}')
    print(f'  Tamanos por cluster: {dict(pd.Series(labels).value_counts().sort_index())}')
    return optics, labels, resumen


def plot_clusters(X, labels, titulo, filename):
    plt.figure(figsize=(10, 7))
    for lab in sorted(set(labels)):
        mask = labels == lab
        if lab == -1:
            plt.scatter(X[mask, 0], X[mask, 1], c='#cccccc', s=15, alpha=0.5, label='Ruido')
        else:
            plt.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.8,
                        edgecolors='black', linewidth=0.3, label=f'Cluster {lab}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[OK] {filename}")


# TANDA 1: min_samples=5
print('\n--- TANDA 1: min_samples=5 ---')

# EX2: PCA + OPTICS
optics_pca, labels_pca, resumen_pca = run_optics(X_pca_2d, 'EX2 (PCA)', min_samples=5)
plot_clusters(
    X_pca_2d, labels_pca,
    f'EX2: OPTICS sobre PCA, min_samples=5 (PC1 {varianza_individual[0]*100:.1f}%, PC2 {varianza_individual[1]*100:.1f}%)',
    'plots/4_mineria/optics_pca.png'
)

# EX4: UMAP + OPTICS
if umap_ok:
    optics_umap, labels_umap, resumen_umap = run_optics(X_umap, 'EX4 (UMAP)', min_samples=5)
    plot_clusters(
        X_umap, labels_umap,
        'EX4: OPTICS sobre UMAP, min_samples=5 (n_neighbors=15, min_dist=0.1)',
        'plots/4_mineria/optics_umap.png'
    )
else:
    optics_umap = labels_umap = resumen_umap = None
    print('UMAP no disponible.')

# TANDA 2: min_samples=15
print('\n--- TANDA 2: min_samples=15 ---')

# EX2.1: PCA + OPTICS
optics_pca2, labels_pca2, resumen_pca2 = run_optics(X_pca_2d, 'EX2.1 (PCA)', min_samples=15)
plot_clusters(
    X_pca_2d, labels_pca2,
    f'EX2.1: OPTICS sobre PCA, min_samples=15 (PC1 {varianza_individual[0]*100:.1f}%, PC2 {varianza_individual[1]*100:.1f}%)',
    'plots/4_mineria/optics_pca_ms15.png'
)

# EX4.1: UMAP + OPTICS
if umap_ok:
    optics_umap2, labels_umap2, resumen_umap2 = run_optics(X_umap, 'EX4.1 (UMAP)', min_samples=15)
    plot_clusters(
        X_umap, labels_umap2,
        'EX4.1: OPTICS sobre UMAP, min_samples=15 (n_neighbors=15, min_dist=0.1)',
        'plots/4_mineria/optics_umap_ms15.png'
    )
else:
    optics_umap2 = labels_umap2 = resumen_umap2 = None

# Reachability plots (2x2: PCA/UMAP x ms=5/ms=15)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for fila, ms, o_pca, o_umap in [
    (0, 'min_samples=5', optics_pca, optics_umap),
    (1, 'min_samples=15', optics_pca2, optics_umap2),
]:
    for ax, optics, nombre in [(axes[fila, 0], o_pca, 'PCA'), (axes[fila, 1], o_umap, 'UMAP')]:
        if optics is None:
            ax.axis('off')
            continue
        ord_ = optics.ordering_
        ax.plot(np.arange(len(ord_)), optics.reachability_[ord_], linewidth=0.7, color='steelblue')
        ax.set_title(f'Reachability Plot - {nombre} ({ms})')
        ax.set_xlabel('Orden de puntos')
        ax.set_ylabel('Distancia de alcance')
        ax.grid(alpha=0.3)
plt.suptitle('OPTICS: Distancia de Alcance (xi=0.05) - Tanda 1 (ms=5) vs Tanda 2 (ms=15)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS / '4_mineria' / 'optics_reachability.png', dpi=300)
plt.close(fig)
print("[OK] optics_reachability.png")

# Comparacion EX2 / EX2.1 / EX4 / EX4.1
filas = {
    'EX2 (PCA, ms=5)': resumen_pca,
    'EX2.1 (PCA, ms=15)': resumen_pca2,
}
if umap_ok:
    filas['EX4 (UMAP, ms=5)'] = resumen_umap
    filas['EX4.1 (UMAP, ms=15)'] = resumen_umap2
df_optics = pd.DataFrame(filas).T

print('\n=== OPTICS: COMPARACION EX2 / EX2.1 / EX4 / EX4.1 ===')
print(df_optics.round(3))

for tanda, cols in [
    ('Tanda 1 (ms=5)', ['EX2 (PCA, ms=5)'] + (['EX4 (UMAP, ms=5)'] if umap_ok else [])),
    ('Tanda 2 (ms=15)', ['EX2.1 (PCA, ms=15)'] + (['EX4.1 (UMAP, ms=15)'] if umap_ok else [])),
]:
    sub = df_optics.loc[cols]
    print(f'\n=> {tanda}: mejor silhouette = {sub["Silhouette"].idxmax()}')

mejor = df_optics['Silhouette'].idxmax()
print(f'\n=> Mejor experimento global por silhouette: {mejor}.')
if 'UMAP' in mejor and umap_ok:
    print('=> Coherente con la conclusion de la reduccion de dimensionalidad.')
else:
    print('=> Pese a las metricas de preservacion, PCA da clusters mas compactos.')

print("\n=== ANALISIS CONSOLIDADO COMPLETO ===")

# ============================================================
# 7. CLUSTERING PARTICIONAL: K-MEANS
# (EX1 = PCA + K-Means, EX3 = UMAP + K-Means)
# ============================================================
print("\n" + "=" * 60)
print("  7. K-MEANS (EX1 = PCA + K-Means, EX3 = UMAP + K-Means)")
print("=" * 60)


def dunn_index(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    dist = squareform(pdist(X))
    max_intra, min_inter = 0.0, np.inf
    for a in unique:
        mask_a = labels == a
        if mask_a.sum() > 1:
            max_intra = max(max_intra, np.max(dist[np.ix_(mask_a, mask_a)]))
        for b in unique:
            if a < b:
                min_inter = min(min_inter, np.min(dist[np.ix_(mask_a, labels == b)]))
    return min_inter / max_intra if max_intra > 0 else np.nan


def grid_kmeans(X, k_range=range(2, 13), seed=42):
    filas, modelos = [], {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=1000)
        km.fit(X)
        modelos[k] = km
        filas.append({
            'k': k,
            'Inercia': km.inertia_,
            'Silhouette': silhouette_score(X, km.labels_),
            'Davies-Bouldin': davies_bouldin_score(X, km.labels_),
            'Calinski-Harabasz': calinski_harabasz_score(X, km.labels_),
            'ARI': adjusted_rand_score(np.asarray(y), km.labels_),
            'NMI': normalized_mutual_info_score(np.asarray(y), km.labels_),
            'Dunn': dunn_index(X, km.labels_),
        })
    return pd.DataFrame(filas).set_index('k'), modelos


def plot_seleccion_k(grid, titulo, filename):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    paneles = [
        (axes[0, 0], 'Inercia', 'o-', 'Codo (menor pendiente)'),
        (axes[0, 1], 'Silhouette', 'o-', 'Maximo'),
        (axes[1, 0], 'Davies-Bouldin', 's-', 'Minimo'),
        (axes[1, 1], 'Calinski-Harabasz', 'D-', 'Maximo'),
    ]
    for ax, col, marker, crit in paneles:
        ax.plot(grid.index, grid[col], marker, color='steelblue', linewidth=2, markersize=7)
        ax.set_xlabel('k')
        ax.set_ylabel('Inercia (WCSS)' if col == 'Inercia' else col)
        ax.set_title(f'{col} por k  (criterio: {crit})')
        ax.grid(alpha=0.3)
        ax.set_xticks(list(grid.index))
    fig.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"[OK] {filename.name}")


# Grid de k para PCA y UMAP
grid_pca, modelos_pca = grid_kmeans(X_pca_2d)
plot_seleccion_k(grid_pca, 'EX1: Seleccion de k - K-Means sobre PCA',
                 PLOTS / '4_mineria' / 'kmeans_codo_pca.png')
print('=== EX1: PCA + K-Means, metricas por k ===')
print(grid_pca.round(3))

if umap_ok:
    grid_umap, modelos_umap = grid_kmeans(X_umap)
    plot_seleccion_k(grid_umap, 'EX3: Seleccion de k - K-Means sobre UMAP',
                     PLOTS / '4_mineria' / 'kmeans_codo_umap.png')
    print('\n=== EX3: UMAP + K-Means, metricas por k ===')
    print(grid_umap.round(3))

# k optimo
k_opt_pca = grid_pca['Silhouette'].idxmax()
print(f'\nk optimo en PCA segun Silhouette: {k_opt_pca}')
print(f'k optimo en PCA segun Calinski-Harabasz: {grid_pca["Calinski-Harabasz"].idxmax()}')
print(f'k optimo en PCA segun Davies-Bouldin: {grid_pca["Davies-Bouldin"].idxmin()}')
if umap_ok:
    k_opt_umap = grid_umap['Silhouette'].idxmax()
    print(f'k optimo en UMAP segun Silhouette: {k_opt_umap}')
    print(f'k optimo en UMAP segun Calinski-Harabasz: {grid_umap["Calinski-Harabasz"].idxmax()}')
    print(f'k optimo en UMAP segun Davies-Bouldin: {grid_umap["Davies-Bouldin"].idxmin()}')
else:
    k_opt_umap = None

# Grafico Silhouette
if umap_ok:
    pares = [(k_opt_pca, modelos_pca[k_opt_pca], X_pca_2d, 'PCA'),
             (k_opt_umap, modelos_umap[k_opt_umap], X_umap, 'UMAP')]
else:
    pares = [(k_opt_pca, modelos_pca[k_opt_pca], X_pca_2d, 'PCA')]

fig, axes = plt.subplots(1, len(pares), figsize=(7 * len(pares), 6), squeeze=False)
axes = axes.flatten()
for ax, (k, km, X, nombre) in zip(axes, pares):
    sil = silhouette_samples(X, km.labels_)
    y_lower = 0
    for i in range(k):
        yi = np.sort(sil[km.labels_ == i])
        ax.fill_betweenx(np.arange(len(yi)), y_lower, y_lower + len(yi), alpha=0.6)
        ax.text(-0.05, y_lower + len(yi) / 2, str(i), fontsize=8)
        y_lower += len(yi)
    ax.axvline(np.mean(sil), color='red', linestyle='--', label=f'Media={np.mean(sil):.3f}')
    ax.set_title(f'Grafico Silhouette {nombre} (k={k})')
    ax.set_xlabel('Silhouette')
    ax.set_ylabel('Cluster')
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / '4_mineria' / 'kmeans_silhouette.png', dpi=300)
plt.close(fig)
print("[OK] kmeans_silhouette.png")

# Clusters finales EX1 y EX3
km_pca = modelos_pca[k_opt_pca]
km_umap = modelos_umap[k_opt_umap] if umap_ok else None

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=km_pca.labels_, cmap='tab10',
           alpha=0.8, edgecolors='black', linewidth=0.3, s=35)
ax.scatter(km_pca.cluster_centers_[:, 0], km_pca.cluster_centers_[:, 1],
           marker='X', c='black', s=140, edgecolors='white', linewidth=1, label='Centroides')
ax.set_xlabel(f'PC1 ({varianza_individual[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({varianza_individual[1]*100:.1f}%)')
ax.set_title(f'EX1: K-Means sobre PCA (k={k_opt_pca})')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / '4_mineria' / 'kmeans_pca.png', dpi=300)
plt.close(fig)
print("[OK] kmeans_pca.png")

if umap_ok:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=km_umap.labels_, cmap='tab10',
               alpha=0.8, edgecolors='black', linewidth=0.3, s=35)
    ax.scatter(km_umap.cluster_centers_[:, 0], km_umap.cluster_centers_[:, 1],
               marker='X', c='black', s=140, edgecolors='white', linewidth=1, label='Centroides')
    ax.set_xlabel('UMAP Dim 1')
    ax.set_ylabel('UMAP Dim 2')
    ax.set_title(f'EX3: K-Means sobre UMAP (k={k_opt_umap})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / '4_mineria' / 'kmeans_umap.png', dpi=300)
    plt.close(fig)
    print("[OK] kmeans_umap.png")

# Interpretacion de centroides (EX1)
print('\n=== INTERPRETACION DE CENTROIDES (EX1: PCA + K-Means) ===')
centroides_pad = np.zeros((k_opt_pca, X_pca.shape[1]))
centroides_pad[:, :2] = km_pca.cluster_centers_
centroides_orig = scaler.inverse_transform(pca.inverse_transform(centroides_pad))
df_centroides = pd.DataFrame(centroides_orig, columns=X_pre_escalado.columns)
df_centroides_z = pd.DataFrame(pca.inverse_transform(centroides_pad),
                               columns=X_scaled.columns)

continuas_orig = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
perfil_cluster = X_pre_escalado.copy()
perfil_cluster['cluster'] = km_pca.labels_
print('--- Perfil medio por cluster (unidades originales) ---')
print(perfil_cluster.groupby('cluster')[continuas_orig].mean().round(2))
print('\n--- Distribucion de severidad (target) por cluster ---')
tabla_sev = pd.crosstab(km_pca.labels_, y)
tabla_sev['total'] = tabla_sev.sum(axis=1)
print(tabla_sev)
print('\n--- Top 5 variables por centroide (|z-score|) ---')
for i in range(k_opt_pca):
    top = df_centroides_z.iloc[i].abs().nlargest(5)
    print(f'  Cluster {i}: ' + ', '.join(f'{v} ({top[v]:.2f})' for v in top.index))

# Metricas adicionales OPTICS + comparacion final
def extra_metrics(X, labels, y_true):
    mask = labels != -1
    res = {}
    if len(np.unique(labels[mask])) >= 2:
        res['Davies-Bouldin'] = davies_bouldin_score(X[mask], labels[mask])
        res['Calinski-Harabasz'] = calinski_harabasz_score(X[mask], labels[mask])
        res['ARI'] = adjusted_rand_score(np.asarray(y_true)[mask], labels[mask])
        res['NMI'] = normalized_mutual_info_score(np.asarray(y_true)[mask], labels[mask])
        res['Dunn'] = dunn_index(X[mask], labels[mask])
    return res


m_optics_pca = extra_metrics(X_pca_2d, labels_pca, y)
m_optics_pca2 = extra_metrics(X_pca_2d, labels_pca2, y)
filas_comp = {
    'K-Means PCA (EX1)': grid_pca.loc[k_opt_pca].to_dict(),
    'OPTICS PCA ms=5 (EX2)': {**resumen_pca, **m_optics_pca},
    'OPTICS PCA ms=15 (EX2.1)': {**resumen_pca2, **m_optics_pca2},
}
if umap_ok:
    m_optics_umap = extra_metrics(X_umap, labels_umap, y)
    m_optics_umap2 = extra_metrics(X_umap, labels_umap2, y)
    filas_comp['K-Means UMAP (EX3)'] = grid_umap.loc[k_opt_umap].to_dict()
    filas_comp['OPTICS UMAP ms=5 (EX4)'] = {**resumen_umap, **m_optics_umap}
    filas_comp['OPTICS UMAP ms=15 (EX4.1)'] = {**resumen_umap2, **m_optics_umap2}

df_comp = pd.DataFrame(filas_comp).T
print('\n=== COMPARACION FINAL: K-Means vs OPTICS (EX1-EX4 + variantes ms=15) ===')
print(df_comp.round(3))

metricas_grafico = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'NMI', 'Dunn']
modelos = list(filas_comp.keys())
n_mod = len(modelos)
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(metricas_grafico))
width = 1 / (n_mod + 1)
colores = plt.cm.tab10(np.linspace(0, 1, n_mod))
valores_brutos = {m: [filas_comp[mod].get(m, np.nan) for mod in modelos] for m in metricas_grafico}
for i, modelo in enumerate(modelos):
    vals = []
    for m in metricas_grafico:
        v = valores_brutos[m][i]
        rng = np.nanmax(valores_brutos[m]) - np.nanmin(valores_brutos[m])
        vals.append((v - np.nanmin(valores_brutos[m])) / rng if rng > 0 and np.isfinite(v) else np.nan)
    ax.bar(x + (i - (n_mod - 1) / 2) * width, vals, width, label=modelo, color=colores[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metricas_grafico)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Valor normalizado (min-max por metrica)')
ax.set_title('Comparacion de metricas: K-Means vs OPTICS (EX1-EX4 + variantes ms=15)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / '5_evaluacion' / 'clustering_metrics.png', dpi=300)
plt.close(fig)
print("[OK] clustering_metrics.png")
