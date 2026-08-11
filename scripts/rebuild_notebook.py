import json
from pathlib import Path

notebook_path = str(Path(__file__).resolve().parent.parent / 'proyecto-add-1-2026.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Quedarse solo con las primeras 22 celdas (trabajo original del usuario)
nb['cells'] = nb['cells'][:22]

# ============================================================
# CELDAS A AGREGAR (desde celda 23 en adelante)
# ============================================================

new_cells = []

# 23 MD: Codificacion categorica
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "cod_md"},
    "source": [
        "## Codificacion de variables categoricas\n",
        "\n",
        "Para que PCA y K-Means interpreten correctamente las variables categoricas, se aplica **One-Hot Encoding**:\n",
        "- **Binarias** (sin cambio): `sex`, `fbs`, `exang`\n",
        "- **Multi-categoria** (one-hot): `cp` (4), `restecg` (3), `slope` (3), `thal` (3)\n",
        "- **Continuas** (sin codificar): `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`\n",
        "\n",
        "> Ver referencias al final del notebook.\n"
    ]
})

# 24 CD: One-Hot Encoding
new_cells.append({
    "cell_type": "code", "metadata": {"id": "cod_cd"},
    "source": [
        "from sklearn.preprocessing import OneHotEncoder\n",
        "\n",
        "binarias = ['sex', 'fbs', 'exang']\n",
        "categoricas_multiclase = ['cp', 'restecg', 'slope', 'thal']\n",
        "continuas = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']\n",
        "\n",
        "encoder = OneHotEncoder(sparse_output=False, drop=None)\n",
        "encoded_array = encoder.fit_transform(X[categoricas_multiclase])\n",
        "encoded_cols = encoder.get_feature_names_out(categoricas_multiclase)\n",
        "df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)\n",
        "\n",
        "print(f'Columnas generadas: {list(df_encoded.columns)}')\n",
        "\n",
        "X_pre_escalado = pd.concat([X[continuas], X[binarias], df_encoded], axis=1)\n",
        "print(f'Shape pre-escalado: {X_pre_escalado.shape}')\n",
        "print(f'Columnas finales: {list(X_pre_escalado.columns)}')"
    ], "execution_count": None, "outputs": []
})

# 25 MD: Escalamiento
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "esc_md"},
    "source": [
        "## Escalamiento / Estandarizacion\n",
        "\n",
        "Se aplica **StandardScaler** (z-score) para que todas las variables tengan media 0 y desviacion estandar 1. "
        "Esto es indispensable antes de PCA y K-Means, ya que ambos dependen de distancias euclidianas y serian "
        "dominados por variables de mayor magnitud (ej. `chol` ~200 vs `oldpeak` ~1).\n"
    ]
})

# 26 CD: StandardScaler
new_cells.append({
    "cell_type": "code", "metadata": {"id": "esc_cd"},
    "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_scaled_array = scaler.fit_transform(X_pre_escalado)\n",
        "X_scaled = pd.DataFrame(X_scaled_array, columns=X_pre_escalado.columns)\n",
        "\n",
        "print('Media por columna (debe ~0):')\n",
        "print(X_scaled.mean().round(4))\n",
        "print('\\nStd por columna (ddof=1, debe ~1):')\n",
        "print(X_scaled.std().round(4))"
    ], "execution_count": None, "outputs": []
})

# 27 MD: PCA theory (brief)
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "pca_md"},
    "source": [
        "# Reduccion de Dimensionalidad: PCA\n",
        "\n",
        "El **PCA** (Pearson, 1901; Hotelling, 1933) transforma variables correlacionadas en componentes "
        "no correlacionados ordenados por varianza explicada. Cada componente es una combinacion lineal "
        "de las originales (autovectores de la matriz de covarianza). Se aplica aqui para condensar las "
        "22 variables y facilitar la visualizacion y el clustering posterior.\n",
        "\n",
        "> Referencias completas al final del notebook.\n"
    ]
})

# 28 CD: PCA fit + varianza
new_cells.append({
    "cell_type": "code", "metadata": {"id": "pca_var_cd"},
    "source": [
        "from sklearn.decomposition import PCA\n",
        "import numpy as np\n",
        "\n",
        "pca = PCA()\n",
        "X_pca = pca.fit_transform(X_scaled)\n",
        "\n",
        "varianza_individual = pca.explained_variance_ratio_\n",
        "varianza_acumulada = np.cumsum(varianza_individual)\n",
        "autovalores = pca.explained_variance_\n",
        "\n",
        "print('=== VARIANZA EXPLICADA POR COMPONENTE ===')\n",
        "for i, (v_ind, v_acum, eig) in enumerate(zip(varianza_individual, varianza_acumulada, autovalores)):\n",
        "    print(f'PC{i+1:>3d}: {v_ind:.4f} ({v_ind*100:5.1f}%%)  |  acum: {v_acum:.4f} ({v_acum*100:5.1f}%%)  |  eig: {eig:.3f}')\n",
        "\n",
        "n_kaiser = sum(autovalores > 1)\n",
        "print(f'\\nComponentes con eigenvalue > 1 (Kaiser): {n_kaiser}')\n",
        "print(f'  Varianza acumulada con {n_kaiser} componentes: {varianza_acumulada[n_kaiser-1]*100:.1f}%')\n",
        "\n",
        "n_80 = np.argmax(varianza_acumulada >= 0.80) + 1\n",
        "print(f'\\nComponentes para >= 80%: {n_80}')\n",
        "print(f'  Interpretacion: se necesitan {n_80}/{X_scaled.shape[1]} componentes -> estructura mayoritariamente NO lineal.')"
    ], "execution_count": None, "outputs": []
})

# 29 CD: Scree plot
new_cells.append({
    "cell_type": "code", "metadata": {"id": "pca_scree_cd"},
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "n_comps = len(varianza_individual)\n",
        "comps = range(1, n_comps + 1)\n",
        "\n",
        "# Scree Plot\n",
        "ax1.plot(comps, autovalores, 'o-', color='steelblue', linewidth=2, markersize=8)\n",
        "ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Kaiser (eig=1)')\n",
        "ax1.set_xlabel('Componente Principal')\n",
        "ax1.set_ylabel('Autovalor')\n",
        "ax1.set_title('Scree Plot')\n",
        "ax1.legend()\n",
        "ax1.grid(alpha=0.3)\n",
        "ax1.set_xticks(comps)\n",
        "\n",
        "# Varianza acumulada\n",
        "ax2.bar(comps, varianza_individual * 100, color='steelblue', alpha=0.7, label='Individual')\n",
        "ax2.plot(comps, varianza_acumulada * 100, 'o-', color='darkorange', linewidth=2, markersize=6, label='Acumulada')\n",
        "ax2.axhline(y=80, color='green', linestyle='--', linewidth=1.5, label='Umbral 80%')\n",
        "ax2.set_xlabel('Componente Principal')\n",
        "ax2.set_ylabel('Varianza Explicada (%)')\n",
        "ax2.set_title('Varianza por Componente')\n",
        "ax2.legend()\n",
        "ax2.grid(alpha=0.3)\n",
        "ax2.set_xticks(comps)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/3_transformacion/pca_varianza.png', dpi=150)\n",
        "plt.show()"
    ], "execution_count": None, "outputs": []
})

# 30 CD: Loadings
new_cells.append({
    "cell_type": "code", "metadata": {"id": "pca_load_cd"},
    "source": [
        "loadings = pd.DataFrame(\n",
        "    pca.components_.T,\n",
        "    columns=[f'PC{i+1}' for i in range(n_comps)],\n",
        "    index=X_scaled.columns\n",
        ")\n",
        "\n",
        "print('=== LOADINGS (primeros 5 PCs) ===')\n",
        "print(loadings.iloc[:, :5].round(3))\n",
        "\n",
        "plt.figure(figsize=(10, 14))\n",
        "sns.heatmap(loadings.iloc[:, :5], annot=True, fmt='.2f', cmap='RdBu_r',\n",
        "            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Peso'})\n",
        "plt.title('Loadings: Contribucion de Variables a PCs')\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/3_transformacion/pca_loadings.png', dpi=150)\n",
        "plt.show()"
    ], "execution_count": None, "outputs": []
})

# 31 CD: PCA 2D proyeccion
new_cells.append({
    "cell_type": "code", "metadata": {"id": "pca_2d_cd"},
    "source": [
        "X_pca_2d = X_pca[:, :2]\n",
        "\n",
        "plt.figure(figsize=(10, 7))\n",
        "scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],\n",
        "                      c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)\n",
        "plt.xlabel(f'PC1 ({varianza_individual[0]*100:.1f}%)')\n",
        "plt.ylabel(f'PC2 ({varianza_individual[1]*100:.1f}%)')\n",
        "plt.title('Proyeccion PCA 2D - Heart Disease')\n",
        "cbar = plt.colorbar(scatter, label='Severidad (0-4)')\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/3_transformacion/pca_proyeccion_2d.png', dpi=150)\n",
        "plt.show()\n",
        "\n",
        "print(f'Varianza explicada en 2D: {varianza_acumulada[1]*100:.1f}%')"
    ], "execution_count": None, "outputs": []
})

# 32 MD: UMAP Teoria + Tabla comparativa
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "umap_md"},
    "source": [
        "# Reduccion de Dimensionalidad: UMAP\n",
        "\n",
        "**UMAP** (McInnes, Healy & Melville, 2018) es una tecnica no lineal que construye un "
        "grafo de k-vecinos en alta dimension y lo optimiza en baja dimension. Se contrasta con PCA "
        "para evaluar si existen estructuras no lineales que PCA no captura.\n",
        "\n",
        "Hiperparametros: `n_neighbors=15`, `min_dist=0.1`, `n_components=2`.\n",
        "\n",
        "## Tabla Comparativa PCA vs UMAP\n",
        "\n",
        "| Criterio | PCA | UMAP |\n",
        "|---|---|---|\n",
        "| **Interpretabilidad** | Alta (autovectores) | Baja (ejes sin significado) |\n",
        "| **Preservacion global** | Alta (maximiza varianza) | Media-Alta |\n",
        "| **Preservacion local** | Baja | Alta (grafo k-NN) |\n",
        "| **Separacion visual** | Moderada | Alta (clusters nitidos) |\n",
        "| **Tiempo computacional** | Bajo (SVD) | Medio (grafos + SGD) |\n",
        "\n",
        "## Experimentos planificados\n",
        "\n",
        "| Exp | Reduccion | Clustering |\n",
        "|---|---|---|\n",
        "| EX1 | PCA | K-Means |\n",
        "| EX2 | PCA | OPTICS |\n",
        "| EX3 | UMAP | K-Means |\n",
        "| EX4 | UMAP | OPTICS |\n"
    ]
})

# 33 CD: Fix numpy + install umap
new_cells.append({
    "cell_type": "code", "metadata": {"id": "umap_install_cd"},
    "source": [
        "# Asegurar numpy compatible con numba/umap\n",
        "import sys, subprocess\n",
        "try:\n",
        "    import numpy as np\n",
        "    if np.__version__ >= '2.5':\n",
        "        print(f'Reinstalando numpy compatible (actual: {np.__version__})...')\n",
        "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy>=2.0,<2.5', '--quiet'])\n",
        "        print('Hecho. Reinicia kernel si numpy ya estaba cargado.')\n",
        "except ImportError:\n",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy>=2.0,<2.5', '--quiet'])\n",
        "\n",
        "!pip install umap-learn --quiet\n",
        "print('umap-learn OK')"
    ], "execution_count": None, "outputs": []
})

# 34 CD: UMAP fit
new_cells.append({
    "cell_type": "code", "metadata": {"id": "umap_fit_cd"},
    "source": [
        "import numpy as np\n",
        "umap_ok = False\n",
        "try:\n",
        "    import umap\n",
        "    print('UMAP import OK')\n",
        "    umap_ok = True\n",
        "except ImportError as e:\n",
        "    print(f'ERROR: {e}')\n",
        "    print('Ejecuta la celda de instalacion anterior y reinicia el kernel.')\n",
        "\n",
        "if umap_ok:\n",
        "    reducer = umap.UMAP(\n",
        "        n_neighbors=15, min_dist=0.1, n_components=2,\n",
        "        random_state=42, n_jobs=1\n",
        "    )\n",
        "    X_umap = reducer.fit_transform(X_scaled)\n",
        "    print(f'Shape UMAP: {X_umap.shape}')\n",
        "else:\n",
        "    print('UMAP no disponible. Se saltan las celdas siguientes.')"
    ], "execution_count": None, "outputs": []
})

# 35 CD: PCA vs UMAP 2D comparacion
new_cells.append({
    "cell_type": "code", "metadata": {"id": "umap_plot_cd"},
    "source": [
        "if umap_ok:\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "    sc1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],\n",
        "                           c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)\n",
        "    axes[0].set_xlabel(f'PC1 ({varianza_individual[0]*100:.1f}%)')\n",
        "    axes[0].set_ylabel(f'PC2 ({varianza_individual[1]*100:.1f}%)')\n",
        "    axes[0].set_title('PCA')\n",
        "    axes[0].grid(alpha=0.3)\n",
        "\n",
        "    sc2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1],\n",
        "                           c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)\n",
        "    axes[1].set_xlabel('UMAP Dim 1')\n",
        "    axes[1].set_ylabel('UMAP Dim 2')\n",
        "    axes[1].set_title('UMAP (n_neighbors=15, min_dist=0.1)')\n",
        "    axes[1].grid(alpha=0.3)\n",
        "\n",
        "    cbar = fig.colorbar(sc2, ax=axes, label='Severidad (0-4)', shrink=0.6)\n",
        "    plt.suptitle('PCA vs UMAP - Heart Disease', fontsize=13, fontweight='bold')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('plots/3_transformacion/pca_vs_umap.png', dpi=150)\n",
        "    plt.show()\n",
        "else:\n",
        "    print('UMAP no disponible.')"
    ], "execution_count": None, "outputs": []
})

# 36 CD: UMAP hiperparametros
new_cells.append({
    "cell_type": "code", "metadata": {"id": "umap_params_cd"},
    "source": [
        "if umap_ok:\n",
        "    fig, axes = plt.subplots(2, 2, figsize=(14, 12))\n",
        "    axes = axes.flatten()\n",
        "\n",
        "    configs = [\n",
        "        (5, 0.1, 'n_neighbors=5 (mas local)'),\n",
        "        (15, 0.1, 'n_neighbors=15 (equilibrado)'),\n",
        "        (30, 0.1, 'n_neighbors=30 (mas global)'),\n",
        "        (15, 0.5, 'min_dist=0.5 (mas disperso)'),\n",
        "    ]\n",
        "\n",
        "    for i, (n_neigh, m_dist, title) in enumerate(configs):\n",
        "        r = umap.UMAP(n_neighbors=n_neigh, min_dist=m_dist, n_components=2, random_state=42)\n",
        "        proj = r.fit_transform(X_scaled)\n",
        "        axes[i].scatter(proj[:, 0], proj[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.3)\n",
        "        axes[i].set_title(title)\n",
        "        axes[i].grid(alpha=0.3)\n",
        "\n",
        "    plt.suptitle('UMAP: Efecto de Hiperparametros', fontsize=14, fontweight='bold')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('plots/3_transformacion/umap_hiperparametros.png', dpi=150)\n",
        "    plt.show()\n",
        "else:\n",
        "    print('UMAP no disponible.')"
    ], "execution_count": None, "outputs": []
})

# 37 MD: PCA vs UMAP - Comparacion objetiva (teoria breve + metricas)
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "metrics_md"},
    "source": [
        "# PCA vs UMAP: Comparacion Objetiva\n",
        "\n",
        "Con 11/22 componentes para alcanzar el 80% de varianza, los datos tienen **baja estructura lineal**. "
        "Para decidir objetivamente si UMAP > PCA, se utilizan metricas de **preservacion topologica** "
        "(Lee & Verleysen, 2009) que miden que tan bien la proyeccion 2D conserva las relaciones de vecindad "
        "del espacio original de 22 dimensiones:\n",
        "\n",
        "| Metrica | Que mide | Mejor |\n",
        "|---|---|---|\n",
        "| **Trustworthiness** | Si vecinos en 2D son vecinos en original | Alto |\n",
        "| **Continuity** | Si vecinos en original son vecinos en 2D | Alto |\n",
        "| **Jaccard k-NN** | Interseccion de conjuntos de k-vecinos (alta vs baja) | Alto |\n",
        "| **Spearman rho** | Correlacion entre matrices de distancia original vs reducida | Alto |\n",
        "| **Silhouette K-Means** | Calidad de clustering sobre la proyeccion | Alto |\n",
        "\n",
        "> Ver referencias al final del notebook.\n"
    ]
})

# 38 CD: Calcular metricas
new_cells.append({
    "cell_type": "code", "metadata": {"id": "metrics_calc_cd"},
    "source": [
        "from sklearn.manifold import trustworthiness\n",
        "from sklearn.neighbors import NearestNeighbors\n",
        "from sklearn.metrics import silhouette_score\n",
        "from sklearn.cluster import KMeans\n",
        "from scipy.stats import spearmanr\n",
        "from scipy.spatial.distance import pdist\n",
        "\n",
        "def jaccard_knn(X_high, X_low, k=10):\n",
        "    nn_high = NearestNeighbors(n_neighbors=k).fit(X_high)\n",
        "    nn_low = NearestNeighbors(n_neighbors=k).fit(X_low)\n",
        "    neigh_high = nn_high.kneighbors(return_distance=False)\n",
        "    neigh_low = nn_low.kneighbors(return_distance=False)\n",
        "    jaccards = []\n",
        "    for i in range(len(X_high)):\n",
        "        inter = len(set(neigh_high[i]) & set(neigh_low[i]))\n",
        "        union = len(set(neigh_high[i]) | set(neigh_low[i]))\n",
        "        jaccards.append(inter / union if union > 0 else 0)\n",
        "    return np.mean(jaccards)\n",
        "\n",
        "def sp_rho_dist(X_high, X_low):\n",
        "    rho, _ = spearmanr(pdist(X_high), pdist(X_low))\n",
        "    return rho\n",
        "\n",
        "def sil_kmeans(X_low, k=2):\n",
        "    labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X_low)\n",
        "    return silhouette_score(X_low, labels)\n",
        "\n",
        "# Calcular para PCA\n",
        "print('Metricas PCA...')\n",
        "m_pca = {\n",
        "    'Trustworthiness': trustworthiness(X_scaled.values, X_pca_2d, n_neighbors=5),\n",
        "    'Continuity':      trustworthiness(X_pca_2d, X_scaled.values, n_neighbors=5),\n",
        "    'Jaccard k-NN':   jaccard_knn(X_scaled.values, X_pca_2d),\n",
        "    'Spearman rho':   sp_rho_dist(X_scaled.values, X_pca_2d),\n",
        "    'Silhouette':     sil_kmeans(X_pca_2d),\n",
        "}\n",
        "\n",
        "# Calcular para UMAP (si disponible)\n",
        "m_umap = {k: 0 for k in m_pca}\n",
        "if umap_ok:\n",
        "    try:\n",
        "        print('Metricas UMAP...')\n",
        "        m_umap = {\n",
        "            'Trustworthiness': trustworthiness(X_scaled.values, X_umap, n_neighbors=5),\n",
        "            'Continuity':      trustworthiness(X_umap, X_scaled.values, n_neighbors=5),\n",
        "            'Jaccard k-NN':   jaccard_knn(X_scaled.values, X_umap),\n",
        "            'Spearman rho':   sp_rho_dist(X_scaled.values, X_umap),\n",
        "            'Silhouette':     sil_kmeans(X_umap),\n",
        "        }\n",
        "    except Exception as e:\n",
        "        print(f'Error UMAP: {e}')\n",
        "else:\n",
        "    print('UMAP no disponible.')\n",
        "\n",
        "# Mostrar tabla\n",
        "df_m = pd.DataFrame({'PCA': m_pca, 'UMAP': m_umap})\n",
        "df_m['Delta'] = df_m['UMAP'] - df_m['PCA']\n",
        "df_m['Ganador'] = df_m[['PCA', 'UMAP']].idxmax(axis=1)\n",
        "\n",
        "print('\\n=== COMPARACION PCA vs UMAP ===')\n",
        "print(df_m.round(4))\n",
        "print(f'\\nUMAP gana en {sum(df_m[\"Ganador\"] == \"UMAP\")}/{len(m_pca)} metricas.')\n",
        "\n",
        "if sum(df_m['Ganador'] == 'UMAP') >= 3:\n",
        "    print('=> CONCLUSION: UMAP es objetivamente superior a PCA para este dataset.')\n",
        "else:\n",
        "    print('=> PCA mantiene ventaja competitiva. Ambas proyecciones son utiles.')"
    ], "execution_count": None, "outputs": []
})

# 39 CD: Grafico de barras metricas
new_cells.append({
    "cell_type": "code", "metadata": {"id": "metrics_bar_cd"},
    "source": [
        "labels = list(m_pca.keys())\n",
        "pca_vals = [m_pca[k] for k in labels]\n",
        "umap_vals = [m_umap[k] for k in labels]\n",
        "\n",
        "x = np.arange(len(labels))\n",
        "width = 0.35\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "b1 = ax.bar(x - width/2, pca_vals, width, label='PCA', color='steelblue', alpha=0.85)\n",
        "b2 = ax.bar(x + width/2, umap_vals, width, label='UMAP', color='darkorange', alpha=0.85)\n",
        "\n",
        "ax.set_ylabel('Score')\n",
        "ax.set_title('PCA vs UMAP: Metricas de Preservacion Topologica', fontsize=13, fontweight='bold')\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(labels, fontsize=10)\n",
        "ax.legend(fontsize=11)\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "ax.set_ylim(0, max(max(pca_vals), max(umap_vals)) * 1.15)\n",
        "\n",
        "for bar in b1:\n",
        "    h = bar.get_height()\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9)\n",
        "for bar in b2:\n",
        "    h = bar.get_height()\n",
        "    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/5_evaluacion/pca_vs_umap_metrics.png', dpi=150)\n",
        "plt.show()"
    ], "execution_count": None, "outputs": []
})

# 40 MD: Interpretacion de metricas
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "metrics_interp_md"},
    "source": [
        "## Interpretacion de Metricas\n",
        "\n",
        "- **Trustworthiness**: mas alto en UMAP => vecinos en 2D reflejan autenticamente el espacio original.\n",
        "- **Continuity**: mas alto en UMAP => la estructura topologica original no se rompe en la proyeccion.\n",
        "- **Jaccard k-NN**: interseccion de conjuntos de vecinos. Fidelidad topologica directa.\n",
        "- **Spearman rho**: correlacion entre todas las distancias. rho=1 seria proyeccion perfecta.\n",
        "- **Silhouette**: utilidad practica para clustering (etapa siguiente del proyecto).\n",
        "\n",
        "**Criterio de decision**: si UMAP gana en 3+ metricas, se considera objetivamente superior.\n",
        "Este resultado se alinea con la teoria: cuando se necesitan >50% de los PCs para explicar el 80% "
        "de la varianza, la estructura es predominantemente no lineal, terreno donde UMAP domina.\n"
    ]
})

# 41 MD: Clustering OPTICS (teoria)
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "optics_md"},
    "source": [
        "# Clustering basado en densidad: OPTICS\n",
        "\n",
        "**OPTICS** (Ankerst, Breunig, Kriegel & Sander, 1999) es un algoritmo de clustering por densidad "
        "que ordena los puntos segun una **distancia de alcance (reachability)** e identifica clusters "
        "como valles en el reachability plot. A diferencia de **DBSCAN**, no requiere el parametro `eps` "
        "(radio global), por lo que maneja clusters de **densidad variable** sin re-tunear hiperparametros.\n",
        "\n",
        "## Cambio respecto a la planificacion\n",
        "\n",
        "En la planificacion original se contemplaba DBSCAN (EX2 y EX4). Se sustituye por OPTICS porque:\n",
        "\n",
        "- DBSCAN exige fijar `eps` a mano (k-distance plot) y es muy sensible a ese valor.\n",
        "- OPTICS solo necesita `min_samples` y extrae los clusters con `xi` (umbral de pendiente del reachability plot).\n",
        "- OPTICS es mas robusto cuando las densidades de los grupos difieren.\n",
        "\n",
        "| Parametro | DBSCAN | OPTICS |\n",
        "|---|---|---|\n",
        "| `eps` | Requerido (sensible) | No aplica (`max_eps=inf`) |\n",
        "| `min_samples` | Requerido | Requerido |\n",
        "| Extraccion de clusters | Directa | `xi` (`cluster_method='xi'`) |\n",
        "| Densidad variable | No | Si |\n",
        "\n",
        "Los experimentos quedan como **EX2 = PCA + OPTICS** y **EX4 = UMAP + OPTICS**.\n"
    ]
})

# 42 CD: Helpers para OPTICS
new_cells.append({
    "cell_type": "code", "metadata": {"id": "optics_helpers_cd"},
    "source": [
        "from sklearn.cluster import OPTICS\n",
        "from sklearn.metrics import silhouette_score\n",
        "\n",
        "def run_optics(X, nombre, min_samples=5, xi=0.05, max_eps=np.inf):\n",
        "    optics = OPTICS(min_samples=min_samples, max_eps=max_eps,\n",
        "                    cluster_method='xi', xi=xi)\n",
        "    optics.fit(X)\n",
        "    labels = optics.labels_\n",
        "\n",
        "    n_clusters = len(set(labels) - {-1})\n",
        "    n_noise = int((labels == -1).sum())\n",
        "    pct_noise = 100 * n_noise / len(labels)\n",
        "    mask = labels != -1\n",
        "    sil = (silhouette_score(X[mask], labels[mask])\n",
        "           if n_clusters >= 2 and mask.sum() >= 2 else np.nan)\n",
        "\n",
        "    resumen = {\n",
        "        'Clusters': n_clusters,\n",
        "        'Ruido (%)': round(pct_noise, 1),\n",
        "        'Silhouette': round(float(sil), 3) if np.isfinite(sil) else np.nan,\n",
        "    }\n",
        "    if n_clusters > 0:\n",
        "        tam = pd.Series(labels).value_counts().drop(-1, errors='ignore')\n",
        "        resumen['Tam min'] = int(tam.min())\n",
        "        resumen['Tam max'] = int(tam.max())\n",
        "\n",
        "    print(f'=== OPTICS {nombre} ===')\n",
        "    print(f'  Hiperparametros: min_samples={min_samples}, xi={xi}')\n",
        "    print(f'  Clusters detectados: {n_clusters}')\n",
        "    print(f'  Ruido (label -1): {n_noise} ({pct_noise:.1f}%)')\n",
        "    print(f'  Silhouette (sin ruido): {resumen.get(\"Silhouette\")}')\n",
        "    print(f'  Tamanos por cluster: {dict(pd.Series(labels).value_counts().sort_index())}')\n",
        "    return optics, labels, resumen\n",
        "\n",
        "def plot_clusters(X, labels, titulo, filename):\n",
        "    plt.figure(figsize=(10, 7))\n",
        "    for lab in sorted(set(labels)):\n",
        "        mask = labels == lab\n",
        "        if lab == -1:\n",
        "            plt.scatter(X[mask, 0], X[mask, 1], c='#cccccc', s=15, alpha=0.5, label='Ruido')\n",
        "        else:\n",
        "            plt.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.8,\n",
        "                        edgecolors='black', linewidth=0.3, label=f'Cluster {lab}')\n",
        "    plt.xlabel('Dim 1')\n",
        "    plt.ylabel('Dim 2')\n",
        "    plt.title(titulo)\n",
        "    plt.legend()\n",
        "    plt.grid(alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(filename, dpi=150)\n",
        "    plt.show()"
    ], "execution_count": None, "outputs": []
})

# 43 CD: EX2 PCA + OPTICS
new_cells.append({
    "cell_type": "code", "metadata": {"id": "optics_pca_cd"},
    "source": [
        "optics_pca, labels_pca, resumen_pca = run_optics(X_pca_2d, 'PCA')\n",
        "plot_clusters(\n",
        "    X_pca_2d, labels_pca,\n",
        "    f'OPTICS sobre PCA (PC1 {varianza_individual[0]*100:.1f}%, PC2 {varianza_individual[1]*100:.1f}%)',\n",
        "    'plots/4_mineria/optics_pca.png'\n",
        ")"
    ], "execution_count": None, "outputs": []
})

# 44 CD: EX4 UMAP + OPTICS
new_cells.append({
    "cell_type": "code", "metadata": {"id": "optics_umap_cd"},
    "source": [
        "if umap_ok:\n",
        "    optics_umap, labels_umap, resumen_umap = run_optics(X_umap, 'UMAP')\n",
        "    plot_clusters(\n",
        "        X_umap, labels_umap,\n",
        "        'OPTICS sobre UMAP (n_neighbors=15, min_dist=0.1)',\n",
        "        'plots/4_mineria/optics_umap.png'\n",
        "    )\n",
        "else:\n",
        "    print('UMAP no disponible.')"
    ], "execution_count": None, "outputs": []
})

# 45 CD: Reachability plots
new_cells.append({
    "cell_type": "code", "metadata": {"id": "optics_reach_cd"},
    "source": [
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5)) if umap_ok else (None, None)\n",
        "if umap_ok:\n",
        "    for ax, optics, nombre in zip(axes, [optics_pca, optics_umap], ['PCA', 'UMAP']):\n",
        "        ord_ = optics.ordering_\n",
        "        ax.plot(np.arange(len(ord_)), optics.reachability_[ord_], linewidth=0.7, color='steelblue')\n",
        "        ax.set_title(f'Reachability Plot - {nombre}')\n",
        "        ax.set_xlabel('Orden de puntos')\n",
        "        ax.set_ylabel('Distancia de alcance')\n",
        "        ax.grid(alpha=0.3)\n",
        "    plt.suptitle('OPTICS: Distancia de Alcance (xi=0.05)', fontsize=13, fontweight='bold')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('plots/4_mineria/optics_reachability.png', dpi=150)\n",
        "    plt.show()\n",
        "else:\n",
        "    ord_ = optics_pca.ordering_\n",
        "    plt.figure(figsize=(8, 5))\n",
        "    plt.plot(np.arange(len(ord_)), optics_pca.reachability_[ord_], linewidth=0.7, color='steelblue')\n",
        "    plt.title('Reachability Plot - PCA')\n",
        "    plt.xlabel('Orden de puntos')\n",
        "    plt.ylabel('Distancia de alcance')\n",
        "    plt.grid(alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('plots/4_mineria/optics_reachability.png', dpi=150)\n",
        "    plt.show()"
    ], "execution_count": None, "outputs": []
})

# 46 CD: Comparacion PCA vs UMAP con OPTICS
new_cells.append({
    "cell_type": "code", "metadata": {"id": "optics_cmp_cd"},
    "source": [
        "filas = {'PCA': resumen_pca}\n",
        "if umap_ok:\n",
        "    filas['UMAP'] = resumen_umap\n",
        "df_optics = pd.DataFrame(filas).T\n",
        "df_optics['Ganador'] = df_optics['Silhouette'].idxmax()\n",
        "\n",
        "print('=== OPTICS: COMPARACION PCA vs UMAP ===')\n",
        "print(df_optics.round(3))\n",
        "\n",
        "mejor = df_optics['Ganador'].iloc[0]\n",
        "print(f'\\n=> La proyeccion con mejor silhouette en OPTICS es {mejor}.')\n",
        "if mejor == 'UMAP' and umap_ok:\n",
        "    print('=> Coherente con la conclusion de la reduccion de dimensionalidad.')\n",
        "else:\n",
        "    print('=> Pese a las metricas de preservacion, PCA da clusters mas compactos.')"
    ], "execution_count": None, "outputs": []
})

# 47 MD: Interpretacion del clustering
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "optics_interp_md"},
    "source": [
        "## Interpretacion del clustering\n",
        "\n",
        "- **Clusters detectados**: cuantos grupos de pacientes encontro OPTICS en cada proyeccion.\n",
        "- **Ruido (%)**: pacientes que OPTICS no asigna a ningun cluster (etiqueta `-1`). Un ruido alto sugiere ausencia de estructura de densidad clara.\n",
        "- **Silhouette (sin ruido)**: compactitud y separacion de los clusters; se calcula excluyendo los puntos de ruido porque la metrica no admite la etiqueta `-1`.\n",
        "- **Tamanos**: `Tam min`/`Tam max` informan si hay grupos muy desiguales (perfiles clinicos raros vs. mayoritarios).\n",
        "\n",
        "> Completar la interpretacion con los valores que entreguen las celdas de ejecucion.\n"
    ]
})

# 49 MD: Clustering particional K-Means
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "kmeans_md"},
    "source": [
        "# Clustering particional: K-Means\n",
        "\n",
        "**K-Means** (Lloyd, 1982; MacQueen, 1967; Hartigan & Wong, 1979) es un algoritmo de particionamiento "
        "que minimiza iterativamente la **inercia** (suma de distancias euclidianas al cuadrado de cada punto a su "
        "centroide). A diferencia de OPTICS, exige fijar el numero de clusters `k` a priori. Aqui se selecciona con "
        "dos criterios complementarios:\n",
        "\n",
        "- **Metodo del codo**: el `k` a partir del cual la inercia deja de decrecer significativamente.\n",
        "- **Silhouette Score** (Rousseeuw, 1987): el `k` que maximiza la cohesion y separacion media de los clusters.\n",
        "\n",
        "## Experimentos del plan experimental\n",
        "\n",
        "| Exp | Reduccion | Clustering | Se implementa en |\n",
        "|---|---|---|---|\n",
        "| EX1 | PCA | K-Means | seleccion de `k` y clusters sobre `X_pca_2d` |\n",
        "| EX3 | UMAP | K-Means | seleccion de `k` y clusters sobre `X_umap` |\n",
        "\n",
        "## Metricas de evaluacion\n",
        "\n",
        "| Metrica | Sentido optimo | Que mide |\n",
        "|---|---|---|\n",
        "| **Silhouette** (Rousseeuw, 1987) | Mayor | Cohesion y separacion media de los clusters |\n",
        "| **Davies-Bouldin** (1979) | Menor | Razon dispersion intra-cluster vs. distancia entre centroides |\n",
        "| **Calinski-Harabasz** (1974) | Mayor | Razon dispersion entre grupos / dispersion intra-grupo |\n",
        "| **ARI** (Hubert & Arabie, 1985) | Mayor | Concordancia con el ground truth `y` (severidad 0-4), ajustada al azar |\n",
        "| **NMI** (Vinh et al., 2010) | Mayor | Informacion mutua normalizada con el ground truth |\n",
        "| **Dunn** (1974) | Mayor | Minima distancia entre clusters / maximo diametro interno |\n",
        "\n",
        "> Referencias completas al final del notebook.\n"
    ]
})

# 50 CD: Helpers K-Means (dunn, grid, plots de seleccion de k)
new_cells.append({
    "cell_type": "code", "metadata": {"id": "kmeans_helpers"},
    "source": [
        "from sklearn.cluster import KMeans\n",
        "from sklearn.metrics import (silhouette_score, silhouette_samples,\n",
        "                             davies_bouldin_score, calinski_harabasz_score,\n",
        "                             adjusted_rand_score, normalized_mutual_info_score)\n",
        "from scipy.spatial.distance import pdist, squareform\n",
        "\n",
        "\n",
        "def dunn_index(X, labels):\n",
        "    \"\"\"Dunn index: min distancia entre clusters / max diametro interno.\"\"\"\n",
        "    unique = np.unique(labels)\n",
        "    if len(unique) < 2:\n",
        "        return np.nan\n",
        "    dist = squareform(pdist(X))\n",
        "    max_intra, min_inter = 0.0, np.inf\n",
        "    for a in unique:\n",
        "        mask_a = labels == a\n",
        "        if mask_a.sum() > 1:\n",
        "            max_intra = max(max_intra, np.max(dist[np.ix_(mask_a, mask_a)]))\n",
        "        for b in unique:\n",
        "            if a < b:\n",
        "                min_inter = min(min_inter, np.min(dist[np.ix_(mask_a, labels == b)]))\n",
        "    return min_inter / max_intra if max_intra > 0 else np.nan\n",
        "\n",
        "\n",
        "def grid_kmeans(X, k_range=range(2, 13), seed=42):\n",
        "    filas, modelos = [], {}\n",
        "    for k in k_range:\n",
        "        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=1000)\n",
        "        km.fit(X)\n",
        "        modelos[k] = km\n",
        "        filas.append({\n",
        "            'k': k,\n",
        "            'Inercia': km.inertia_,\n",
        "            'Silhouette': silhouette_score(X, km.labels_),\n",
        "            'Davies-Bouldin': davies_bouldin_score(X, km.labels_),\n",
        "            'Calinski-Harabasz': calinski_harabasz_score(X, km.labels_),\n",
        "            'ARI': adjusted_rand_score(np.asarray(y), km.labels_),\n",
        "            'NMI': normalized_mutual_info_score(np.asarray(y), km.labels_),\n",
        "            'Dunn': dunn_index(X, km.labels_),\n",
        "        })\n",
        "    return pd.DataFrame(filas).set_index('k'), modelos\n",
        "\n",
        "\n",
        "def plot_seleccion_k(grid, titulo, filename):\n",
        "    fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n",
        "    paneles = [\n",
        "        (axes[0, 0], 'Inercia', 'o-', 'Codo (menor pendiente)'),\n",
        "        (axes[0, 1], 'Silhouette', 'o-', 'Maximo'),\n",
        "        (axes[1, 0], 'Davies-Bouldin', 's-', 'Minimo'),\n",
        "        (axes[1, 1], 'Calinski-Harabasz', 'D-', 'Maximo'),\n",
        "    ]\n",
        "    for ax, col, marker, crit in paneles:\n",
        "        ax.plot(grid.index, grid[col], marker, color='steelblue', linewidth=2, markersize=7)\n",
        "        ax.set_xlabel('k')\n",
        "        ax.set_ylabel('Inercia (WCSS)' if col == 'Inercia' else col)\n",
        "        ax.set_title(f'{col} por k  (criterio: {crit})')\n",
        "        ax.grid(alpha=0.3)\n",
        "        ax.set_xticks(list(grid.index))\n",
        "    fig.suptitle(titulo, fontsize=13, fontweight='bold')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(filename, dpi=150)\n",
        "    plt.show()\n"
    ], "execution_count": None, "outputs": []
})

# 51 CD: Grid de k para PCA y UMAP (metodo del codo + metricas)
new_cells.append({
    "cell_type": "code", "metadata": {"id": "kmeans_grid"},
    "source": [
        "grid_pca, modelos_pca = grid_kmeans(X_pca_2d)\n",
        "plot_seleccion_k(grid_pca, 'EX1: Seleccion de k - K-Means sobre PCA', 'plots/4_mineria/kmeans_codo_pca.png')\n",
        "print('=== EX1: PCA + K-Means, metricas por k ===')\n",
        "print(grid_pca.round(3))\n",
        "\n",
        "if umap_ok:\n",
        "    grid_umap, modelos_umap = grid_kmeans(X_umap)\n",
        "    plot_seleccion_k(grid_umap, 'EX3: Seleccion de k - K-Means sobre UMAP', 'plots/4_mineria/kmeans_codo_umap.png')\n",
        "    print('\\n=== EX3: UMAP + K-Means, metricas por k ===')\n",
        "    print(grid_umap.round(3))\n",
        "else:\n",
        "    print('UMAP no disponible: EX3 omitido.')\n"
    ], "execution_count": None, "outputs": []
})

# 52 CD: k optimo y grafico Silhouette
new_cells.append({
    "cell_type": "code", "metadata": {"id": "kmeans_kopt"},
    "source": [
        "k_opt_pca = grid_pca['Silhouette'].idxmax()\n",
        "print(f'k optimo en PCA segun Silhouette: {k_opt_pca}')\n",
        "print(f'k optimo en PCA segun Calinski-Harabasz: {grid_pca[\"Calinski-Harabasz\"].idxmax()}')\n",
        "print(f'k optimo en PCA segun Davies-Bouldin: {grid_pca[\"Davies-Bouldin\"].idxmin()}')\n",
        "print('El metodo del codo se aprecia en el panel de Inercia del grafico anterior.')\n",
        "\n",
        "if umap_ok:\n",
        "    k_opt_umap = grid_umap['Silhouette'].idxmax()\n",
        "    print(f'k optimo en UMAP segun Silhouette: {k_opt_umap}')\n",
        "    print(f'k optimo en UMAP segun Calinski-Harabasz: {grid_umap[\"Calinski-Harabasz\"].idxmax()}')\n",
        "    print(f'k optimo en UMAP segun Davies-Bouldin: {grid_umap[\"Davies-Bouldin\"].idxmin()}')\n",
        "    pares = [(k_opt_pca, modelos_pca[k_opt_pca], X_pca_2d, 'PCA'),\n",
        "             (k_opt_umap, modelos_umap[k_opt_umap], X_umap, 'UMAP')]\n",
        "else:\n",
        "    k_opt_umap = None\n",
        "    print('UMAP no disponible: EX3 omitido.')\n",
        "    pares = [(k_opt_pca, modelos_pca[k_opt_pca], X_pca_2d, 'PCA')]\n",
        "\n",
        "fig, axes = plt.subplots(1, len(pares), figsize=(7 * len(pares), 6), squeeze=False)\n",
        "axes = axes.flatten()\n",
        "for ax, (k, km, X, nombre) in zip(axes, pares):\n",
        "    sil = silhouette_samples(X, km.labels_)\n",
        "    y_lower = 0\n",
        "    for i in range(k):\n",
        "        yi = np.sort(sil[km.labels_ == i])\n",
        "        ax.fill_betweenx(np.arange(len(yi)), y_lower, y_lower + len(yi), alpha=0.6)\n",
        "        ax.text(-0.05, y_lower + len(yi) / 2, str(i), fontsize=8)\n",
        "        y_lower += len(yi)\n",
        "    ax.axvline(np.mean(sil), color='red', linestyle='--', label=f'Media={np.mean(sil):.3f}')\n",
        "    ax.set_title(f'Grafico Silhouette {nombre} (k={k})')\n",
        "    ax.set_xlabel('Silhouette')\n",
        "    ax.set_ylabel('Cluster')\n",
        "    ax.grid(alpha=0.3)\n",
        "    ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/4_mineria/kmeans_silhouette.png', dpi=150)\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []
})

# 53 CD: EX1 y EX3 - clusters finales
new_cells.append({
    "cell_type": "code", "metadata": {"id": "kmeans_clusters"},
    "source": [
        "km_pca = modelos_pca[k_opt_pca]\n",
        "km_umap = modelos_umap[k_opt_umap] if umap_ok else None\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 7))\n",
        "ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=km_pca.labels_, cmap='tab10',\n",
        "           alpha=0.8, edgecolors='black', linewidth=0.3, s=35)\n",
        "ax.scatter(km_pca.cluster_centers_[:, 0], km_pca.cluster_centers_[:, 1],\n",
        "           marker='X', c='black', s=140, edgecolors='white', linewidth=1, label='Centroides')\n",
        "ax.set_xlabel(f'PC1 ({varianza_individual[0]*100:.1f}%)')\n",
        "ax.set_ylabel(f'PC2 ({varianza_individual[1]*100:.1f}%)')\n",
        "ax.set_title(f'EX1: K-Means sobre PCA (k={k_opt_pca})')\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/4_mineria/kmeans_pca.png', dpi=150)\n",
        "plt.show()\n",
        "\n",
        "if umap_ok:\n",
        "    fig, ax = plt.subplots(figsize=(10, 7))\n",
        "    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=km_umap.labels_, cmap='tab10',\n",
        "               alpha=0.8, edgecolors='black', linewidth=0.3, s=35)\n",
        "    ax.scatter(km_umap.cluster_centers_[:, 0], km_umap.cluster_centers_[:, 1],\n",
        "               marker='X', c='black', s=140, edgecolors='white', linewidth=1, label='Centroides')\n",
        "    ax.set_xlabel('UMAP Dim 1')\n",
        "    ax.set_ylabel('UMAP Dim 2')\n",
        "    ax.set_title(f'EX3: K-Means sobre UMAP (k={k_opt_umap})')\n",
        "    ax.legend()\n",
        "    ax.grid(alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('plots/4_mineria/kmeans_umap.png', dpi=150)\n",
        "    plt.show()\n"
    ], "execution_count": None, "outputs": []
})

# 54 CD: Interpretacion de centroides (EX1)
new_cells.append({
    "cell_type": "code", "metadata": {"id": "kmeans_centroides"},
    "source": [
        "print('=== INTERPRETACION DE CENTROIDES (EX1: PCA + K-Means) ===')\n",
        "print('Los centroides se reconstruyen al espacio original (inversa de PCA + inversa del scaler).\\n')\n",
        "\n",
        "centroides_pad = np.zeros((k_opt_pca, X_pca.shape[1]))\n",
        "centroides_pad[:, :2] = km_pca.cluster_centers_\n",
        "centroides_orig = scaler.inverse_transform(pca.inverse_transform(centroides_pad))\n",
        "df_centroides = pd.DataFrame(centroides_orig, columns=X_pre_escalado.columns)\n",
        "df_centroides_z = pd.DataFrame(pca.inverse_transform(centroides_pad),\n",
        "                               columns=X_scaled.columns)\n",
        "\n",
        "continuas_orig = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']\n",
        "perfil_cluster = X_pre_escalado.copy()\n",
        "perfil_cluster['cluster'] = km_pca.labels_\n",
        "\n",
        "print('--- Perfil medio por cluster (variables continuas, unidades originales) ---')\n",
        "print(perfil_cluster.groupby('cluster')[continuas_orig].mean().round(2))\n",
        "\n",
        "print('\\n--- Distribucion de severidad (target) por cluster ---')\n",
        "tabla_sev = pd.crosstab(km_pca.labels_, y)\n",
        "tabla_sev['total'] = tabla_sev.sum(axis=1)\n",
        "print(tabla_sev)\n",
        "\n",
        "print('\\n--- Centroides reconstruidos (unidades originales) ---')\n",
        "print(df_centroides.round(2))\n",
        "\n",
        "print('\\n--- Top 5 variables por centroide (|z-score|, mayor contribucion) ---')\n",
        "for i in range(k_opt_pca):\n",
        "    top = df_centroides_z.iloc[i].abs().nlargest(5)\n",
        "    print(f'  Cluster {i}: ' + ', '.join(f'{v} ({top[v]:.2f})' for v in top.index))\n"
    ], "execution_count": None, "outputs": []
})

# 55 CD: Metricas adicionales para OPTICS + comparacion final
new_cells.append({
    "cell_type": "code", "metadata": {"id": "kmeans_compare"},
    "source": [
        "def extra_metrics(X, labels, y_true):\n",
        "    mask = labels != -1\n",
        "    res = {}\n",
        "    if len(np.unique(labels[mask])) >= 2:\n",
        "        res['Davies-Bouldin'] = davies_bouldin_score(X[mask], labels[mask])\n",
        "        res['Calinski-Harabasz'] = calinski_harabasz_score(X[mask], labels[mask])\n",
        "        res['ARI'] = adjusted_rand_score(np.asarray(y_true)[mask], labels[mask])\n",
        "        res['NMI'] = normalized_mutual_info_score(np.asarray(y_true)[mask], labels[mask])\n",
        "        res['Dunn'] = dunn_index(X[mask], labels[mask])\n",
        "    return res\n",
        "\n",
        "m_optics_pca = extra_metrics(X_pca_2d, labels_pca, y)\n",
        "print('=== OPTICS PCA: metricas adicionales (sin ruido) ===')\n",
        "print(pd.Series(m_optics_pca).round(3))\n",
        "\n",
        "filas_comp = {\n",
        "    'K-Means PCA': grid_pca.loc[k_opt_pca].to_dict(),\n",
        "    'OPTICS PCA': {**resumen_pca, **m_optics_pca},\n",
        "}\n",
        "if umap_ok:\n",
        "    m_optics_umap = extra_metrics(X_umap, labels_umap, y)\n",
        "    print('\\n=== OPTICS UMAP: metricas adicionales (sin ruido) ===')\n",
        "    print(pd.Series(m_optics_umap).round(3))\n",
        "    filas_comp['K-Means UMAP'] = grid_umap.loc[k_opt_umap].to_dict()\n",
        "    filas_comp['OPTICS UMAP'] = {**resumen_umap, **m_optics_umap}\n",
        "\n",
        "df_comp = pd.DataFrame(filas_comp).T\n",
        "print('\\n=== COMPARACION FINAL: K-Means vs OPTICS (PCA vs UMAP) ===')\n",
        "print(df_comp.round(3))\n",
        "\n",
        "metricas_grafico = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'NMI', 'Dunn']\n",
        "fig, ax = plt.subplots(figsize=(13, 6))\n",
        "x = np.arange(len(metricas_grafico))\n",
        "width = 0.2\n",
        "colores = ['steelblue', 'darkorange', 'mediumseagreen', 'indianred']\n",
        "valores_brutos = {m: [filas_comp[mod].get(m, np.nan) for mod in filas_comp] for m in metricas_grafico}\n",
        "for i, modelo in enumerate(filas_comp):\n",
        "    vals = []\n",
        "    for m in metricas_grafico:\n",
        "        v = valores_brutos[m][i]\n",
        "        rng = np.nanmax(valores_brutos[m]) - np.nanmin(valores_brutos[m])\n",
        "        vals.append((v - np.nanmin(valores_brutos[m])) / rng if rng > 0 and np.isfinite(v) else np.nan)\n",
        "    ax.bar(x + (i - 1.5) * width, vals, width, label=modelo, color=colores[i], alpha=0.85)\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(metricas_grafico)\n",
        "ax.set_ylim(0, 1.15)\n",
        "ax.set_ylabel('Valor normalizado (min-max por metrica)')\n",
        "ax.set_title('Comparacion de metricas: K-Means vs OPTICS (EX1-EX4)', fontsize=13, fontweight='bold')\n",
        "ax.legend(fontsize=9)\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig('plots/5_evaluacion/clustering_metrics.png', dpi=150)\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []
})

# 56 MD: Interpretacion del clustering K-Means (a completar)
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "kmeans_interp_md"},
    "source": [
        "## Interpretacion del clustering K-Means\n",
        "\n",
        "Completar con los valores reales de la ejecucion:\n",
        "\n",
        "- **Seleccion de k**: el metodo del codo indica [completar], mientras que Silhouette / Calinski-Harabasz / Davies-Bouldin sugieren [completar].\n",
        "- **EX1 (PCA + K-Means)**: perfil clinico de cada cluster (edad, colesterol, presion, thalach, oldpeak) y relacion con la severidad del target.\n",
        "- **EX3 (UMAP + K-Means)**: comparacion con EX1; UMAP deberia producir clusters mas compactos y separados.\n",
        "- **Comparacion K-Means vs OPTICS**: K-Means asume clusters esfericos y de tamano similar; OPTICS detecta densidades variables y aísla ruido. Analizar Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI y NMI.\n"
    ]
})

# 48 MD: Referencias (seccion final)
new_cells.append({
    "cell_type": "markdown", "metadata": {"id": "refs_md"},
    "source": [
        "# Referencias\n",
        "\n",
        "- Ankerst, M., Breunig, M. M., Kriegel, H.-P., & Sander, J. (1999). OPTICS: Ordering points to identify the clustering structure. *ACM SIGMOD Record*, 28(2), 49-60.\n",
        "- Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1), 1-27.\n",
        "- Cattell, R. B. (1966). The scree test for the number of factors. "
        "*Multivariate Behavioral Research*, 1(2), 245-276.\n",
        "- Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. "
        "*IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224-227.\n",
        "- Detrano, R., Janosi, A., et al. (1989). International application of a new probability algorithm "
        "for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.\n",
        "- Dunn, J. C. (1974). Well-separated clusters and optimal fuzzy partitions. *Journal of Cybernetics*, 4(1), 95-104.\n",
        "- Hartigan, J. A., & Wong, M. A. (1979). Algorithm AS 136: A k-means clustering algorithm. "
        "*Applied Statistics*, 28(1), 100-108.\n",
        "- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. "
        "*Journal of Educational Psychology*, 24(6), 417-441.\n",
        "- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.\n",
        "- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.\n",
        "- Kaiser, H. F. (1960). The application of electronic computers to factor analysis. "
        "*Educational and Psychological Measurement*, 20(1), 141-151.\n",
        "- Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: "
        "Rank-based criteria. *Neurocomputing*, 72(7-9), 1431-1443.\n",
        "- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.\n",
        "- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. "
        "*Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297.\n",
        "- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation "
        "and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.\n",
        "- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. "
        "*Philosophical Magazine*, 2(11), 559-572.\n",
        "- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation "
        "of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.\n",
        "- UCI Machine Learning Repository. Heart Disease Dataset. "
        "https://archive.ics.uci.edu/dataset/45/heart+disease\n",
        "- Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: "
        "Variants, properties, normalization and correction for chance. *Journal of Machine Learning Research*, 11, 2837-2854.\n"
    ]
})

# ============================================================
# Agregar y guardar
# ============================================================
nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'OK - Notebook limpio: {len(nb["cells"])} celdas (22 originales + {len(new_cells)} nuevas)')
