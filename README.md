<div align="center">
  <h1>Unsupervised Text Clustering</h1>
  <p><strong>Chapter 8 — Natural Language Processing & Text Processing Pipeline</strong></p>

  <p align="center">
    <img src="https://img.shields.io/badge/Language-Python_3-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Environment-Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Environment">
    <img src="https://img.shields.io/badge/Dataset-TMDB_Movies-01b4e4?style=flat-square" alt="Dataset">
    <img src="https://img.shields.io/badge/Algorithms-K--Means_%26_Agglomerative-blue?style=flat-square" alt="Algorithms">
    <img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" alt="Status">
  </p>

  <p align="center">
    Unsupervised text clustering on TMDB movie overviews comparing Non-Hierarchical K-Means and Hierarchical Agglomerative Clustering across sparse TF-IDF and dense continuous Word2Vec embeddings with Elbow, Silhouette, Word Cloud, and UMAP 2D evaluations.
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Overview

Repositori ini memuat implementasi mandiri untuk modul **Chapter 8 - Text Clustering** pada mata kuliah *Text Processing* Program Studi Sains Data Universitas Negeri Surabaya. Praktikum ini berfokus pada pengelompokan korpus teks dokumen film *The Movie Database* (TMDB) secara tanpa supervisi (*unsupervised learning*) menggunakan fitur leksikal TF-IDF dan representasi dense kontinu Word2Vec hasil *feature engineering* sebelumnya.

Notebook utama [`Text_Clustering_Rava_Welfee.ipynb`](Text_Clustering_Rava_Welfee.ipynb) membandingkan dua paradigma klasterisasi utama:
1. **Non-Hierarchical Clustering (K-Means)**
2. **Hierarchical Agglomerative Clustering (Bottom-Up Tree)**

## Repository Structure

```text
Text-Clustering/
├── data/
│   └── tmdb_preprocessed.csv                  # Preprocessed TMDB movie metadata and overviews
├── models/
│   └── tmdb_word2vec.w2v                      # Pre-trained Word2Vec CBOW model
├── .gitignore                                 # Git ignore rules
├── requirements.txt                           # Python dependencies
├── Text_Clustering_Rava_Welfee.ipynb          # Main executed text clustering notebook
└── README.md                                  # Project documentation
```

## Clustering Methodologies

### 1. K-Means Clustering
Mempartisi korpus menjadi $K$ kelompok dengan meminimalkan jumlah kuadrat jarak titik data terhadap pusat klaster (*within-cluster sum of squares* / inertia):

$$
J = \sum_{j=1}^{K} \sum_{i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2
$$

- **Elbow Method**: Menguji rentang $K = 1 \dots 10$ untuk menemukan titik siku penurunan inersia paling signifikan.
- **Silhouette Analysis**: Mengukur rasio kohesi intra-cluster terhadap separasi inter-cluster:
  $$s(i) = \frac{b(i) - a(i)}{\max[a(i), b(i)]}$$
- **Centroid Profiling**: Mengidentifikasi istilah dengan bobot tertinggi di setiap centroid untuk mengekstrak tema naratif.
- **Word Cloud Visualization**: Memvisualisasikan awan kata dominan 4-panel untuk masing-masing klaster.

### 2. Hierarchical Agglomerative Clustering
Membangun struktur pohon pengelompokan dari bawah ke atas (*bottom-up*) dengan mengevaluasi empat kriteria *linkage*:
- **Single Linkage**: $L(r, s) = \min D(x_{ri}, x_{sj})$
- **Complete Linkage**: $L(r, s) = \max D(x_{ri}, x_{sj})$
- **Average Linkage**: $L(r, s) = \frac{1}{n_r n_s} \sum_{i=1}^{n_r} \sum_{j=1}^{n_s} D(x_{ri}, x_{sj})$
- **Ward Linkage**: Meminimalkan kenaikan total variansi dalam cluster (Ward minimum variance method).
- **Dendrogram**: Diagram hierarkis dengan *cut-off line threshold* untuk menentukan partisi $K=4$.

### 3. Clustering pada Continuous Dense Embeddings (Word2Vec)
Mengelompokkan dokumen menggunakan representasi dense 100-dimensi hasil *mean pooling*. Pengujian membuktikan bahwa representasi kontinu mengatasi *curse of dimensionality* (Silhouette Score meningkat signifikan hingga **~0.073** dibandingkan representasi TF-IDF yang memiliki *sparsity* >98%).

### 4. Proyeksi Manifold 2D (UMAP)
Memproyeksikan data teks berdimensi tinggi ke koordinat 2D menggunakan UMAP dengan metrik kosinus untuk memvisualisasikan separasi klaster K-Means versus Hierarchical secara berdampingan.

## Model Performance & Evaluation

| Metode Klaster | Fitur Input | Dimensi Fitur | Sparsity | Silhouette Score | Karakteristik Utama |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means** | TF-IDF | 1,000 | >98% | 0.0013 | Pemisahan leksikal cepat berbasis kata kunci |
| **Agglomerative (Ward)** | TF-IDF | 1,000 | >98% | 0.0034 | Struktur hierarki seimbang dengan visualisasi dendrogram |
| **K-Means (Word2Vec)** | Dense Continuous | 100 | 0.00% | **0.0729** | Klaster semantik padat bebas *curse of dimensionality* |

## Installation and Execution

```bash
# 1. Clone repository
git clone https://github.com/rava-andika/Text-Clustering.git
cd Text-Clustering

# 2. Buat virtual environment (opsional)
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan notebook
jupyter notebook Text_Clustering_Rava_Welfee.ipynb
```
Semua path aset menggunakan *adaptive relative path* sehingga siap dijalankan di lingkungan lokal Jupyter maupun Google Colab.
