# Bank Customer Churn Prediction & Multi-Paradigm Behavioral Clustering

An end-to-end machine learning study featuring **supervised classification** (churn prediction under severe class imbalance) and **unsupervised behavioral segmentation** (benchmarking partitional, hierarchical, density-based, and probabilistic clustering models).

Developed as part of the graduate Machine Learning curriculum at the **Faculty of Electrical Engineering (ETF), University of Sarajevo**.

---

## 📌 Project Architecture & Scope

This project is divided into two comprehensive milestones based on a real-world banking credit card dataset (**10,127 instances, 19+ attributes**):

1. **Part I: Data Sanitization & Supervised Classification (Churn Prediction)**
   * Rigorous automated cleaning of synthetically corrupted real-world data (fuzzy string matching, regex-based sanitation, statistical median imputation).
   * Preventing data leakage (identifying and stripping pre-computed posterior attributes and artifacts).
   * Building production-grade `scikit-learn` pipelines (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
   * Mitigating extreme class imbalance (~5.17 : 1 ratio) using **SMOTE**, class weighting, and probability threshold tuning.
   * Benchmarking **k-NN**, **Support Vector Machines (RBF Kernel)**, and **Random Forest**, followed by a **Soft Voting Ensemble**.

2. **Part II: Unsupervised Customer Segmentation & Dimensionality Reduction**
   * Assessing clustering tendency via **Hopkins Statistic** ($H = 0.84$).
   * Dimensionality reduction using **Principal Component Analysis (PCA)** preserving 85–90% cumulative variance.
   * Comparative evaluation of 5 distinct clustering paradigms:
     * **Partitional:** k-Means & PAM (Partitioning Around Medoids)
     * **Hierarchical:** Agglomerative Clustering (Ward Linkage)
     * **Density-Based:** DBSCAN (k-distance heuristic, noise/outlier isolation)
     * **Model-Based (Probabilistic):** Gaussian Mixture Models (GMM) with full covariance optimization and soft assignment.
   * Hypothesis-driven behavioral and financial customer profiling.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Core ML & Pipelines:** `scikit-learn`, `imbalanced-learn` (SMOTE)
* **Data Processing & String Matching:** `pandas`, `numpy`, `thefuzz` / fuzzy regex heuristics
* **Dimensionality Reduction:** PCA (Principal Component Analysis)
* **Visualization:** `matplotlib`, `seaborn`

---

## 🔬 Part 1: Supervised Churn Classification

### 1. Data Cleaning & Feature Engineering
* **Noise Removal:** Handled ~10–25% injected typographic and structural noise per feature. Applied fuzzy string matching to map corrupted demographic inputs (e.g., `Education_Level`, `Marital_Status`, `Income_Category`) into canonical classes.
* **Leakage Prevention:** Detected and removed synthetic artifact columns (`Account_Status`, `Last_Login_Days`) and legacy model outputs (`Naive_Bayes_Classifier...`) to prevent artificial inflation of predictive performance.
* **Pipeline Integration:** Built leak-free pipelines using `ColumnTransformer` guaranteeing that scaling parameters and categorical encodings were learned strictly on training splits (70/15/15 stratified train/val/test split).

### 2. Model Performance & Evaluation
Due to severe class imbalance (83.8% existing vs. 16.2% churned), models were evaluated using **Precision**, **Recall**, **F1-score**, and **Cohen's Kappa ($\kappa$)** rather than misleading raw accuracy.

| Model | Focus & Optimization | Test Accuracy | Precision | Recall (Churn) | Cohen's Kappa ($\kappa$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **k-NN** | Minkowski metric ($p=2, k=8$), distance-weighted, custom threshold (0.25) | ~87.4% | ~87.5% | ~87.4% | 0.5389 |
| **SVM (RBF)** | SMOTE balanced, $C=1.0, \gamma=\text{'scale'}$, Permutation Importance | ~86.5% | 88.0% | 70.0% | 0.5338 |
| **Random Forest** *(Lead: Elma Tirak)* | Stratified 5-Fold CV, `RandomizedSearchCV` (800 trees, `max_depth=15`, `min_samples_leaf=4`) | **92.6%** | **82.6%** | **97.2%** | **0.7104** |
| **Soft Voting Ensemble** | k-NN + SVM + Random Forest weighted probability voting | **89.3%** | **81.0%** | 44.9% (Base) / **73.7%** (SMOTE) | 0.5228 / **0.5977** |

> **Key Finding:** Random Forest demonstrated intrinsic robustness to class imbalance, achieving an outstanding **97.2% Recall** on churned customers without requiring synthetic oversampling, outperforming both distance-based and boundary-margin approaches.

---

## 📊 Part 2: Unsupervised Customer Segmentation

### 1. Clustering Tendency & PCA
* **Hopkins Statistic:** Evaluated at **$H = 0.84$**, confirming significant spatial clustering tendency over uniform random noise.
* **Dimensionality Reduction:** 38 post-encoding features were reduced to **23 principal components** (capturing ~90% of total variance), eliminating multicollinearity between financial balance and credit limits.

### 2. Comparative Clustering Benchmark
Clustering quality was assessed using internal validation metrics: **Silhouette Coefficient ($S$)**, **Davies-Bouldin Index ($DB$)**, and **Calinski-Harabasz Index ($CH$)**.

* **k-Means & PAM ($k=4$):** Identified financial capacity as the primary grouping axis. k-Means with PCA achieved $S = 0.100, DB = 2.673, CH = 636.2$.
* **Hierarchical (Ward Linkage):** Discovered an optimal macro binary partition ($k=2, S=0.4385, DB=1.1174, CH=15490.5$) separating high-limit cardholders from conservative spenders.
* **DBSCAN ($\varepsilon=2.2, min\_samples=30$):** 
  * Isolated 53.76% of instances as sparse/transitional boundary noise.
  * Discovered an ultra-dense, niche **Premium Segment (0.32%, Credit Limit ~33,240 KM)** exhibiting identical transaction volume to standard users.
* **Gaussian Mixture Model (GMM) - Soft Clustering *(Lead: Elma Tirak)*:**
  * Modeled latent probabilistic components via Expectation-Maximization (EM).
  * Evaluated covariance topologies (`full`, `tied`, `diag`, `spherical`), confirming `full` covariance captured complex multi-attribute correlations.
  * Optimal component selection via Bayesian Information Criterion (**BIC**) and Akaike Information Criterion (**AIC**) yielding **$K = 10$ components**.
  * **Key Insight:** GMM uniquely isolated a completely homogeneous academic subgroup (**Cluster 7: 100% Doctorate holders**) and distinguished high credit limits from actual utilization habits.

---

## 👥 Authors & Contributions

This project was collaboratively executed by:

* **Elma Tirak** ([@etirak1](https://github.com/etirak1))  
  * *Lead:* Random Forest modeling & hyperparameter optimization, SMOTE sensitivity benchmark, Gaussian Mixture Models (GMM) tuning & covariance analysis, exploratory data analysis.
* **Benjamin Hadžihasanović**  
  * *Lead:* Support Vector Machine (SVM) modeling, RBF kernel optimization, DBSCAN spatial density clustering & noise isolation.
* **Tarik Čaluk**  
  * *Lead:* k-Nearest Neighbors (k-NN) classification, threshold tuning, Agglomerative Hierarchical Clustering (Ward linkage).

---

## 📄 License
This repository is released for academic, research, and portfolio demonstration purposes.
