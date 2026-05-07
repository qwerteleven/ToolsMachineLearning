# Tools for Machine Learning

A collection of self-contained notebooks and scripts implementing classic machine learning algorithms from scratch — built to understand how they work internally, not just use library APIs. Each file is independent with no cross-dependencies.

## Contents

### K-Nearest Neighbours — classification and K selection (`Clasification K-Nearest Neighbours and k election.ipynb`)

From-scratch KNN classifier on the Wine dataset (178 samples, 13 features, 3 classes), with decision boundary visualisation across a meshgrid.

- Two distance metrics implemented: Euclidean and Manhattan — decision boundaries visualised side by side
- K selection via 10-fold cross-validation across K = 1..100, with train and test accuracy curves plotted together
- Features used: Ash vs Proline (2D for visualisation), with z-score normalisation

---

### K-Means clustering with animated GIF output (`ClusteringK-MeansGIF.ipynb`)

From-scratch K-Means and K-Means++ on the Iris dataset, with convergence animated frame by frame into a GIF.

- **K-Means++** initialisation — distance-weighted centroid seeding, reducing sensitivity to random starts
- **Elbow method** via `yellowbrick` for automatic K selection
- Convergence check with tolerance `0.001` — stops early when centroids stabilise
- Output: `Simple_kmeans.gif` showing centroid movement and cluster reassignment across iterations

```python
kmeansModel.init_position_centroid_kmeans_plus()
kmeansModel.animated_update(n=20)
```

---

### Linear Regression with MSE (`LinearRegression MSE.ipynb`)

Simple and multiple linear regression on the Boston Housing dataset, implemented via the closed-form OLS solution.

- OLS formula derived and applied manually: `W = (XᵀX)⁻¹XᵀY`
- MSE computed manually and verified against `sklearn`
- Includes data analysis: correlation coefficient between rooms (RM) and price, distribution histogram with truncation artefact noted at $50k
- Extends to multiple features with sklearn comparison

---

### Logistic Regression — preprocessing comparison (`LogisticClasification.ipynb`)

Binary classification on the Breast Cancer Wisconsin dataset (569 samples, 30 features), benchmarking 6 preprocessing pipelines across 250 random train/test splits to reduce split variance.

| Pipeline | Mean accuracy |
|----------|--------------|
| Simple (no preprocessing) | 94.4% |
| Polynomial features | 95.0% |
| Scale only | 97.6% |
| Polynomial → Scale | 96.2% |
| Scale → Polynomial | 97.0% |
| PCA (20 components) + Scale | **97.8%** |

Includes correlation heatmap (seaborn + matplotlib) across all 30 features. PCA retains 90% variance in 20 components.

---

### Logistic function, Iris exploration, and Monty Hall simulation (`LogisticFunction IrisFLower MontyHall.ipynb`)

Three independent topics in one notebook:

- **Sigmoid function** — plotted across [-10, 10], paired with a normal distribution visualisation
- **Iris dataset exploration** — descriptive statistics, petal length histograms per class, sepal vs petal scatter
- **Monty Hall simulation** — 100,000 trials confirming the ~66.7% win rate when switching doors

---

### Polynomial vs Linear Regression (`PolynomialRegression vs Linear Regression.ipynb`)

Regression on an insurance claims time series, comparing polynomial degree 1–6 using both manual OLS and sklearn, with MSE reported per degree.

| Degree | MSE |
|--------|-----|
| 1 | 108.6 |
| 2 | 72.5 |
| 3 | 54.2 |
| 4 | 50.8 |
| 5 | 45.5 |
| 6 | **19.1** |

The `poly_matrix()` function builds the Vandermonde-style design matrix manually — useful for understanding what `sklearn.PolynomialFeatures` does internally.

---

### Semi-supervised MNIST classifier (`MNISTsemi-supervisedClasificator-KNN-LogisR-LineR-PolyR-Kmeans.ipynb`)

MNIST digit classification comparing semi-supervised K-Means against supervised models, after PCA dimensionality reduction (784 → 87 components at 90% variance).

| Method | Accuracy | Labelled data used |
|--------|----------|--------------------|
| K-Means (semi-supervised) | ~53.8% | 0.5% |
| Linear Regression | ~57.0% | 75% |
| Polynomial Regression | ~83.9% | 75% |
| KNN (k=10) | ~90.2% | 75% |
| Logistic Regression | ~90.9% | 75% |

The semi-supervised approach assigns cluster labels by majority vote on a small labelled subset, then predicts the full dataset — achieving 53.8% accuracy using only 0.5% of labels. Each cluster is visualised as a mean digit image to verify label assignment.

---

### Self-Organising Map classifier (`SOM_clasificator.ipynb`)

A from-scratch SOM implementation in TensorFlow 1.x, applied to the Chicago Crime dataset. The SOM learns a 100×100 topological map of crime ward data, with crime type labels overlaid on the trained map.

- Gaussian neighbourhood function with decaying radius and learning rate across 300 epochs
- PCA used for colour encoding of SOM positions
- Requires TensorFlow 1.x (`tensorflow.compat.v1`)

> **Note:** The notebook contains an incomplete reference to `image_grid` and `mapped` — these variables are not defined in the shared code. The SOM training and map visualisation run correctly; only the final labelled overlay plot is broken.

---

### Mandelbrot Set renderer (`mandelbrot.py`)

Pure-Python Mandelbrot renderer using PIL. Maps escape velocity to greyscale intensity.

```bash
python mandelbrot.py
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WIDTH / HEIGHT` | 600 × 400 | Output resolution |
| `MAX_ITER` | 80 | Max iterations |
| `RE_START / RE_END` | -2 to 1 | Real axis window |
| `IM_START / IM_END` | -1 to 1 | Imaginary axis window |

---

## Installation

```bash
pip install numpy scikit-learn pandas seaborn scipy matplotlib yellowbrick gif Pillow tensorflow
```

Each notebook is self-contained — install only the libraries it uses.

## Datasets required

| Notebook | Dataset | Source |
|----------|---------|--------|
| `LogisticClasification.ipynb` | `breast_cancer_wisconsin.csv` | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| `PolynomialRegression...ipynb` | `insurance.csv` | Provided in repo |
| `MNISTsemi-supervised...ipynb` | `mnist_train_small.csv` | Provided in repo |
| `SOM_clasificator.ipynb` | `b.csv` (Chicago Crime) | [Chicago Data Portal](https://data.cityofchicago.org/) |

## Structure

```
├── Clasification K-Nearest Neighbours and k election.ipynb
├── ClusteringK-MeansGIF.ipynb
├── LinearRegression MSE.ipynb
├── LogisticClasification.ipynb
├── LogisticFunction IrisFLower MontyHall.ipynb
├── MNISTsemi-supervisedClasificator-KNN-LogisR-LineR-PolyR-Kmeans.ipynb
├── PolynomialRegression vs Linear Regression.ipynb
├── SOM_clasificator.ipynb
├── mandelbrot.py
└── README.md
```

## License

MIT