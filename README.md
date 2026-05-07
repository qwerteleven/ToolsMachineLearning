# Tools for Machine Learning

A collection of self-contained notebooks and scripts exploring machine learning algorithms from scratch — focusing on understanding how they work internally, not just calling library APIs.

## Contents

### K-Means Clustering with animated GIF output (`ClusteringK-MeansGIF.ipynb`)

A from-scratch implementation of K-Means and K-Means++ on the Iris dataset, with two key features:

- **Animated GIF generation** — each iteration of centroid movement and cluster assignment is captured as a frame and saved to `Simple_kmeans.gif`, making the convergence process visually explicit
- **K-Means++ initialisation** — smarter centroid seeding using distance-weighted probability, reducing sensitivity to bad random starts compared to naive random initialisation
- **Elbow method** — automatic K selection via inertia curve using `yellowbrick`

The custom `Kmeans` class implements the full algorithm manually: centroid initialisation, assignment step, update step, convergence check (tolerance 0.001), and visualisation — without delegating to `sklearn.cluster.KMeans`.

Both initialisation strategies are available:

```python
kmeansModel.init_position_centroid()         # random init (standard K-Means)
kmeansModel.init_position_centroid_kmeans_plus()  # distance-weighted init (K-Means++)
kmeansModel.animated_update(n=20)            # run n iterations, save GIF
```

---

### Mandelbrot Set renderer (`mandelbrot.py`)

A pure-Python renderer of the Mandelbrot set using PIL. Iterates each pixel in the complex plane and maps escape velocity to greyscale intensity.

```python
python mandelbrot.py
```

Configurable parameters at the top of the file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WIDTH / HEIGHT` | 600 × 400 | Output image resolution |
| `MAX_ITER` | 80 | Maximum iterations before declaring convergence |
| `RE_START / RE_END` | -2 to 1 | Real axis window |
| `IM_START / IM_END` | -1 to 1 | Imaginary axis window |

> **Note:** The current implementation applies `math.exp()` to the coordinate mapping, which produces a non-linear projection of the complex plane. Remove the `math.exp()` calls for a standard linear Mandelbrot render.

---

## Installation

```bash
pip install numpy scikit-learn pandas seaborn scipy matplotlib yellowbrick gif Pillow
```

Each file is independent — install only what you need.

## Structure

```
├── ClusteringK-MeansGIF.ipynb   # K-Means / K-Means++ from scratch with GIF output
├── mandelbrot.py                 # Mandelbrot set renderer
└── README.md
```

## License

MIT