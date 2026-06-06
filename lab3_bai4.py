import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X = mnist.data
y = mnist.target
X = X[:2000]
y = y[:2000]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y.astype(int),
    cmap="tab10",
    s=10
)
plt.title("PCA Visualization of MNIST")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y.astype(int),
    cmap="tab10",
    s=10
)
plt.title("t-SNE Visualization of MNIST")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()
plt.show()