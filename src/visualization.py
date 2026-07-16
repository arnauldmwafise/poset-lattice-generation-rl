import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import bitstring_to_matrix

def compute_transitive_reduction(matrix: np.ndarray) -> np.ndarray:
    """Computes the transitive reduction (covering relation) of a finite poset."""
    n = matrix.shape[0]
    strict_order = matrix.copy()
    np.fill_diagonal(strict_order, 0)
    reduction = strict_order.copy()
    for w in range(n):
        for u in range(n):
            for v in range(n):
                if strict_order[u, w] and strict_order[w, v]:
                    reduction[u, v] = 0
    return reduction

def generate_hasse_layout(reduction: np.ndarray) -> dict:
    """Generates a balanced Hasse diagram layout separated by algebraic rank."""
    n = reduction.shape[0]
    ranks = np.zeros(n, dtype=int)
    for _ in range(n):
        for u in range(n):
            for v in range(n):
                if reduction[u, v]:
                    if ranks[u] >= ranks[v]:
                        ranks[v] = ranks[u] + 1
                        
    max_rank = ranks.max() if ranks.size > 0 else 0
    rank_groups = {r: [] for r in range(max_rank + 1)}
    for node, r in enumerate(ranks):
        rank_groups[r].append(node)
        
    pos = {}
    for r, nodes in rank_groups.items():
        num_nodes = len(nodes)
        for idx, node in enumerate(nodes):
            x = idx - (num_nodes - 1) / 2.0 if num_nodes > 1 else 0.0
            y = r
            pos[node] = (x, y)
    return pos

def render_vector_pdf(matrix: np.ndarray, file_path: str, title: str = "Hasse Diagram"):
    """Generates a native vector PDF layout without external dependencies."""
    reduction = compute_transitive_reduction(matrix)
    pos = generate_hasse_layout(reduction)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    n = reduction.shape[0]
    
    # Render structural covering edges
    for u in range(n):
        for v in range(n):
            if reduction[u, v] == 1:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#4A5568", shrinkA=12, shrinkB=12)
                )
                
    # Render algebraic element nodes
    for node, (x, y) in pos.items():
        ax.plot(x, y, "o", markersize=20, color="#3182CE", zorder=3)
        ax.text(x, y, str(node), color="white", ha="center", va="center", weight="bold", fontsize=9, zorder=4)
        
    ax.set_title(title, fontsize=12, weight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    
    # Save directly as vector format
    fig.savefig(file_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
