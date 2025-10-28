# plotter.py
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse

def plot_clusters(X, labels):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("Clusters")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")