import io, base64
import matplotlib.pyplot as plt

def plot_bar(labels, values) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Resultados")
    ax.set_ylabel("Valor")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")
