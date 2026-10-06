import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\SAILAJA\miniproject"

GRAPHS_DIR = os.path.join(PROJECT_DIR, "results", "graphs")
GRADCAM_DIR = os.path.join(PROJECT_DIR, "results", "gradcam")

os.makedirs(GRAPHS_DIR, exist_ok=True)

# ============================================================
# 1. MODEL ACCURACY COMPARISON
# ============================================================

models = [
    "DenseNet121",
    "DenseNet121 + CBAM",
    "DenseNet121 + CBAM + Focal Loss"
]

accuracies = [95.64, 95.84, 97.12]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, accuracies)

plt.ylabel("Test Accuracy (%)")
plt.xlabel("Model")
plt.title("Comparison of Model Test Accuracy")
plt.ylim(90, 100)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.15,
        f"{value:.2f}%",
        ha="center"
    )

plt.xticks(rotation=12)
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS_DIR, "model_accuracy_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

class_names = [
    "colon_aca",
    "colon_n",
    "lung_aca",
    "lung_n",
    "lung_scc"
]

cm = np.array([
    [500, 0,   0,   0,   0],
    [5,   495, 0,   0,   0],
    [1,   0,   469, 0,   30],
    [0,   0,   0,   500, 0],
    [4,   0,   32,  0,   463]
])

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")

plt.title(
    "Confusion Matrix - DenseNet121 + CBAM + Focal Loss"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(5),
    class_names,
    rotation=35,
    ha="right"
)

plt.yticks(range(5), class_names)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.colorbar(label="Number of Images")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS_DIR, "confusion_matrix_final.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 3. PRECISION / RECALL / F1
# ============================================================

precision = [0.98, 1.00, 0.94, 1.00, 0.94]
recall =    [1.00, 0.99, 0.94, 1.00, 0.93]
f1 =        [0.99, 0.99, 0.94, 1.00, 0.93]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 5))

plt.bar(
    x - width,
    np.array(precision) * 100,
    width,
    label="Precision"
)

plt.bar(
    x,
    np.array(recall) * 100,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    np.array(f1) * 100,
    width,
    label="F1-score"
)

plt.ylabel("Score (%)")
plt.xlabel("Class")
plt.title(
    "Per-Class Performance - DenseNet121 + CBAM + Focal Loss"
)

plt.xticks(
    x,
    class_names,
    rotation=25,
    ha="right"
)

plt.ylim(85, 105)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS_DIR, "per_class_metrics.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 4. COMBINE ALL 5 GRAD-CAM IMAGES
# ============================================================

gradcam_files = [
    ("Colon Adenocarcinoma", "gradcam_colon_aca.png"),
    ("Normal Colon", "gradcam_colon_n.png"),
    ("Lung Adenocarcinoma", "gradcam_lung_aca.png"),
    ("Normal Lung", "gradcam_lung_n.png"),
    ("Lung Squamous Cell Carcinoma", "gradcam_lung_scc.png")
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for i, (title, filename) in enumerate(gradcam_files):

    image_path = os.path.join(
        GRADCAM_DIR,
        filename
    )

    if os.path.exists(image_path):

        image = Image.open(image_path)

        axes[i].imshow(image)

        axes[i].set_title(
            title,
            fontsize=11
        )

        axes[i].axis("off")

    else:

        axes[i].text(
            0.5,
            0.5,
            f"Missing:\n{filename}",
            ha="center",
            va="center"
        )

fig.suptitle(
    "Grad-CAM Visualizations - DenseNet121 + CBAM + Focal Loss",
    fontsize=15
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    os.path.join(
        GRAPHS_DIR,
        "gradcam_all_classes.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# DONE
# ============================================================

print("=" * 60)
print("FINAL RESULTS VISUALIZATION COMPLETED")
print("=" * 60)

print("\nSaved to:")
print(GRAPHS_DIR)

print("\nGenerated files:")

print("1. model_accuracy_comparison.png")
print("2. confusion_matrix_final.png")
print("3. per_class_metrics.png")
print("4. gradcam_all_classes.png")

print("\nGrad-CAM files checked from:")
print(GRADCAM_DIR)