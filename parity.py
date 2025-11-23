import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
truth_path = "truth.tif"
pred_path = "pred.tif"
pdf_output = "parity.pdf"

# -----------------------------------------------------------
# Load rasters
# -----------------------------------------------------------
with rasterio.open(truth_path) as src:
    truth = src.read(1).astype(float)
    truth_nodata = src.nodata

with rasterio.open(pred_path) as src:
    pred = src.read(1).astype(float)
    pred_nodata = src.nodata

# -----------------------------------------------------------
# Create valid mask
# -----------------------------------------------------------
mask = np.ones_like(truth, dtype=bool)

if truth_nodata is not None:
    mask &= truth != truth_nodata
if pred_nodata is not None:
    mask &= pred != pred_nodata

mask &= ~np.isnan(truth) & ~np.isnan(pred)

truth_valid = truth[mask]
pred_valid = pred[mask]

# -----------------------------------------------------------
# METRICS
# -----------------------------------------------------------
# MAE & RMSE on continuous depth
mae = np.mean(np.abs(pred_valid - truth_valid))
rmse = np.sqrt(np.mean((pred_valid - truth_valid)**2))

# Flood definition based directly on truth raster
truth_bin = (truth > 0).astype(int)
pred_bin = (pred > 0).astype(int)

# IoU helper
def compute_iou(a, b, mask, cls):
    inter = ((a == cls) & (b == cls) & mask).sum()
    union = (((a == cls) | (b == cls)) & mask).sum()
    if union == 0:
        return np.nan
    return inter / union

iou_flooded = compute_iou(truth_bin, pred_bin, mask, 1)
iou_nonflooded = compute_iou(truth_bin, pred_bin, mask, 0)

# % area misclassified using user's definition (False Positives only)
false_positive_mask = (pred_bin == 1) & (truth_bin == 0) & mask
percent_misclassified = false_positive_mask.sum() / mask.sum() * 100

# -----------------------------------------------------------
# PLOTS
# -----------------------------------------------------------
plot_files = {}

# Scatter
plt.figure(figsize=(6, 5))
plt.scatter(truth_valid, pred_valid, s=1)
plt.xlabel("Truth Depth")
plt.ylabel("Predicted Depth")
plt.title("Truth vs Prediction Scatter")
plt.grid(True)
scatter_path = "scatter_plot.png"
plt.savefig(scatter_path, dpi=300)
plt.close()
plot_files["Scatter Plot"] = scatter_path

# Error map
error = np.zeros_like(truth)
error[mask] = pred[mask] - truth[mask]

plt.figure(figsize=(6, 5))
plt.imshow(error, cmap="coolwarm")
plt.colorbar(label="Prediction - Truth (m)")
plt.title("Error Map")
error_map_path = "error_map.png"
plt.savefig(error_map_path, dpi=300)
plt.close()
plot_files["Error Map"] = error_map_path

# Flood masks visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(truth_bin, cmap="Blues")
plt.title("Truth Flood Mask")

plt.subplot(1, 2, 2)
plt.imshow(pred_bin, cmap="Oranges")
plt.title("Prediction Flood Mask")

flood_map_path = "flood_maps.png"
plt.savefig(flood_map_path, dpi=300)
plt.close()
plot_files["Flood Maps"] = flood_map_path

# -----------------------------------------------------------
# PDF REPORT
# -----------------------------------------------------------
with PdfPages(pdf_output) as pdf:
    # Page 1 (Metrics)
    plt.figure(figsize=(8, 5))
    plt.axis("off")
    text = (
        "FLOOD MODEL EVALUATION REPORT\n\n"
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"IoU (Flooded): {iou_flooded:.4f}\n"
        f"IoU (Non-Flooded): {iou_nonflooded:.4f}\n"
        f"% Area Misclassified (FP only): {percent_misclassified:.2f}%\n"
        f"Flood definition used: truth > 0 (truth controls classification)\n"
    )
    plt.text(0.1, 0.5, text, fontsize=12)
    pdf.savefig()
    plt.close()

    # Add plots
    for title, path in plot_files.items():
        img = plt.imread(path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        pdf.savefig()
        plt.close()

print("\n✓ Finished successfully.")
print(f"PDF saved as: {pdf_output}")
