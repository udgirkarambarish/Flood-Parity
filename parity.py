import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rasterio.warp import reproject, Resampling

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
truth_path = "truth.tif"
pred_path = "pred.tif"
pdf_output = "results_report.pdf"

# -----------------------------------------------------------
# Load rasters and align pred to truth
# -----------------------------------------------------------
with rasterio.open(truth_path) as truth_src, rasterio.open(pred_path) as pred_src:
    truth = truth_src.read(1).astype(float)
    pred = pred_src.read(1).astype(float)

    truth_nodata = truth_src.nodata
    pred_nodata = pred_src.nodata

    # If grid/resolution/CRS/transform differ, resample pred onto truth grid
    need_resample = (
        truth.shape != pred.shape
        or truth_src.transform != pred_src.transform
        or truth_src.crs != pred_src.crs
    )

    if need_resample:
        print("Resampling pred.tif to match truth.tif grid...")
        pred_resampled = np.zeros_like(truth, dtype=np.float32)

        reproject(
            source=pred,
            destination=pred_resampled,
            src_transform=pred_src.transform,
            src_crs=pred_src.crs,
            dst_transform=truth_src.transform,
            dst_crs=truth_src.crs,
            resampling=Resampling.bilinear,
        )
        pred = pred_resampled

# -----------------------------------------------------------
# Valid mask (exclude NoData and NaNs)
# -----------------------------------------------------------
mask = np.ones_like(truth, dtype=bool)

if truth_nodata is not None:
    mask &= truth != truth_nodata
if pred_nodata is not None:
    mask &= pred != pred_nodata

mask &= ~np.isnan(truth) & ~np.isnan(pred)

# If nothing valid, bail out
if mask.sum() == 0:
    raise RuntimeError("No valid overlapping pixels between truth and pred after masking.")

truth_valid = truth[mask]
pred_valid = pred[mask]

# -----------------------------------------------------------
# Metrics: MAE, RMSE
# -----------------------------------------------------------
mae = np.mean(np.abs(pred_valid - truth_valid))
rmse = np.sqrt(np.mean((pred_valid - truth_valid) ** 2))

# -----------------------------------------------------------
# Binary flood masks based on truth
# flooded = truth > 0
# predicted flooded = pred > 0
# -----------------------------------------------------------
truth_bin = np.zeros_like(truth, dtype=np.uint8)
pred_bin = np.zeros_like(truth, dtype=np.uint8)

truth_bin[mask] = (truth[mask] > 0).astype(np.uint8)
pred_bin[mask] = (pred[mask] > 0).astype(np.uint8)

# -----------------------------------------------------------
# IoU (Intersection over Union) for flooded / non-flooded
# -----------------------------------------------------------
def compute_iou(a, b, valid_mask, cls):
    inter = ((a == cls) & (b == cls) & valid_mask).sum()
    union = (((a == cls) | (b == cls)) & valid_mask).sum()
    if union == 0:
        return np.nan
    return inter / union

iou_flooded = compute_iou(truth_bin, pred_bin, mask, 1)
iou_nonflooded = compute_iou(truth_bin, pred_bin, mask, 0)

# -----------------------------------------------------------
# % area misclassified (user definition: false positives only)
#   FP = pred_bin = 1 and truth_bin = 0
# -----------------------------------------------------------
false_positive_mask = (pred_bin == 1) & (truth_bin == 0) & mask
percent_misclassified = false_positive_mask.sum() / mask.sum() * 100.0

# -----------------------------------------------------------
# Plots
# -----------------------------------------------------------
plot_files = {}

# 1) Truth vs Pred scatter
plt.figure(figsize=(6, 5))
plt.scatter(truth_valid, pred_valid, s=1)
plt.xlabel("Truth Depth")
plt.ylabel("Predicted Depth")
plt.title("Truth vs Prediction Scatter")
plt.grid(True)
scatter_path = "scatter_plot.png"
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
plt.close()
plot_files["Scatter Plot"] = scatter_path

# 2) Error map (pred - truth)
error = np.zeros_like(truth, dtype=float)
error[mask] = pred[mask] - truth[mask]

plt.figure(figsize=(6, 5))
im = plt.imshow(error, cmap="coolwarm")
plt.colorbar(im, label="Prediction - Truth (m)")
plt.title("Error Map")
error_map_path = "error_map.png"
plt.savefig(error_map_path, dpi=300, bbox_inches="tight")
plt.close()
plot_files["Error Map"] = error_map_path

# 3) Thresholded flood maps (truth vs pred)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(truth_bin, cmap="Blues")
plt.title("Truth Flood Mask")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_bin, cmap="Oranges")
plt.title("Prediction Flood Mask")
plt.axis("off")

flood_map_path = "flood_maps.png"
plt.savefig(flood_map_path, dpi=300, bbox_inches="tight")
plt.close()
plot_files["Flood Maps"] = flood_map_path

# -----------------------------------------------------------
# PDF report
# -----------------------------------------------------------
with PdfPages(pdf_output) as pdf:
    # Page 1: metrics summary
    plt.figure(figsize=(8, 5))
    plt.axis("off")
    text = (
        "FLOOD PARITY EVALUATION REPORT\n\n"
        f"MAE (depth): {mae:.4f}\n"
        f"RMSE (depth): {rmse:.4f}\n"
        f"IoU (Flooded): {iou_flooded:.4f}\n"
        f"IoU (Non-Flooded): {iou_nonflooded:.4f}\n"
        f"% Area Misclassified (FP only): {percent_misclassified:.2f}%\n\n"
        "Flood classification rule:\n"
        "  flooded           = truth > 0\n"
        "  predicted flooded = pred > 0\n"
        "Pred was resampled to truth grid if needed.\n"
    )

    plt.text(0.05, 0.5, text, fontsize=11, va="center", ha="left")
    pdf.savefig()
    plt.close()

    # Additional pages: plots
    for title, path in plot_files.items():
        img = plt.imread(path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        pdf.savefig()
        plt.close()

print("Done.")
print(f"PDF report saved as: {pdf_output}")
