import rasterio
import numpy as np

truth_path = "truth.tif"
pred_path = "pred.tif"

print("\n=== LOADING RASTERS ===")
with rasterio.open(truth_path) as truth_src:
    truth = truth_src.read(1).astype(float)
    print("Truth CRS:", truth_src.crs)
    print("Truth bounds:", truth_src.bounds)
    print("Truth resolution:", truth_src.res)
    print("Truth nodata:", truth_src.nodata)
    print("Truth shape:", truth.shape)
    print("Truth depth stats (min, max):", np.nanmin(truth), np.nanmax(truth))

with rasterio.open(pred_path) as pred_src:
    pred = pred_src.read(1).astype(float)
    print("\nPred CRS:", pred_src.crs)
    print("Pred bounds:", pred_src.bounds)
    print("Pred resolution:", pred_src.res)
    print("Pred nodata:", pred_src.nodata)
    print("Pred shape:", pred.shape)
    print("Pred depth stats (min, max):", np.nanmin(pred), np.nanmax(pred))

print("\n=== OVERLAP CHECK ===")
with rasterio.open(truth_path) as t, rasterio.open(pred_path) as p:
    if t.crs != p.crs:
        print("⚠ CRS mismatch — rasters are in different map projections.")
    else:
        print("✔ CRS matches.")

    # Bounding box overlap check
    tb, pb = t.bounds, p.bounds
    overlap = not (
        pb.right < tb.left or
        pb.left > tb.right or
        pb.top < tb.bottom or
        pb.bottom > tb.top
    )
    print("Overlap in map space:", overlap)

print("\n=== NOTE ===")
print("If CRS differs or overlap=False, parity metrics will fail until alignment/resampling is corrected.")
