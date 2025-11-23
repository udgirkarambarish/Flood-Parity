# Flood-Parity

FloodParity is a lightweight evaluation tool used to validate high-resolution flood predictions from surrogate models by comparing `pred.tif` (predicted flood depth) against `truth.tif` (actual flood depth). It generates both quantitative metrics and visual plots, along with an automated PDF report.

* * * * *

Features
--------

-   Evaluates continuous flood depth predictions

-   Implements the following metrics:

    -   MAE (Mean Absolute Error)

    -   RMSE (Root Mean Square Error)

    -   IoU for flooded class

    -   IoU for non-flooded class

    -   % area misclassified (false positives only)

-   Produces visual plots:

    -   Truth vs Predicted scatter

    -   Error map

    -   Thresholded flood maps

-   Generates a consolidated PDF report containing metrics + plots

* * * * *

Input Format
------------

Two GeoTIFF raster files are required:

-   `truth.tif` → ground-truth flood water depth

-   `pred.tif` → surrogate-model predicted water depth

Flood classification is derived directly from the truth raster:

-   flooded = depth > 0

-   predicted flooded = pred > 0

No artificial hydrology-based threshold is applied to truth.

* * * * *

How to Run
----------

1.  Place `truth.tif` and `pred.tif` in the working directory.

2.  Install the requirements:

    ```
    pip install -r requirements.txt
    ```

3.  Run the evaluation script:

    ```
    python parity.py
    ```

After execution, the following files are generated:

-   scatter_plot.png

-   error_map.png

-   flood_maps.png

-   parity.pdf

* * * * *

Output Report
-------------

The generated PDF includes:

-   All evaluation metrics

-   Scatter plot

-   Error map

-   Truth vs predicted flood mask comparison

This setup enables fast and reliable benchmarking for hydrodynamic surrogate models such as FNO-based flood predictors.

* * * * *