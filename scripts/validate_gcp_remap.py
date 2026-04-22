"""
export_gcp_projections.py

Run as a Metashape script (File > Run Script) on the ASFM project after
alignment and optimization are complete.

Exports a CSV: marker_label, camera_label, pixel_x, pixel_y
These pixel coordinates have known world coordinates in gcp_reference.csv,
so they can be used to validate the NPZ remapping end-to-end.

Usage in Metashape console:
    import runpy; runpy.run_path("export_gcp_projections.py")
"""
import csv
from pathlib import Path
import Metashape
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


project_path = Path("data/longterm_images2/semifield-developed-images/NC_2026-04-10/autosfm/project/NC_2026-04-10.psx")
doc = Metashape.Document()
doc.open(str(project_path), read_only=False, ignore_lock=True)
chunk = doc.chunk
output_path = Path("data/longterm_images2/semifield-developed-images/NC_2026-04-10/autosfm/reference/gcp_projections.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
for marker in chunk.markers:
    if marker.position is None:
        continue
    for camera, proj in marker.projections.items():
        if camera.transform is None:
            continue
        rows.append({
            "marker_label": marker.label,
            "camera_label": camera.label,
            "pixel_x": proj.coord.x,
            "pixel_y": proj.coord.y,
        })

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["marker_label", "camera_label", "pixel_x", "pixel_y"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Exported {len(rows)} GCP projections to {output_path}")

"""
test_gcp_remap.py

Run on SciNet after transferring NPZ grids and gcp_projections.csv.
Remaps known GCP pixel locations and compares against known world coords.

Usage:
    python test_gcp_remap.py \
        --projections path/to/reference/gcp_projections.csv \
        --gcp-ref     path/to/reference/gcp_reference.csv \
        --grids       path/to/reference/pixel_world_grids
"""


def load_camera_grid(npz_path):
    data = np.load(str(npz_path), allow_pickle=True)
    kw = dict(method='linear', bounds_error=False, fill_value=np.nan)
    ix = RegularGridInterpolator((data['v_pixels'], data['u_pixels']), data['world_x'], **kw)
    iy = RegularGridInterpolator((data['v_pixels'], data['u_pixels']), data['world_y'], **kw)
    w = int(data['sensor_width'][0])
    h = int(data['sensor_height'][0])
    return ix, iy, w, h

def load_gcp_reference(gcp_ref_path):
    """
    Load GCP reference table, collapsing duplicate labels only when their
    coordinates are identical.

    Returns
    -------
    pd.DataFrame
        Indexed by label, with unique labels only.

    Raises
    ------
    ValueError
        If a duplicate label has conflicting coordinate values.
    """
    gcps = pd.read_csv(gcp_ref_path)

    required_cols = ["label", "Estimated_X", "Estimated_Y"]
    missing = [col for col in required_cols if col not in gcps.columns]
    if missing:
        raise ValueError(f"Missing required columns in GCP reference: {missing}")

    deduped_rows = []

    for label, group in gcps.groupby("label", sort=False):
        if len(group) == 1:
            deduped_rows.append(group.iloc[0])
            continue

        first_x = group.iloc[0]["Estimated_X"]
        first_y = group.iloc[0]["Estimated_Y"]

        same_x = np.isclose(group["Estimated_X"].to_numpy(dtype=float), float(first_x), equal_nan=True).all()
        same_y = np.isclose(group["Estimated_Y"].to_numpy(dtype=float), float(first_y), equal_nan=True).all()

        if same_x and same_y:
            print(f"Deduplicating label '{label}' ({len(group)} identical rows)")
            deduped_rows.append(group.iloc[0])
        else:
            print(f"\nConflicting duplicate rows found for label '{label}':")
            print(group[["label", "Estimated_X", "Estimated_Y"]].to_string(index=False))
            raise ValueError(
                f"Duplicate label '{label}' has conflicting Estimated_X / Estimated_Y values."
            )

    gcps_deduped = pd.DataFrame(deduped_rows).set_index("label")
    return gcps_deduped

def main():
    
    gcp_ref_path = Path("data/longterm_images2/semifield-developed-images/NC_2026-04-10/autosfm/reference/gcp_reference.csv")
    grids = Path("data/longterm_images2/semifield-developed-images/NC_2026-04-10/autosfm/reference/pixel_world_grids")
    
    proj  = pd.read_csv(output_path)
    gcps  = load_gcp_reference(gcp_ref_path)
    

    print(f"Testing {len(proj)} GCP projections across {proj['camera_label'].nunique()} cameras\n")
    print(f"{'marker':<20} {'camera':<30} {'err_x_m':>10} {'err_y_m':>10} {'dist_m':>10}")
    print("-" * 80)

    errors = []
    skipped = 0

    for _, row in proj.iterrows():
        marker = row['marker_label']
        camera = row['camera_label']

        if marker not in gcps.index:
            skipped += 1
            continue

        npz_path = grids / f"{camera}.npz"
        if not npz_path.exists():
            skipped += 1
            continue

        ix, iy, w, h = load_camera_grid(npz_path)

        # pixel_x/y from Metashape are absolute pixel coords
        pt = np.array([[row['pixel_y'], row['pixel_x']]])
        remapped_x = float(ix(pt)[0])
        remapped_y = float(iy(pt)[0])

        known_x = gcps.loc[marker, 'Estimated_X']
        known_y = gcps.loc[marker, 'Estimated_Y']

        err_x = remapped_x - known_x
        err_y = remapped_y - known_y

        print(err_x, err_y)
        dist  = (err_x**2 + err_y**2) ** 0.5
        errors.append(dist)

        # dist is a panda series
        
        status = "OK" if dist < 0.05 else "WARN"
        print(f"{marker:<20} {camera:<30} {err_x:>10.4f} {err_y:>10.4f} {dist:>10.4f}  [{status}]")

    print(f"\nSkipped: {skipped} (no NPZ or no matching GCP reference)")
    if errors:
        print(f"Mean error:   {np.mean(errors):.4f} m")
        print(f"Max error:    {np.max(errors):.4f} m")
        print(f"95th pct:     {np.percentile(errors, 95):.4f} m")
        print("\nExpected: mean < 0.02 m for a well-aligned project.")
        print("Values > 0.05 m suggest a grid step that is too coarse or a")
        print("model built from a different alignment than the NPZ export.")


if __name__ == "__main__":
    main()
