from pathlib import Path
from dashboard.viz import generate_task2_artifacts

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

datasets = ["smartphones", "tractors", "user_upload"]

for ds in datasets:
    result = generate_task2_artifacts(OUTPUTS, ds, k=10)
    p = result["paths"]
    print(
        f"[OK] {ds} -> "
        f"dashboard_inputs: {p['dashboard_inputs'].as_posix()} | "
        f"html: {p['fig_neighbor'].name}, {p['fig_hist'].name}, {p['fig_pca'].name}"
    )