from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px


# =========================
# Helpers / Loading
# =========================
def _clean_str(x) -> str:
    return str(x).strip()


def dataset_paths(outputs_dir: str | Path, dataset: str) -> dict[str, Path]:
    base = Path(outputs_dir) / dataset
    return {
        "base": base,
        "pre": base / "preprocessed_output.csv",
        "dm": base / "distance_matrix.csv",
        "nn": base / "nearest_neighbors.csv",
        "summary": base / "distance_summary.txt",
        "dashboard_inputs": base / "dashboard_inputs.csv",
        "fig_neighbor": base / "neighbor_bar.html",
        "fig_hist": base / "distance_hist.html",
        "fig_pca": base / "pca_embedding.html",
    }


def load_preprocessed_numeric(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}. PCA requires numeric features.")
    return num


def load_distance_matrix(path: str | Path) -> pd.DataFrame:
    dm = pd.read_csv(path, index_col=0)
    dm.index = dm.index.astype(str).str.strip()
    dm.columns = dm.columns.astype(str).str.strip()
    dm = dm.apply(pd.to_numeric, errors="coerce")
    if dm.isna().any().any():
        dm = dm.fillna(dm.mean(numeric_only=True))
    return dm


def load_neighbors_table(path: str | Path) -> pd.DataFrame:
    nn = pd.read_csv(path)

    expected = {"product", "neighbor", "distance"}
    if expected.issubset(set(nn.columns)):
        nn["product"] = nn["product"].astype(str).str.strip()
        nn["neighbor"] = nn["neighbor"].astype(str).str.strip()
        nn["distance"] = pd.to_numeric(nn["distance"], errors="coerce")
        return nn

    nn2 = nn.copy()
    nn2.iloc[:, 0] = nn2.iloc[:, 0].astype(str).str.strip()
    nn2.iloc[:, 2] = nn2.iloc[:, 2].astype(str).str.strip()
    nn2.iloc[:, -1] = pd.to_numeric(nn2.iloc[:, -1], errors="coerce")
    nn2 = nn2.rename(
        columns={
            nn2.columns[0]: "product",
            nn2.columns[2]: "neighbor",
            nn2.columns[-1]: "distance",
        }
    )
    return nn2[["product", "neighbor", "distance"]]


# =========================
# Visualizations
# =========================
def neighbors_for_product(nn_df: pd.DataFrame, product: str, k: int = 10) -> pd.DataFrame:
    product = _clean_str(product)
    sub = nn_df[nn_df["product"] == product].sort_values("distance").head(k).copy()
    return sub[["neighbor", "distance"]]


def plot_neighbor_bar(neighbors_df: pd.DataFrame, selected_product: str):
    fig = px.bar(
        neighbors_df.sort_values("distance", ascending=True),
        x="distance",
        y="neighbor",
        orientation="h",
        title=f"Nearest neighbors of selected product"
    )
    return fig


def plot_distance_histogram(dm: pd.DataFrame, nbins: int = 40):
    arr = dm.to_numpy()
    n = arr.shape[0]
    tri = arr[np.triu_indices(n, k=1)]
    tri = tri[~np.isnan(tri)]
    fig = px.histogram(x=tri, nbins=nbins, title="Distance distribution (all product pairs)")
    fig.update_layout(xaxis_title="distance", yaxis_title="count")
    return fig


def compute_pca_embedding(preprocessed_numeric: pd.DataFrame, product_keys: list[str]) -> pd.DataFrame:
    product_keys = [_clean_str(p) for p in product_keys]
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(preprocessed_numeric.values)
    return pd.DataFrame(
        {"product": product_keys, "embedding_x": emb[:, 0], "embedding_y": emb[:, 1]}
    )


def plot_embedding(embedding_df: pd.DataFrame, selected: str, neighbors: list[str]):
    df = embedding_df.copy()
    df["product"] = df["product"].astype(str).str.strip()
    selected = _clean_str(selected)
    neighbors = [_clean_str(x) for x in neighbors]

    df["role"] = "other"
    df.loc[df["product"] == selected, "role"] = "selected"
    df.loc[df["product"].isin(neighbors), "role"] = "neighbor"

    fig = px.scatter(
        df,
        x="embedding_x",
        y="embedding_y",
        hover_name="product",
        color="role",
        title="2D embedding (PCA)"
    )
    return fig


# =========================
# dashboard_inputs export
# =========================
def nearest_neighbor_distance(dm: pd.DataFrame) -> pd.Series:
    nn = []
    for p in dm.index:
        s = dm.loc[p].drop(index=p, errors="ignore")
        nn.append(float(s.min()))
    return pd.Series(nn, index=dm.index, name="nearest_neighbor_distance")


def export_dashboard_inputs(embedding_df: pd.DataFrame, dm: pd.DataFrame, out_path: str | Path):
    out = embedding_df.copy()
    out["product"] = out["product"].astype(str).str.strip()

    nn_dist = nearest_neighbor_distance(dm)
    nn_map = {str(k).strip(): float(v) for k, v in nn_dist.items()}
    out["nearest_neighbor_distance"] = out["product"].map(nn_map)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


# =========================
# One-call bundle for Task2
# =========================
def generate_task2_artifacts(
    outputs_dir: str | Path,
    dataset: str,
    selected_product: str | None = None,
    k: int = 10,
    html_include_plotlyjs: str = "cdn",
):
    """
    Generates:
      - neighbor bar/table (fig + df)
      - distance distribution histogram (fig)
      - PCA 2D embedding (fig + df)
      - dashboard_inputs.csv export

    Saves HTML figures to outputs/<dataset>/:
      - neighbor_bar.html
      - distance_hist.html
      - pca_embedding.html
    """
    paths = dataset_paths(outputs_dir, dataset)

    # validate required files
    for key in ["pre", "dm", "nn"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required file for {dataset}: {paths[key]}")

    nn_df = load_neighbors_table(paths["nn"])
    dm = load_distance_matrix(paths["dm"])
    pre = load_preprocessed_numeric(paths["pre"])

    if selected_product is None:
        selected_product = sorted(nn_df["product"].unique())[0]

    neigh_df = neighbors_for_product(nn_df, selected_product, k=k)
    fig_neigh = plot_neighbor_bar(neigh_df, selected_product)

    fig_hist = plot_distance_histogram(dm)

    emb_df = compute_pca_embedding(pre, list(dm.index))
    fig_pca = plot_embedding(emb_df, selected_product, neighbors=list(neigh_df["neighbor"]))

    # export dashboard inputs
    export_dashboard_inputs(emb_df, dm, paths["dashboard_inputs"])

    # save htmls
    paths["base"].mkdir(parents=True, exist_ok=True)
    fig_neigh.write_html(paths["fig_neighbor"], include_plotlyjs=html_include_plotlyjs)
    fig_hist.write_html(paths["fig_hist"], include_plotlyjs=html_include_plotlyjs)
    fig_pca.write_html(paths["fig_pca"], include_plotlyjs=html_include_plotlyjs)

    return {
        "paths": paths,
        "selected_product": selected_product,
        "neighbors_df": neigh_df,
        "embedding_df": emb_df,
        "fig_neighbor": fig_neigh,
        "fig_hist": fig_hist,
        "fig_pca": fig_pca,
    }