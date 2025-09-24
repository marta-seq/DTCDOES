from tqdm import tqdm
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as ad
def create_anndata(label_file, meta_file, fov_folders):
    """
    Create an AnnData object by reading multiple loom files, adding cell labels and metadata.
    """
    meta = pd.read_excel(meta_file)

    all_adatas = []

    for fov_folder in tqdm(fov_folders):
        print("Processing:", fov_folder)
        loom_file = f"/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx/Baysor-master/OUTPUT/cellpose_masks/{fov_folder}/segmentation_counts.loom"

        # Read loom
        adata = sc.read_loom(loom_file)
        adata.var_names = adata.var["Name"].astype(str)

        # Standardize CellID
        adata.obs = adata.obs.reset_index().rename(columns={"index": "CellID"})
        adata.obs["CellID"] = adata.obs["CellID"].astype(str).str.replace(r"\.0$", "", regex=True)
        adata.obs = adata.obs.drop_duplicates(subset="CellID").set_index("CellID")

        # Add cell labels from meta CSV
        meta_labels = pd.read_csv(label_file)
        df_meta = meta_labels.loc[meta_labels['orig.ident'] == fov_folder].copy()
        df_meta["CellID"] = df_meta["CellID"].astype(str).str.replace(r"\.0$", "", regex=True)
        df_meta = df_meta.set_index("CellID")

        new_cols = [c for c in df_meta.columns if c not in adata.obs.columns]
        adata.obs = adata.obs.join(df_meta[new_cols], how="left")

        # Add per-FOV metadata
        meta_info = meta.copy()
        meta_info["FOV"] = "FOV" + meta_info["COSMX_FOV"].astype(int).astype(str).str.zfill(3)
        meta_fov = meta_info.loc[meta_info["FOV"] == fov_folder].copy()

        for col in meta_fov.columns:
            if col not in adata.obs.columns:
                adata.obs[col] = meta_fov.iloc[0][col]

        # Add explicit FOV column
        adata.obs["FOV"] = fov_folder

        all_adatas.append(adata)

    # Concatenate all FOV adatas
    adata = all_adatas[0].concatenate(
        all_adatas[1:],
        join="outer",  # keep all variables (genes)
        batch_key="FOV",  # creates a column indicating original FOV
        batch_categories=[adata.obs["FOV"].iloc[0] for adata in all_adatas]
    )

    return adata


def classify_neat1(adata):
    """
    Add NEAT1 normalized expression and categorical thresholds to adata.obs.
    Non-epithelial cells will have NaN values in the *_epi_* columns.
    """

    # locate NEAT1 gene
    neat1_index = np.where(adata.var_names == 'NEAT1')[0]
    if len(neat1_index) == 0:
        raise ValueError("NEAT1 not found in adata.var_names")
    neat1_index = neat1_index[0]

    # total cells NEAT1 expression
    if hasattr(adata.X, "toarray"):  # sparse matrix
        neat1_total = adata.X[:, neat1_index].toarray().flatten()
    else:
        neat1_total = adata.X[:, neat1_index].flatten()
    adata.obs['NEAT1_total_normalized'] = neat1_total

    # subset to epithelial cells
    epi_mask = adata.obs['label'] == 'epithelial'
    adata_epi = adata[epi_mask]

    # NEAT1 in epithelial cells (NaN for non-epithelial)
    neat1_epi = np.full(adata.n_obs, np.nan)
    if hasattr(adata_epi.X, "toarray"):
        neat1_epi[epi_mask] = adata_epi.X[:, neat1_index].toarray().flatten()
    else:
        neat1_epi[epi_mask] = adata_epi.X[:, neat1_index].flatten()
    adata.obs['NEAT1_epi_normalized'] = neat1_epi

    # ---------- median thresholds ----------
    median_total = adata.obs["NEAT1_total_normalized"].median()
    median_epi = pd.Series(neat1_epi[epi_mask]).median()

    adata.obs['median_NEAT1_total_norm'] = pd.Categorical(
        np.where(adata.obs["NEAT1_total_normalized"] > median_total, "High", "Low"),
        categories=["Low", "High"]
    )

    median_epi_cat = np.full(adata.n_obs, np.nan, dtype=object)
    median_epi_cat[epi_mask] = np.where(
        adata.obs.loc[epi_mask, "NEAT1_epi_normalized"] > median_epi, "High", "Low"
    )
    adata.obs['median_NEAT1_epi_norm'] = pd.Categorical(
        median_epi_cat, categories=["Low", "High"]
    )

    # ---------- tertile thresholds ----------
    q33_total, q66_total = adata.obs["NEAT1_total_normalized"].quantile([0.33, 0.66])
    adata.obs['tertile_NEAT1_total_norm'] = pd.cut(
        adata.obs["NEAT1_total_normalized"],
        bins=[-np.inf, q33_total, q66_total, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    # for epithelial only
    q33_epi, q66_epi = pd.Series(neat1_epi[epi_mask]).quantile([0.33, 0.66])
    tertile_epi = pd.Series(np.nan, index=adata.obs.index, dtype="object")
    tertile_epi.loc[epi_mask] = pd.cut(
        adata.obs.loc[epi_mask, "NEAT1_epi_normalized"],
        bins=[-np.inf, q33_epi, q66_epi, np.inf],
        labels=['Low', 'Medium', 'High']
    )
    adata.obs['tertile_NEAT1_epi_norm'] = pd.Categorical(
        tertile_epi, categories=['Low', 'Medium', 'High']
    )

    # diagnostics
    print(adata.obs['NEAT1_total_normalized'].describe())
    print(adata.obs['NEAT1_epi_normalized'].describe())
    print(adata.obs['tertile_NEAT1_total_norm'].value_counts())
    print(adata.obs['tertile_NEAT1_epi_norm'].value_counts(dropna=False))
    print(adata.obs['median_NEAT1_total_norm'].value_counts())
    print(adata.obs['median_NEAT1_epi_norm'].value_counts(dropna=False))

    return adata


def plot_neat1(adata, save_folder=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    # Color palette for tertiles
    palette = {"Low": "#4575b4", "Medium": "#fee090", "High": "#d73027"}

    # --- Total ---
    axes[0].set_title("NEAT1 Total Normalized")
    sns.histplot(
        data=adata.obs,
        x="NEAT1_total_normalized",
        hue="tertile_NEAT1_total_norm",
        bins=50,
        palette=palette,
        multiple="stack",
        ax=axes[0]
    )
    axes[0].axvline(adata.obs["NEAT1_total_normalized"].mean(), color="blue", linestyle="-", label="Mean")
    axes[0].axvline(adata.obs["NEAT1_total_normalized"].median(), color="red", linestyle="--", label="Median")
    axes[0].legend()

    # --- Epithelial ---
    axes[1].set_title("NEAT1 Epithelial Normalized")
    sns.histplot(
        data=adata.obs,
        x="NEAT1_epi_normalized",
        hue="tertile_NEAT1_epi_norm",
        bins=50,
        palette=palette,
        multiple="stack",
        ax=axes[1]
    )
    axes[1].axvline(adata.obs["NEAT1_epi_normalized"].mean(), color="blue", linestyle="-", label="Mean")
    axes[1].axvline(adata.obs["NEAT1_epi_normalized"].median(), color="red", linestyle="--", label="Median")
    axes[1].legend()

    plt.tight_layout()
    if save_folder:
        plt.savefig(f"{save_folder}/NEAT1_distribution.png", dpi=300)
    plt.show()


def merge_agglomerate_data_with_adata(adata, neat1_df, radius, threshold,
                                      cell_id_col=None, fov_col='FOV'):
    """
    Merge NEAT1 agglomerate data with AnnData object

    Parameters
    ----------
    adata : AnnData
        Main spatial data object
    neat1_df : pd.DataFrame
        Results from analyze_neat1_agglomerates with columns:
        cell_id, NEAT1_total, NEAT1_aggl_mols, n_agglomerates, fov
    radius : int
        Radius parameter used in analysis (for column naming)
    threshold : int
        Threshold parameter used in analysis (for column naming)
    cell_id_col : str or None
        Column name for cell ID in adata.obs. If None, uses adata.obs.index
    fov_col : str
        Column name for FOV in adata.obs

    Returns
    -------
    AnnData with new columns added to .obs
    """

    print(f"Merging NEAT1 agglomerate data (r={radius}, th={threshold})")
    print(f"NEAT1 data shape: {neat1_df.shape}")
    print(f"AnnData shape: {adata.shape}")
    print(f"Available adata.obs columns: {list(adata.obs.columns)}")

    # Create column names with parameters
    col_suffix = f"_r{radius}_th{threshold}"

    # Rename columns to include parameters
    neat1_merge = neat1_df.copy()
    neat1_merge = neat1_merge.rename(columns={
        'NEAT1_total': f'NEAT1_total{col_suffix}',
        'NEAT1_aggl_mols': f'NEAT1_aggl_mols{col_suffix}',
        'n_agglomerates': f'n_agglomerates{col_suffix}'
    })

    # Add has_aggl column
    neat1_merge[f'has_aggl{col_suffix}'] = neat1_merge[f'n_agglomerates{col_suffix}'] > 0

    # Handle cell ID - use index if cell_id_col is None or doesn't exist
    if cell_id_col is None or cell_id_col not in adata.obs.columns:
        print("Using adata.obs.index as cell identifiers")
        adata_cell_ids = adata.obs.index.astype(str)
    else:
        adata_cell_ids = adata.obs[cell_id_col].astype(str)

    # Clean NEAT1 cell_ids (remove .0 suffix as in your code)
    neat1_merge["cell_id"] = neat1_merge["cell_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Create merge keys that combine cell_id and fov for unique matching
    adata.obs['merge_key'] = adata_cell_ids + "_" + adata.obs[fov_col].astype(str)
    neat1_merge['merge_key'] = neat1_merge['cell_id'].astype(str) + "_" + neat1_merge['fov'].astype(str)

    # Check overlap before merge
    adata_keys = set(adata.obs['merge_key'])
    neat1_keys = set(neat1_merge['merge_key'])
    overlap = adata_keys.intersection(neat1_keys)

    print(f"Cells in adata: {len(adata_keys)}")
    print(f"Cells in NEAT1 data: {len(neat1_keys)}")
    print(f"Overlapping cells: {len(overlap)}")

    if len(overlap) == 0:
        print("WARNING: No overlapping cells found. Check cell_id and fov column matching.")
        print(f"Sample adata keys: {list(adata_keys)[:5]}")
        print(f"Sample NEAT1 keys: {list(neat1_keys)[:5]}")
        return adata

    # Merge data
    merge_cols = ['merge_key', f'NEAT1_total{col_suffix}', f'NEAT1_aggl_mols{col_suffix}',
                  f'n_agglomerates{col_suffix}', f'has_aggl{col_suffix}']

    adata_obs_merged = adata.obs.merge(
        neat1_merge[merge_cols],
        on='merge_key',
        how='left'
    )

    # Fill NaN values with 0 for counts and False for has_aggl
    adata_obs_merged[f'NEAT1_total{col_suffix}'] = adata_obs_merged[f'NEAT1_total{col_suffix}'].fillna(0).astype(int)
    adata_obs_merged[f'NEAT1_aggl_mols{col_suffix}'] = adata_obs_merged[f'NEAT1_aggl_mols{col_suffix}'].fillna(
        0).astype(int)
    adata_obs_merged[f'n_agglomerates{col_suffix}'] = adata_obs_merged[f'n_agglomerates{col_suffix}'].fillna(0).astype(
        int)
    adata_obs_merged[f'has_aggl{col_suffix}'] = adata_obs_merged[f'has_aggl{col_suffix}'].fillna(False)

    # Update adata.obs
    adata.obs = adata_obs_merged.set_index(adata.obs.index)

    # Remove temporary merge_key
    adata.obs = adata.obs.drop('merge_key', axis=1)

    # Summary
    n_with_data = (adata.obs[f'NEAT1_total{col_suffix}'] > 0).sum()
    n_with_aggl = (adata.obs[f'has_aggl{col_suffix}'] == True).sum()

    print(f"Successfully merged:")
    print(f"  - Cells with NEAT1 data: {n_with_data}")
    print(f"  - Cells with agglomerates: {n_with_aggl}")
    print(f"  - New columns added: {[col for col in adata.obs.columns if col_suffix in col]}")

    return adata
