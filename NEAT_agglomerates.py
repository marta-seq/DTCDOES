import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import geopandas as gpd
from shapely.geometry import Polygon, Point
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw
from tqdm import tqdm
import tifffile
import re
import sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append('/home/martinha/PycharmProjects/phd/spatial_transcriptomics/GRIDGEN')

def parse_dapi(segmentation_file):
    """Parse DAPI image file and normalize"""
    im = []
    with tifffile.TiffFile(segmentation_file) as tif:
        for page in tif.pages:
            image = page.asarray()
            im.append(image)
    im = np.asarray(im)
    img = im[-1]
    p_99 = np.percentile(img, 99)
    p_01 = np.percentile(img, 1)
    img = np.where(img > p_99, p_99, img)
    img = np.where(img < p_01, 0, img)

    v_min = img.min(axis=(0, 1), keepdims=True)
    v_max = img.max(axis=(0, 1), keepdims=True)
    img = (img - v_min) / (v_max - v_min)
    return img


def load_and_preprocess_data(fov_folder, root_directory_data):
    """Load and preprocess molecular data for one FOV"""
    folder_path_data = os.path.join(root_directory_data, fov_folder)
    file_data = [file for file in os.listdir(folder_path_data) if '__target_call_coord.csv' in file][0]
    file_data = os.path.join(folder_path_data, file_data)
    df_data = pd.read_csv(file_data)

    # Filter out control genes
    keywords_to_exclude = ['SystemControl', 'Negativ']
    pattern = '|'.join(map(re.escape, keywords_to_exclude))
    df = df_data[~df_data['target'].str.contains(pattern, case=False, regex=True)]
    df_total = df.rename(columns={'x': 'X', 'y': 'Y'})

    return df_total


def calculate_neat1_agglomerates(df_total, radius=15, threshold=3):
    """Calculate NEAT1 agglomerates using KDTree and DBSCAN clustering"""
    from gridgen import get_arrays as ga
    from gridgen import contours

    height = int(max(df_total['X'])) + 1
    width = int(max(df_total['Y'])) + 1

    print(f'n genes: {len(df_total["target"].unique())}')
    print(f'shape: {height}, {width}')
    print(f'n hits {len(df_total)}')

    # Create arrays and get NEAT1 subset
    target_dict_total = {target: index for index, target in enumerate(df_total['target'].unique())}
    array_total = ga.transform_df_to_array(df=df_total, target_dict=target_dict_total,
                                           array_shape=(height, width, len(target_dict_total))).astype(np.int8)

    df_subset_neat1, array_subset_neat1, target_indices_subset_neat1 = ga.get_subset_arrays(
        df_total, array_total, target_dict_total, target_list=['NEAT1'], target_col='target'
    )

    # Calculate neighbor counts
    CGD = contours.KDTreeContours(df_subset_neat1, contour_name='NEAT1', height=height, width=width)
    CGD.get_kdt_dist(radius=radius)
    df = CGD.kd_tree_data

    # Define agglomerates
    df["NEAT1_aggl"] = df["NEAT1_neighbor_count"] > threshold

    # DBSCAN clustering for agglomerate identification
    pos_points = df[df["NEAT1_aggl"] == True]
    if len(pos_points) > 0:
        coords = pos_points[['X', 'Y']].to_numpy()
        clustering = DBSCAN(eps=10, min_samples=3).fit(coords)
        df.loc[pos_points.index, "agglomerate_id"] = clustering.labels_
        df["agglomerate_id"] = df["agglomerate_id"].fillna(-1)
    else:
        df["agglomerate_id"] = -1

    return df


def plot_neighbor_counts(df, save_path=None, dpi=300):
    """Plot spatial distribution of NEAT1 neighbor counts"""
    plt.figure(figsize=(5, 5))
    scp = plt.scatter(
        df["X"], df["Y"],
        c=df["NEAT1_neighbor_count"],
        cmap="viridis",
        s=1,
        alpha=0.8,
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.colorbar(scp, label="NEAT1 neighbor count")
    plt.title("Spatial distribution of NEAT1 neighbor counts")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_agglomerates(df, threshold, save_path=None, dpi=300):
    """Plot NEAT1 agglomerate-positive cells"""
    plt.figure(figsize=(5, 5))
    colors = {True: "red", False: "lightgray"}

    plt.scatter(
        df["X"], df["Y"],
        c=df["NEAT1_aggl"].map(colors),
        s=1,
        alpha=0.8
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(f"NEAT1 paraspeckle-positive cells (threshold={threshold})")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_dbscan_clusters(df, save_path=None, dpi=300):
    """Plot DBSCAN clusters of NEAT1 agglomerates"""
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(
        df["X"], df["Y"],
        c=df["agglomerate_id"].astype(int),
        s=1,
        alpha=0.8,
        cmap="tab20"
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title("DBSCAN identification of NEAT1 clusters")
    plt.colorbar(scatter, label="Cluster ID")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_dapi_overlay(df, dapi_file, save_path=None, dpi=300):
    """Create DAPI overlay with NEAT1 agglomerates"""
    background_image = parse_dapi(dapi_file)
    background_image = (background_image * 255).astype(np.uint8)
    background_image = Image.fromarray(background_image, 'L')
    background_image = background_image.convert('RGB')
    draw = ImageDraw.Draw(background_image)

    target_color = {True: "red", False: "lightgreen"}

    for index, row in df.iterrows():
        x, y = row['X'], row['Y']
        fill_color = target_color[row["NEAT1_aggl"]]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill_color)

    plt.figure(figsize=(10, 10))
    plt.imshow(background_image)
    plt.axis("off")
    plt.title("DAPI overlay with NEAT1 agglomerates")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def assign_molecules_to_cells(df, geojson_path):
    """Assign molecules to cells using polygon segmentation"""
    # Load polygons
    with open(geojson_path, "r") as f:
        polygons_json = json.load(f)

    polygons = []
    cell_ids = []

    for g in polygons_json["geometries"]:
        coords = g["coordinates"][0]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        if len(coords) >= 4:
            polygons.append(Polygon(coords))
            cell_ids.append(g["cell"])

    gdf_polygons = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": polygons})

    # Molecule positions
    gdf_mols = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["X"], df["Y"]),
        crs="EPSG:4326"
    )

    # Assign molecules to polygons
    joined = gpd.sjoin(gdf_mols, gdf_polygons, how="left", predicate="within")

    return joined, gdf_polygons


def calculate_cell_summary(joined):
    """Calculate per-cell summary statistics"""
    cell_summary = (
        joined.groupby("cell_id")
        .agg(
            NEAT1_total=("target", "count"),
            NEAT1_aggl_mols=("NEAT1_aggl", "sum"),
            n_agglomerates=("agglomerate_id", lambda x: len(set([i for i in x if i != -1])))
        )
        .reset_index()
    )
    return cell_summary


def plot_cell_features(gdf_polygons, cell_summary, save_path=None, dpi=300):
    """Plot per-cell features as polygon maps"""
    gdf_polygons = gdf_polygons.merge(cell_summary, on="cell_id", how="left")
    features_to_plot = ["NEAT1_total", "NEAT1_aggl_mols", "n_agglomerates"]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, col in zip(axes, features_to_plot):
        gdf_polygons.plot(
            column=col,
            cmap="Reds",
            legend=True,
            ax=ax,
            linewidth=0.5,
            edgecolor="black"
        )
        ax.invert_yaxis()
        ax.axis("equal")
        ax.set_title(f"Per-cell {col}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def process_single_fov(fov_folder, root_directory_data, radius=15, threshold=3,
                       save_folder=None, dpi=300):
    """Process a single FOV for NEAT1 agglomerate analysis"""

    print(f"Processing FOV: {fov_folder}")

    # Create FOV-specific save folder
    if save_folder:
        fov_save_folder = os.path.join(save_folder, fov_folder)
        os.makedirs(fov_save_folder, exist_ok=True)
    else:
        fov_save_folder = None

    # Load data
    df_total = load_and_preprocess_data(fov_folder, root_directory_data)

    # Calculate agglomerates
    df = calculate_neat1_agglomerates(df_total, radius=radius, threshold=threshold)

    # Create plots
    if fov_save_folder:
        neighbor_path = os.path.join(fov_save_folder, "neat1_neighbor_counts.png")
        aggl_path = os.path.join(fov_save_folder, "neat1_agglomerates.png")
        cluster_path = os.path.join(fov_save_folder, "dbscan_clusters.png")
        dapi_path = os.path.join(fov_save_folder, "dapi_overlay.png")
        cell_path = os.path.join(fov_save_folder, "cell_features.png")
    else:
        neighbor_path = aggl_path = cluster_path = dapi_path = cell_path = None

    plot_neighbor_counts(df, neighbor_path, dpi)
    plot_agglomerates(df, threshold, aggl_path, dpi)
    plot_dbscan_clusters(df, cluster_path, dpi)

    # DAPI overlay
    dapi_file = f'/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx_data/S0/S0/20230628_151317_S4/CellStatsDir/Morphology2D/20230628_151317_S4_C902_P99_N99_F{fov_folder[3:]}.TIF'
    if os.path.exists(dapi_file):
        plot_dapi_overlay(df, dapi_file, dapi_path, dpi)

    # Cell assignment and summary
    geojson_path = f'/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx/Baysor-master/OUTPUT/cellpose_masks/{fov_folder}/segmentation_polygons.json'
    if os.path.exists(geojson_path):
        joined, gdf_polygons = assign_molecules_to_cells(df, geojson_path)
        cell_summary = calculate_cell_summary(joined)
        cell_summary["fov"] = fov_folder

        plot_cell_features(gdf_polygons, cell_summary, cell_path, dpi)

        return cell_summary
    else:
        print(f"Warning: Segmentation file not found for {fov_folder}")
        return None


def analyze_neat1_agglomerates(fov_folders, root_directory_data, radius=15, threshold=3,
                               save_folder="results/neat1_agglomerates", dpi=300):
    """
    Main function to analyze NEAT1 agglomerates across multiple FOVs

    Parameters
    ----------
    fov_folders : list
        List of FOV folder names to process
    root_directory_data : str
        Root directory containing molecular data
    radius : int, default 15
        Radius for neighbor search in pixels
    threshold : int, default 3
        Threshold for defining agglomerates
    save_folder : str
        Directory to save results and plots
    dpi : int, default 300
        DPI for saved figures

    Returns
    -------
    pd.DataFrame
        Combined results across all FOVs with columns:
        cell_id, NEAT1_total, NEAT1_aggl_mols, n_agglomerates, fov
    """

    print(f"=== NEAT1 AGGLOMERATE ANALYSIS ===")
    print(f"Parameters: radius={radius}, threshold={threshold}")
    print(f"Processing {len(fov_folders)} FOVs")
    print(f"Save folder: {save_folder}")

    # Create main save directory
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    all_cell_dfs = []

    for fov_folder in tqdm(fov_folders, desc="Processing FOVs"):
        try:
            cell_summary = process_single_fov(
                fov_folder, root_directory_data, radius, threshold,
                save_folder, dpi
            )
            if cell_summary is not None:
                all_cell_dfs.append(cell_summary)
        except Exception as e:
            print(f"Error processing {fov_folder}: {e}")
            continue

    if all_cell_dfs:
        # Combine all results
        cell_W_NEAT = pd.concat(all_cell_dfs, ignore_index=True)

        # Save combined results
        if save_folder:
            results_path = os.path.join(save_folder, "neat1_agglomerates_summary.csv")
            cell_W_NEAT.to_csv(results_path, index=False)
            print(f"Combined results saved to: {results_path}")

        print(f"Analysis complete. Processed {len(cell_W_NEAT)} cells across {len(all_cell_dfs)} FOVs")
        return cell_W_NEAT
    else:
        print("No successful FOV processing")
        return None