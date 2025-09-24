import os

import scanpy as sc

from utils.adata_utils import create_anndata, classify_neat1, plot_neat1, merge_agglomerate_data_with_adata
from DGE import (de_analysis, de_analysis_per_fov,
                 mixed_effects_de_analysis, compare_with_naive_analysis)
import neighboors as NN
import NEAT_agglomerates as NA
if __name__ == '__main__':
    folders = [
        "results/de_analysis",
        "results/de_analysis_per_fov",
        "results/mixed_effects_de",
        "results/comparison_naive",
        "results/per_celltype_de",
        "results/per_fov_meta_analysis"
    ]

    for f in folders:
        os.makedirs(f, exist_ok=True)  # creates each folder if it doesn't already exist


    label_file = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx/Baysor-master/OUTPUT/Meta_data_CellPose_Baysor0.7_epithelial.csv'
    meta_file = "/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx_data/S0/cosmx_metadata.xlsx"
    root_directory = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx/Baysor-master/OUTPUT/NCVS'
    root_directory_data = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx_data/S0/S0/20230628_151317_S4/AnalysisResults/iz38iruwno'

    fov_folders = [folder_name for folder_name in os.listdir(root_directory) if
                   os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith("FOV")]

    # create cell anndata
    adata = create_anndata(label_file, meta_file, fov_folders)
    # preprocessing
    adata = adata[~adata.obs["label"].isna()].copy()
    adata.raw = adata  # stores raw counts before normalization/log
    sc.pp.normalize_total(adata, target_sum=1e4)  # counts per 10k
    sc.pp.log1p(adata)  # log-transform
    adata.write_h5ad('results/adata_cells.h5ad')

    # added columns to adata.obs:
    # median_NEAT1_total_norm, median_NEAT1_epi_norm, tertile_NEAT1_total_norm, tertile_NEAT1_epi_norm
    adata = classify_neat1(adata)
    plot_neat1(adata, save_folder='results')

    # add agglomerate information
    r = 15
    th = 3
    agg_df = NA.analyze_neat1_agglomerates(
        fov_folders=fov_folders,
        root_directory_data=root_directory_data,
        radius=r,  # your r parameter
        threshold=th,  # your threshold parameter
        save_folder=f"results/neat1_agglomerates_{r}_{th}",
        dpi=400  # high resolution
    )
    merge_agglomerate_data_with_adata(adata, agg_df, r, th, cell_id_col='cell_ID', fov_col='FOV')
    # Differential expression analyses of epithelial cells basedon NEAT1 expression
    for column in ["tertile_NEAT1_epi_norm","median_NEAT1_epi_norm",
                   f"has_aggl_r{r}_th{th}"]:

        result_df = de_analysis(adata, column, save_folder='results/de_analysis')

        results = de_analysis_per_fov(adata, column, fov_column='FOV', save_folder='results/de_analysis_per_fov')

        result_df = mixed_effects_de_analysis(adata, column, patient_column='FOV', n_jobs=4,
                                         save_folder='results/mixed_effects_de')
        comparison = compare_with_naive_analysis(adata, column, patient_column='FOV',
                                             save_folder='results/comparison_naive')

    # neighborhoods
    r = 200  # radius in pixels
    neighborhoods = NN.get_epithelial_neighborhoods(adata, radius=r, fov_col='FOV') # counts of each neighborhoods
    cols_to_merge = [
        "tertile_NEAT1_epi_norm",
        "median_NEAT1_epi_norm",
        f"has_aggl_r{r}_th{th}"
    ]

    # Important: ensure you have the cell identifier (index) as a column
    neighborhoods = neighborhoods.merge(
        adata.obs[cols_to_merge],
        left_on="cell_index",   # or whatever column in neighborhoods uniquely matches adata.obs.index
        right_index=True,
        how="left"
    )
    for column in ["tertile_NEAT1_epi_norm", "median_NEAT1_epi_norm"]:

        neat1_summary = NN.create_neighborhood_heatmaps(neighborhoods, column=column)
        # not accounting per fov
        save_path = f'results/neighborhood_composition_r{r}/no_fov'
        os.makedirs(save_path, exist_ok=True)

        test_results = NN.test_neighborhood_composition_differences(
            neighborhoods,
            column=column,
            save_folder= save_path,
            account_for_fov=False,
            fov_col='FOV'
        )
        NN.create_summary_plots(test_results, title=f"{column}_comparisons",
                                save_path=f"{save_path}/{column}_neigh_counts.png")

        # accounting FOV
        save_path = f'results/neighborhood_composition_r{r}/account_fov'
        os.makedirs(save_path, exist_ok=True)

        test_results = NN.test_neighborhood_composition_differences(
            neighborhoods,
            column=column,
            save_folder= save_path,
            account_for_fov=True,
            fov_col='FOV'
        )
        NN.create_summary_plots(test_results, title=f"{column}_comparisons",
                                save_path=f"{save_path}/{column}_neigh_counts.png")
        # mixed effects
        NN.test_neighborhood_composition_differences_mixed_effects(
            neighborhoods,
            column=column,
            fov_col='FOV',
            save_folder=save_path
        )

        ##############################
        # per neighborhood expression


        # add radius to the folder. some results are wrng in the folders

        # per cell type

        for cell_type in adata.obs['label'].unique():
            print(f"==============={cell_type}==============")
            cell_neighborhoods = NN.get_celltype_specific_neighborhood_expression(adata, cell_type,
                                                                                  radius=r,
                                                                                  neat1_col=column)
            # Regular DE (pooled across all FOVs)
            save_path = f'results/neighborhood_celltype_r{r}/{cell_type}/no_fov'
            os.makedirs(save_path, exist_ok=True)
            cell_de = NN.run_de_on_neighborhoods(cell_neighborhoods, column,
                                                 save_folder=save_path)

            # FOV-adjusted DE (aggregate by FOV first)
            save_path = f'results/neighborhood_celltype_r{r}/{cell_type}/account_fov'
            os.makedirs(save_path, exist_ok=True)
            cell_de_fov = NN.run_de_on_neighborhoods(cell_neighborhoods, column,
                                                     account_for_fov=True, fov_col='FOV',
                                                     save_folder=save_path)

            # Mixed-effects DE (FOV as random effect)
            cell_de_mixed = NN.run_de_on_neighborhoods_mixed_effects(cell_neighborhoods, column,
                                                                     fov_col='FOV',
                                                                     save_folder=save_path)

            # Then add visualizations
            de_results = {
                'pooled': cell_de,
                'fov_adjusted': cell_de_fov,
                'mixed_effects': cell_de_mixed
            }

            NN.create_neighborhood_de_report(cell_type, cell_neighborhoods, de_results, column,
                                             save_folder=save_path)






    ### survival analysis
    # pass for now