import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from anndata import AnnData
from scipy.sparse import issparse
from IPython.display import display
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests
import warnings
import itertools

warnings.filterwarnings('ignore')


def process_single_fov(adata, fov, radius, x_col, y_col, cell_type_col, fov_col, all_cell_types):
    """
    Process neighborhoods for a single FOV
    """

    # Get data for this FOV
    if fov is not None:
        fov_data = adata.obs[adata.obs[fov_col] == fov].copy()
        print(f"\n--- {fov}: {len(fov_data)} cells ---")
    else:
        fov_data = adata.obs.copy()
        print(f"\nProcessing {len(fov_data)} cells")

    if len(fov_data) < 10:
        print("Too few cells, skipping")
        return None

    # Get epithelial cells only for analysis
    epithelial_cells = fov_data[fov_data[cell_type_col] == 'epithelial'].copy()
    print(f"Epithelial cells: {len(epithelial_cells)}")

    if len(epithelial_cells) == 0:
        print("No epithelial cells, skipping")
        return None

    # Get coordinates for ALL cells (for finding neighbors)
    all_coords = fov_data[[x_col, y_col]].values
    epithelial_coords = epithelial_cells[[x_col, y_col]].values

    # Build spatial index with ALL cells
    nbrs = NearestNeighbors(radius=radius).fit(all_coords)

    # For each epithelial cell, find neighbors among ALL cells
    neighbor_indices = nbrs.radius_neighbors(epithelial_coords, return_distance=False)

    results = []

    for i, epithelial_neighbors in enumerate(neighbor_indices):

        # Get the epithelial cell info
        epi_cell = epithelial_cells.iloc[i]
        epi_index_in_fov = epithelial_cells.index[i]  # Index in fov_data

        # Find which index this epithelial cell has in the full fov_data
        epi_position_in_all = fov_data.index.get_loc(epi_index_in_fov)

        # Remove the epithelial cell itself from its neighbors
        neighbors_excluding_self = [idx for idx in epithelial_neighbors if idx != epi_position_in_all]

        if len(neighbors_excluding_self) == 0:
            continue  # Skip isolated cells

        # Get neighbor cell types
        neighbor_cell_types = fov_data.iloc[neighbors_excluding_self][cell_type_col]
        neighbor_counts = neighbor_cell_types.value_counts()

        # Create result row
        result_row = {
            'cell_index': epi_index_in_fov,
            'x_coord': epi_cell[x_col],
            'y_coord': epi_cell[y_col],
            'n_neighbors': len(neighbors_excluding_self)
        }

        # Add FOV if provided
        if fov is not None:
            result_row['FOV'] = fov

        # Add NEAT1 info if available
        neat1_cols = ['NEAT1_has_aggl', 'tertile_NEAT1_norm', 'median_NEAT1_norm']
        for col in neat1_cols:
            if col in epi_cell.index:
                result_row[col] = epi_cell[col]

        # Add counts for each cell type
        for cell_type in all_cell_types:
            result_row[f'count_{cell_type}'] = neighbor_counts.get(cell_type, 0)

        results.append(result_row)

    if results:
        df = pd.DataFrame(results)
        print(f"Analyzed {len(df)} epithelial cells")
        print(f"Average neighbors: {df['n_neighbors'].mean():.1f}")
        return df
    else:
        print("No valid neighborhoods found")
        return None

### DOES NEAT correlateswith neighboors

def get_epithelial_neighborhoods(adata, radius=50, x_col='x', y_col='y',
                                 cell_type_col='label', fov_col=None):
    """
    Get neighborhood cell type counts for epithelial cells only
    Returns simple DataFrame with counts for each cell type
    """

    print(f"=== EPITHELIAL NEIGHBORHOODS (radius={radius}) ===")

    # Get all unique cell types
    all_cell_types = sorted(adata.obs[cell_type_col].unique())
    print(f"Cell types: {all_cell_types}")

    # Process each FOV separately or all together
    if fov_col:
        fovs = adata.obs[fov_col].unique()
        print(f"Processing {len(fovs)} FOVs")
        all_results = []

        for fov in fovs:
            fov_results = process_single_fov(adata, fov, radius, x_col, y_col,
                                             cell_type_col, fov_col, all_cell_types)
            if fov_results is not None:
                all_results.append(fov_results)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return None
    else:
        return process_single_fov(adata, None, radius, x_col, y_col,
                                  cell_type_col, None, all_cell_types)

def create_neighborhood_heatmaps(
    neighborhoods_df,
    column='median_NEAT1_norm',
    save_folder='results/neighborhood_heatmaps'
):
    """
    Create heatmaps of mean neighborhood composition per FOV
    for all categories found in `column`.
    If the column has exactly two categories, also plot a
    difference heatmap (cat2 - cat1).
    """
    print(f"=== CREATING NEIGHBORHOOD HEATMAPS for '{column}' ===")

    os.makedirs(save_folder, exist_ok=True)

    # Cell-type count/proportion columns
    count_cols = [c for c in neighborhoods_df.columns if c.startswith('count_')]
    cell_types = [c.replace('count_', '') for c in count_cols]

    # Convert counts to percentages
    for c in count_cols:
        pct_col = c.replace('count_', 'pct_')
        neighborhoods_df[pct_col] = (
            neighborhoods_df[c] / neighborhoods_df['n_neighbors'] * 100
        )
    pct_cols = [c.replace('count_', 'pct_') for c in count_cols]

    # All unique categories in the grouping column
    categories = neighborhoods_df[column].dropna().unique()
    categories = sorted(categories, key=lambda x: str(x))
    print(f"Found categories: {categories}")

    # Build summary matrix: mean % per FOV and category
    summary = (
        neighborhoods_df
        .groupby(['FOV', column])[pct_cols].mean()
        .reset_index()
    )
    summary.rename(columns={f"pct_{ct}": ct for ct in cell_types}, inplace=True)

    # Save summary
    csv_path = os.path.join(save_folder, f"neighborhood_summary_by_{column}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")

    # Plot heatmaps for each category
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(5*n_cats, 6), squeeze=False)
    axes = axes[0]

    for ax, cat in zip(axes, categories):
        sub = summary[summary[column] == cat].set_index('FOV')[cell_types]
        sns.heatmap(sub, ax=ax, cmap='viridis', vmin=0, vmax=100,
                    cbar_kws={'label': '% of neighbors'})
        ax.set_title(f"{column} = {cat}")
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('FOV')

    plt.tight_layout()
    heatmap_path = os.path.join(save_folder, f"neighborhood_heatmaps_{column}.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"Saved heatmaps to {heatmap_path}")
    plt.close()

    # If exactly two categories, create difference heatmap
    if n_cats == 2:
        cat1, cat2 = categories
        m1 = (summary[summary[column] == cat1]
              .set_index('FOV')[cell_types]
              .reindex(summary['FOV'].unique()))
        m2 = (summary[summary[column] == cat2]
              .set_index('FOV')[cell_types]
              .reindex(summary['FOV'].unique()))
        diff = m2 - m1  # cat2 minus cat1

        plt.figure(figsize=(6, 6))
        sns.heatmap(diff, cmap='coolwarm', center=0,
                    cbar_kws={'label': f'{cat2} - {cat1} (% neighbors)'})
        plt.title(f"Difference Heatmap: {cat2} – {cat1}")
        plt.xlabel('Cell Type')
        plt.ylabel('FOV')
        diff_path = os.path.join(save_folder,
                                 f"neighborhood_difference_{column}.png")
        plt.tight_layout()
        plt.savefig(diff_path, dpi=300)
        print(f"Saved difference heatmap to {diff_path}")
        plt.close()

    return summary


def test_neighborhood_composition_differences(
        neighborhoods_df,
        column='median_NEAT1_norm',
        save_folder='results/neighborhood_stats',
        account_for_fov=False,
        fov_col='FOV'
):
    """
    Test differences in neighborhood composition (% of each cell type)
    across categories in `column` using Mann–Whitney U (pairwise).

    Parameters
    ----------
    neighborhoods_df : pd.DataFrame
        Data with columns like 'pct_<celltype>' and the grouping column.
    column : str
        Column name with categorical groups (e.g. 'median_NEAT1_norm').
    save_folder : str
        Directory to save CSV results.
    account_for_fov : bool
        If True, aggregate by FOV first to avoid pseudoreplication.
    fov_col : str
        Column name for FOV/sample identifier.

    Returns
    -------
    pd.DataFrame with test statistics and adjusted p-values.
    """
    print(f"\n=== NEIGHBORHOOD COMPOSITION TESTS for '{column}' ===")
    print(f"FOV accounting: {account_for_fov}")
    os.makedirs(save_folder, exist_ok=True)

    # FOV aggregation if requested
    if account_for_fov:
        if fov_col not in neighborhoods_df.columns:
            raise ValueError(f"FOV column '{fov_col}' not found in data")

        pct_cols = [c for c in neighborhoods_df.columns if c.startswith('pct_')]
        agg_dict = {col: 'mean' for col in pct_cols}

        # Aggregate by FOV and group, taking mean percentages per FOV
        df_agg = neighborhoods_df.groupby([fov_col, column]).agg(agg_dict).reset_index()
        print(f"Aggregated from {len(neighborhoods_df)} cells to {len(df_agg)} FOV-group combinations")
        working_df = df_agg
    else:
        working_df = neighborhoods_df

    pct_cols = [c for c in working_df.columns if c.startswith('pct_')]
    cell_types = [c.replace('pct_', '') for c in pct_cols]
    groups = working_df[column].dropna().unique()

    print(f"Groups found: {list(groups)}")
    results = []

    # All pairwise combinations of groups
    for g1, g2 in itertools.combinations(groups, 2):
        df1 = working_df[working_df[column] == g1]
        df2 = working_df[working_df[column] == g2]

        for ct in cell_types:
            col = f"pct_{ct}"
            v1 = df1[col].dropna()
            v2 = df2[col].dropna()

            # Skip if too few samples
            min_n = 3 if account_for_fov else 5  # Lower threshold for FOV-aggregated data
            if len(v1) < min_n or len(v2) < min_n:
                continue

            stat, pval = mannwhitneyu(v1, v2, alternative='two-sided')
            effect = v1.mean() - v2.mean()

            results.append({
                'cell_type': ct,
                'group1': g1,
                'group2': g2,
                'mean_group1': v1.mean(),
                'std_group1': v1.std(),
                'n_group1': len(v1),
                'mean_group2': v2.mean(),
                'std_group2': v2.std(),
                'n_group2': len(v2),
                'effect_size': effect,
                'mann_whitney_stat': stat,
                'p_value': pval
            })

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if not results_df.empty:
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_adjusted'] = p_adj

        def stars(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        results_df['significance'] = results_df['p_adjusted'].apply(stars)

        # Print significant hits
        sig_hits = results_df[results_df['p_adjusted'] < 0.05]
        print(f"\nSignificant results ({len(sig_hits)}):")
        for _, r in sig_hits.iterrows():
            print(
                f"{r['cell_type']} ({r['group1']} vs {r['group2']}): "
                f"Effect={r['effect_size']:+.1f}%, p_adj={r['p_adjusted']:.4g} {r['significance']}"
            )
    else:
        print("No valid comparisons (not enough samples).")

    # Save results
    suffix = "_fov_adjusted" if account_for_fov else "_cell_level"
    out_path = os.path.join(save_folder, f"neighborhood_tests_{column}{suffix}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    return results_df


def test_neighborhood_composition_differences_mixed_effects(
        neighborhoods_df,
        column='median_NEAT1_norm',
        fov_col='FOV',
        save_folder='results/neighborhood_stats'
):
    """
    Test differences in neighborhood composition using mixed-effects models
    with FOV as random effect.

    Requires: pip install statsmodels

    Parameters
    ----------
    neighborhoods_df : pd.DataFrame
        Data with columns like 'pct_<celltype>' and the grouping column.
    column : str
        Column name with categorical groups.
    fov_col : str
        Column name for FOV/sample identifier (random effect).
    save_folder : str
        Directory to save CSV results.

    Returns
    -------
    pd.DataFrame with mixed-effects test results.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import mixedlm
    except ImportError:
        raise ImportError("Please install statsmodels: pip install statsmodels")

    print(f"\n=== MIXED-EFFECTS NEIGHBORHOOD TESTS for '{column}' ===")
    os.makedirs(save_folder, exist_ok=True)

    pct_cols = [c for c in neighborhoods_df.columns if c.startswith('pct_')]
    cell_types = [c.replace('pct_', '') for c in pct_cols]

    results = []

    for ct in cell_types:
        pct_col = f"pct_{ct}"

        # Skip if insufficient data
        if neighborhoods_df[pct_col].isna().sum() > len(neighborhoods_df) * 0.5:
            continue

        try:
            # Fit mixed-effects model: outcome ~ group + (1|FOV)
            formula = f"{pct_col} ~ C({column})"
            model = mixedlm(formula, neighborhoods_df, groups=neighborhoods_df[fov_col])
            result = model.fit(reml=False)  # Use ML for model comparison

            # Extract group comparisons from model
            params = result.params
            pvals = result.pvalues

            # Get group means for effect sizes
            group_means = neighborhoods_df.groupby(column)[pct_col].mean()

            for param_name, pval in pvals.items():
                if param_name.startswith('C('):  # Group comparison parameters
                    # Extract group names from parameter (simplified)
                    effect_size = params[param_name]

                    results.append({
                        'cell_type': ct,
                        'parameter': param_name,
                        'effect_size': effect_size,
                        'p_value': pval,
                        'model_aic': result.aic,
                        'n_obs': result.nobs,
                        'n_groups': len(neighborhoods_df[fov_col].unique())
                    })

        except Exception as e:
            print(f"Failed to fit model for {ct}: {e}")
            continue

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Multiple testing correction
        _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_adjusted'] = p_adj

        def stars(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        results_df['significance'] = results_df['p_adjusted'].apply(stars)

        # Print significant results
        sig_hits = results_df[results_df['p_adjusted'] < 0.05]
        print(f"\nSignificant mixed-effects results ({len(sig_hits)}):")
        for _, r in sig_hits.iterrows():
            print(f"{r['cell_type']}: Effect={r['effect_size']:+.2f}%, "
                  f"p_adj={r['p_adjusted']:.4g} {r['significance']}")

    # Save results
    out_path = os.path.join(save_folder, f"neighborhood_tests_{column}_mixed_effects.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Mixed-effects results saved to {out_path}")

    return results_df










##############################################################
##############################################################
###############################################################
# DE on neighborhoods
def run_per_fov_de_meta_analysis(neighborhood_adata, group_col='median_NEAT1_norm', save_folder=None):
    """
    Run DE analysis within each FOV separately, then meta-analyze results
    Includes log fold changes, percentage metrics, dotplots
    """

    print(f"=== PER-FOV DE ANALYSIS WITH META-ANALYSIS ===")
    print(f"Grouping by: {group_col}")

    if group_col not in neighborhood_adata.obs.columns:
        print(f"ERROR: Column {group_col} not found")
        return None

    if 'FOV' not in neighborhood_adata.obs.columns:
        print("ERROR: FOV column not found")
        return None

    # Convert to categorical if needed
    if neighborhood_adata.obs[group_col].dtype == 'bool':
        neighborhood_adata.obs[group_col] = neighborhood_adata.obs[group_col].astype(str).astype('category')
    elif not hasattr(neighborhood_adata.obs[group_col], 'cat'):
        neighborhood_adata.obs[group_col] = neighborhood_adata.obs[group_col].astype('category')

    # Show overall group sizes
    overall_counts = neighborhood_adata.obs[group_col].value_counts()
    print(f"Overall group sizes: {overall_counts.to_dict()}")

    fovs = neighborhood_adata.obs['FOV'].unique()
    print(f"Processing {len(fovs)} FOVs")

    # Check FOV-group combinations
    fov_group_summary = neighborhood_adata.obs.groupby(['FOV', group_col]).size().unstack(fill_value=0)
    print(f"\nFOV-group distribution:")
    print(fov_group_summary)

    # Run DE for each FOV
    fov_results = []
    successful_fovs = []

    for fov in fovs:
        print(f"\n--- Analyzing FOV: {fov} ---")

        fov_subset = neighborhood_adata[neighborhood_adata.obs['FOV'] == fov].copy()
        group_counts = fov_subset.obs[group_col].value_counts()

        print(f"Group sizes in {fov}: {group_counts.to_dict()}")

        # Check if FOV has sufficient cells in each group
        if len(group_counts) < 2 or min(group_counts) < 5:
            print(f"Skipping {fov}: insufficient cells per group (min 5 required)")
            continue

        try:
            # Run DE analysis for this FOV
            sc.tl.rank_genes_groups(fov_subset, group_col, method='wilcoxon', pts=True)
            sc.tl.filter_rank_genes_groups(fov_subset, groupby=group_col, min_fold_change=1)

            # Extract results
            result = fov_subset.uns['rank_genes_groups_filtered']

            # Parse results into structured format
            fov_de_results = []

            for group in result['names'].dtype.names:
                notna_idxs = pd.Series(result['names'][group]).notna()
                genes_group = pd.Series(result['names'][group]).dropna()

                for i, gene in enumerate(genes_group):
                    if pd.isna(gene):
                        continue

                    idx = notna_idxs[notna_idxs].index[i]

                    fov_de_results.append({
                        'FOV': fov,
                        'Gene': gene,
                        'Group': group,
                        'logFC': result['logfoldchanges'][group][idx],
                        'pval': result['pvals'][group][idx],
                        'padj': result['pvals_adj'][group][idx],
                        'perc_pos_cells': result['pts'][group].loc[gene],
                        'perc_pos_cells_others': result['pts_rest'][group].loc[gene],
                        'perc_diff': result['pts'][group].loc[gene] - result['pts_rest'][group].loc[gene]
                    })

            if fov_de_results:
                fov_results.extend(fov_de_results)
                successful_fovs.append(fov)
                print(f"Successfully analyzed {fov}: {len(fov_de_results)} significant gene-group pairs")
            else:
                print(f"No significant results in {fov}")

        except Exception as e:
            print(f"Error analyzing {fov}: {e}")
            continue

    if not fov_results:
        print("No successful FOV analyses")
        return None

    print(f"\nSuccessfully analyzed {len(successful_fovs)} FOVs")
    print(f"Total significant gene-group pairs across all FOVs: {len(fov_results)}")

    # Convert to DataFrame for meta-analysis
    fov_df = pd.DataFrame(fov_results)
    fov_df['FC_percDiff'] = fov_df['logFC'] * fov_df['perc_diff']

    # Meta-analysis: combine results for each gene-group combination
    print(f"\n=== META-ANALYSIS ===")

    meta_results = []

    for (gene, group), gene_group_data in fov_df.groupby(['Gene', 'Group']):
        if len(gene_group_data) < 2:  # Need results from multiple FOVs
            continue

        # Combine p-values using Fisher's method
        p_values = gene_group_data['pval'].values
        try:
            combined_stat, combined_pval = combine_pvalues(p_values, method='fisher')
        except:
            combined_pval = 1.0

        # Calculate summary statistics
        mean_logfc = gene_group_data['logFC'].mean()
        std_logfc = gene_group_data['logFC'].std()
        median_logfc = gene_group_data['logFC'].median()

        mean_perc_pos = gene_group_data['perc_pos_cells'].mean()
        mean_perc_others = gene_group_data['perc_pos_cells_others'].mean()
        mean_perc_diff = gene_group_data['perc_diff'].mean()

        # Direction consistency
        positive_effects = (gene_group_data['logFC'] > 0).sum()
        negative_effects = (gene_group_data['logFC'] < 0).sum()
        direction_consistent = max(positive_effects, negative_effects) >= len(gene_group_data) * 0.7

        meta_results.append({
            'Gene': gene,
            'Group': group,
            'n_fovs': len(gene_group_data),
            'mean_logFC': mean_logfc,
            'median_logFC': median_logfc,
            'std_logFC': std_logfc,
            'combined_pval': combined_pval,
            'mean_perc_pos_cells': mean_perc_pos,
            'mean_perc_pos_cells_others': mean_perc_others,
            'mean_perc_diff': mean_perc_diff,
            'mean_FC_percDiff': gene_group_data['FC_percDiff'].mean(),
            'direction_consistent': direction_consistent,
            'n_positive_fovs': positive_effects,
            'n_negative_fovs': negative_effects,
            'supporting_fovs': gene_group_data['FOV'].tolist(),
            'individual_logfc': gene_group_data['logFC'].tolist()
        })

    meta_df = pd.DataFrame(meta_results)

    if len(meta_df) == 0:
        print("No genes found in multiple FOVs")
        return {'fov_results': fov_df, 'meta_results': None}

    # Multiple testing correction on combined p-values
    _, meta_df['combined_padj'], _, _ = multipletests(meta_df['combined_pval'], method='fdr_bh')
    meta_df = meta_df.sort_values('combined_padj')
    meta_df.to_csv(os.path.join(save_folder, f"meta_analysis_{group_col}.csv"), index=False)

    # Filter for significance and sort
    significant_meta = meta_df[meta_df['combined_padj'] < 0.05].copy()
    significant_meta = significant_meta.sort_values('mean_FC_percDiff', ascending=False)

    print(f"Genes significant after meta-analysis (padj < 0.05): {len(significant_meta)}")

    if len(significant_meta) > 0:
        print(f"\nTop 10 meta-analysis results:")
        display_cols = ['Gene', 'Group', 'n_fovs', 'mean_logFC', 'combined_padj',
                        'mean_perc_diff', 'direction_consistent']
        print(significant_meta[display_cols].head(10))

        # Create visualizations
        create_meta_analysis_plots(neighborhood_adata, significant_meta, fov_df,
                                   group_col, successful_fovs,
                                   save_folder = save_folder)

    return {
        'fov_results': fov_df,
        'meta_results': significant_meta,
        'all_meta_results': meta_df,
        'successful_fovs': successful_fovs
    }


def create_meta_analysis_plots(neighborhood_adata, significant_meta, fov_df, group_col, successful_fovs,
                               save_folder = None):
    """Create plots for meta-analysis results"""

    print(f"\n=== CREATING VISUALIZATIONS ===")

    if len(significant_meta) == 0:
        print("No significant results to plot")
        return

    # Create subset for plotting (only successful FOVs)
    plot_adata = neighborhood_adata[neighborhood_adata.obs['FOV'].isin(successful_fovs)].copy()

    # Plot 1: Dotplot of top significant genes
    top_genes = significant_meta.head(50)['Gene'].unique().tolist()
    available_genes = [g for g in top_genes if g in plot_adata.var_names]

    if len(available_genes) > 0:
        print(f"Creating dotplot for top {len(available_genes)} genes")
        sc.pl.dotplot(plot_adata, available_genes, groupby=group_col,
                      swap_axes=True, save = '_top_meta_genes.png')
        plt.title('Top Meta-Analysis Genes')
        plt.show()

    # Plot 2: Specific genes dotplot
    specific_genes = ['NEAT1', 'VEGFA', 'MALAT1', 'CD55', 'IL6', 'TNF', 'CXCL8']
    available_specific = [g for g in specific_genes if g in plot_adata.var_names]

    # set the path in scanpy
    sc.settings.figdir = save_folder if save_folder else '.'
    if len(available_specific) > 0:
        print(f"Creating dotplot for specific genes: {available_specific}")
        sc.pl.dotplot(plot_adata, available_specific,
                      groupby=group_col, swap_axes=True,
                      save = '_specific_genes.png')
        plt.title('Specific Genes of Interest')
        plt.show()

    # Plot 3: Meta-analysis summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Effect size vs number of FOVs
    axes[0, 0].scatter(significant_meta['n_fovs'], significant_meta['mean_logFC'],
                       alpha=0.6, s=60)
    axes[0, 0].set_xlabel('Number of FOVs')
    axes[0, 0].set_ylabel('Mean Log Fold Change')
    axes[0, 0].set_title('Effect Size vs FOV Support')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Direction consistency
    consistent = significant_meta['direction_consistent']
    colors = ['red' if c else 'gray' for c in consistent]
    axes[0, 1].scatter(significant_meta['mean_logFC'], -np.log10(significant_meta['combined_padj']),
                       c=colors, alpha=0.6, s=60)
    axes[0, 1].set_xlabel('Mean Log Fold Change')
    axes[0, 1].set_ylabel('-log10(Combined P-value)')
    axes[0, 1].set_title('Volcano Plot\n(Red = Direction Consistent)')
    axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # FOV consistency histogram
    axes[1, 0].hist(significant_meta['n_fovs'], bins=range(2, significant_meta['n_fovs'].max() + 2),
                    alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Number of Supporting FOVs')
    axes[1, 0].set_ylabel('Number of Genes')
    axes[1, 0].set_title('FOV Support Distribution')

    # Effect size variability
    axes[1, 1].scatter(significant_meta['mean_logFC'], significant_meta['std_logFC'],
                       alpha=0.6, s=60)
    axes[1, 1].set_xlabel('Mean Log Fold Change')
    axes[1, 1].set_ylabel('Standard Deviation of Log Fold Change')
    axes[1, 1].set_title('Effect Size Consistency Across FOVs')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'meta_analysis_summary.png'), dpi=300)
    plt.show()


def create_summary_plots(results_df, title=None, save_path=None):
    """
    Create a horizontal bar plot of effect sizes and significance
    from the pairwise test results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from test_neighborhood_differences (one row per comparison).
    title : str, optional
        Overall plot title.
    save_path : str, optional
        Path to save the figure (e.g. 'results/summary_effects.png').
    """
    if results_df.empty:
        print("No results to plot.")
        return

    # Sort by effect size for clearer visualization
    plot_df = results_df.sort_values("effect_size")

    # Label for y-axis includes cell type and group comparison
    plot_df["label"] = (
        plot_df["cell_type"] + " (" + plot_df["group1"] + " vs " + plot_df["group2"] + ")"
    )

    # Color bars by significance
    colors = ["red" if p < 0.05 else "gray" for p in plot_df["p_adjusted"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.3)))
    ax.barh(plot_df["label"], plot_df["effect_size"], color=colors, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Effect Size (% difference)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Neighborhood Composition Differences\n(Red = FDR < 0.05)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()

# difference but expression
def get_celltype_specific_neighborhood_expression(adata, target_cell_type,
                                                  radius=50,
                                                  x_col='x', y_col='y',
                                                  cell_type_col='label',
                                                  neat1_col= 'median_NEAT1_norm',
                                                  fov_col='FOV'):
    """
    Function 3: Get neighborhood expression from specific cell type only

    For each epithelial cell, get average expression from neighbors of the specified cell type only
    """

    print(f"\n=== FUNCTION 3: {target_cell_type.upper()} NEIGHBORHOOD EXPRESSION ===")
    print(f"Radius: {radius}")

    # Get expression matrix
    if issparse(adata.X):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X.copy()

    all_neighborhood_data = []
    all_neighborhood_expr = []

    for fov in adata.obs[fov_col].unique():
        print(f"\n--- Processing {fov} ---")

        # Get FOV data
        fov_mask = adata.obs[fov_col] == fov
        fov_obs = adata.obs[fov_mask].copy()
        fov_expr = expr_matrix[fov_mask]

        # Count target cell type in FOV
        target_cell_count = (fov_obs[cell_type_col] == target_cell_type).sum()
        epi_count = (fov_obs[cell_type_col] == 'epithelial').sum()

        print(f"Epithelial cells: {epi_count}, {target_cell_type}: {target_cell_count}")

        if target_cell_count == 0 or epi_count == 0:
            print(f"Skipping FOV - insufficient cells")
            continue

        # Get epithelial cells
        epi_mask = fov_obs[cell_type_col] == 'epithelial'
        epi_obs = fov_obs[epi_mask].copy()
        epi_indices = np.where(epi_mask)[0]

        # Get coordinates
        all_coords = fov_obs[[x_col, y_col]].values
        epi_coords = epi_obs[[x_col, y_col]].values

        # Find neighbors
        nbrs = NearestNeighbors(radius=radius).fit(all_coords)
        neighbor_indices = nbrs.radius_neighbors(epi_coords, return_distance=False)

        cells_with_target_neighbors = 0

        for i, epi_neighbors in enumerate(neighbor_indices):
            epi_idx_in_fov = epi_indices[i]

            # Remove epithelial cell itself
            neighbors_only = [idx for idx in epi_neighbors if idx != epi_idx_in_fov]

            if len(neighbors_only) == 0:
                # No neighbors at all
                neighborhood_expr = np.zeros(expr_matrix.shape[1])
                n_target_neighbors = 0
            else:
                # Filter neighbors to target cell type only
                neighbor_cell_types = fov_obs.iloc[neighbors_only][cell_type_col]
                target_neighbor_indices = [neighbors_only[j] for j, ct in enumerate(neighbor_cell_types) if
                                           ct == target_cell_type]

                if len(target_neighbor_indices) == 0:
                    # No neighbors of target cell type
                    neighborhood_expr = np.zeros(expr_matrix.shape[1])
                    n_target_neighbors = 0
                else:
                    # Average expression from target cell type neighbors
                    neighborhood_expr = fov_expr[target_neighbor_indices].mean(axis=0)
                    n_target_neighbors = len(target_neighbor_indices)
                    cells_with_target_neighbors += 1

            # Store epithelial cell metadata
            epi_cell = epi_obs.iloc[i]
            neighborhood_data = {
                'original_cell_index': fov_obs.index[epi_idx_in_fov],
                'FOV': fov,
                'n_neighbors': n_target_neighbors,  # Changed from 'n_target_neighbors' to 'n_neighbors'
                'target_cell_type': target_cell_type,
                'x_coord': epi_cell[x_col],
                'y_coord': epi_cell[y_col]
            }

            # Add NEAT1 classifications

            neighborhood_data[neat1_col] = epi_cell[neat1_col]

            all_neighborhood_data.append(neighborhood_data)
            all_neighborhood_expr.append(neighborhood_expr)

        print(
            f"Epithelial cells with {target_cell_type} neighbors: {cells_with_target_neighbors}/{len(epi_obs)} ({cells_with_target_neighbors / len(epi_obs) * 100:.1f}%)")

    # Create neighborhood AnnData
    neighborhood_expr_matrix = np.vstack(all_neighborhood_expr)
    neighborhood_obs = pd.DataFrame(all_neighborhood_data)

    celltype_neighborhood_adata = sc.AnnData(
        X=neighborhood_expr_matrix,
        obs=neighborhood_obs,
        var=adata.var.copy()
    )

    print(f"\nFinal {target_cell_type} neighborhood dataset: {celltype_neighborhood_adata.n_obs} epithelial cells")
    print(
        f"Cells with {target_cell_type} neighbors: {(neighborhood_obs['n_neighbors'] > 0).sum()}")  # Updated reference
    print(
        f"Average {target_cell_type} neighbors per cell: {neighborhood_obs['n_neighbors'].mean():.1f}")  # Updated reference


    return celltype_neighborhood_adata


def run_de_on_neighborhoods(neighborhood_adata, group_col='median_NEAT1_norm',
                            save_folder = ""):
    """
    Function 2: Run differential expression on neighborhood profiles

    Takes the neighborhood AnnData from Function 1 and runs DE analysis
    """

    print(f"\n=== FUNCTION 2: DE ANALYSIS ON NEIGHBORHOODS ===")
    print(f"Grouping by: {group_col}")

    if group_col not in neighborhood_adata.obs.columns:
        print(f"ERROR: Column {group_col} not found")
        return None

    # Show group sizes
    group_counts = neighborhood_adata.obs[group_col].value_counts()
    print(f"Group sizes: {group_counts.to_dict()}")

    # Convert boolean columns to categorical for scanpy compatibility
    if neighborhood_adata.obs[group_col].dtype == 'bool':
        print(f"Converting boolean {group_col} to categorical")
        neighborhood_adata.obs[group_col] = neighborhood_adata.obs[group_col].astype(str).astype('category')
    elif not hasattr(neighborhood_adata.obs[group_col], 'cat'):
        print(f"Converting {group_col} to categorical")
        neighborhood_adata.obs[group_col] = neighborhood_adata.obs[group_col].astype('category')

    # Filter out cells with zero neighbors (optional)
    neighbor_col = 'n_neighbors' if 'n_neighbors' in neighborhood_adata.obs.columns else 'n_target_neighbors'

    has_neighbors = neighborhood_adata.obs[neighbor_col] > 0
    if has_neighbors.sum() < len(neighborhood_adata):
        print(
            f"Filtering to {has_neighbors.sum()} cells with neighbors (excluding {(~has_neighbors).sum()} isolated cells)")
        neighborhood_adata_filtered = neighborhood_adata[has_neighbors].copy()
    else:
        neighborhood_adata_filtered = neighborhood_adata.copy()

    # Run DE analysis
    sc.tl.rank_genes_groups(neighborhood_adata_filtered, group_col, method='wilcoxon', pts=True)
    sc.tl.filter_rank_genes_groups(neighborhood_adata_filtered, groupby=group_col, min_fold_change=1)

    # Plot rank genes
    sc.settings.savdir = save_folder if save_folder else '.'
    sc.pl.rank_genes_groups(neighborhood_adata_filtered, n_genes=50,
                            sharey=False, save = f'_rank_genes_{group_col}.png')
    plt.show()

    # Extract detailed results
    result = neighborhood_adata_filtered.uns['rank_genes_groups_filtered']

    # Parse into DataFrame
    clusters = []
    genes = []
    logFC = []
    pval = []
    padj = []
    perc_cluster = []
    perc_others = []
    perc_diff = []

    for group in result['names'].dtype.names:
        notna_idxs = pd.Series(result['names'][group]).notna()
        genes_group = pd.Series(result['names'][group]).dropna()
        clusters.extend([group] * genes_group.size)
        genes.extend(genes_group)
        logFC.extend(result['logfoldchanges'][group][notna_idxs])
        pval.extend(result['pvals'][group][notna_idxs])
        padj.extend(result['pvals_adj'][group][notna_idxs])
        perc_cluster.extend(result['pts'][group].loc[genes_group])
        perc_others.extend(result['pts_rest'][group].loc[genes_group])
        perc_diff.extend(result['pts'][group].loc[genes_group] - result['pts_rest'][group].loc[genes_group])

    result_df = pd.DataFrame({
        'Cluster': clusters,
        'Gene': genes,
        'logFC': logFC,
        'pval': pval,
        'padj': padj,
        'perc_pos_cells': perc_cluster,
        'perc_pos_cells_others': perc_others,
        'perc_diff': perc_diff
    })

    result_df = result_df[result_df.padj < 0.05]
    result_df['FC_percDiff'] = result_df.logFC * result_df.perc_diff
    result_df = result_df.sort_values('FC_percDiff', ascending=False)

    print(f"Significant genes (padj < 0.05): {len(result_df)}")

    if len(result_df) > 0:
        print("\nTop 10 upregulated genes:")
        print(result_df.head(10)[['Gene', 'Cluster', 'logFC', 'padj', 'perc_diff', 'FC_percDiff']])

        # Create dotplots
        best_genes = list(result_df['Gene'].unique()[:50])
        sc.pl.dotplot(neighborhood_adata_filtered, best_genes, groupby=group_col, swap_axes=True)
        plt.show()

        # Specific genes
        specific_genes = ['NEAT1', 'VEGFA', 'MALAT1', 'CD55']
        available_genes = [g for g in specific_genes if g in neighborhood_adata_filtered.var_names]
        if available_genes:
            sc.pl.dotplot(neighborhood_adata_filtered, available_genes, groupby=group_col, swap_axes=True)
            plt.show()
    result_df.to_csv(f'{save_folder}/de_results_neighborhoods_{group_col}.csv', index=False)
    return result_df


def run_de_on_neighborhoods(neighborhood_adata, group_col='median_NEAT1_norm',
                            account_for_fov=False, fov_col='FOV',
                            save_folder=""):
    """
    Run differential expression on neighborhood profiles

    Parameters
    ----------
    neighborhood_adata : AnnData
        Neighborhood expression data
    group_col : str
        Column to group by
    account_for_fov : bool
        If True, aggregate by FOV first to avoid pseudoreplication
    fov_col : str
        FOV column name
    save_folder : str
        Save directory

    Returns
    -------
    DataFrame with DE results
    """

    print(f"\n=== DE ANALYSIS ON NEIGHBORHOODS ===")
    print(f"Grouping by: {group_col}, FOV adjustment: {account_for_fov}")

    if group_col not in neighborhood_adata.obs.columns:
        print(f"ERROR: Column {group_col} not found")
        return None

    # Show group sizes
    group_counts = neighborhood_adata.obs[group_col].value_counts()
    print(f"Group sizes: {group_counts.to_dict()}")

    # FOV aggregation if requested
    if account_for_fov:
        if fov_col not in neighborhood_adata.obs.columns:
            raise ValueError(f"FOV column '{fov_col}' not found")

        print("Aggregating by FOV to avoid pseudoreplication...")
        working_adata = _aggregate_by_fov(neighborhood_adata, group_col, fov_col)
        suffix = "_fov_adjusted"
    else:
        working_adata = neighborhood_adata.copy()
        suffix = "_pooled"

    # Convert to categorical
    if working_adata.obs[group_col].dtype == 'bool':
        working_adata.obs[group_col] = working_adata.obs[group_col].astype(str).astype('category')
    elif not hasattr(working_adata.obs[group_col], 'cat'):
        working_adata.obs[group_col] = working_adata.obs[group_col].astype('category')

    # Filter cells with neighbors
    neighbor_col = 'n_neighbors' if 'n_neighbors' in working_adata.obs.columns else 'n_target_neighbors'
    if neighbor_col in working_adata.obs.columns:
        has_neighbors = working_adata.obs[neighbor_col] > 0
        if has_neighbors.sum() < len(working_adata):
            print(f"Filtering to {has_neighbors.sum()} cells with neighbors")
            working_adata = working_adata[has_neighbors].copy()

    # Run DE analysis
    import scanpy as sc
    sc.tl.rank_genes_groups(working_adata, group_col, method='wilcoxon', pts=True)
    sc.tl.filter_rank_genes_groups(working_adata, groupby=group_col, min_fold_change=1)

    # Extract results
    result = working_adata.uns['rank_genes_groups_filtered']
    result_df = _extract_de_results(result)

    # Save results
    save_path = os.path.join(save_folder, f'de_results{suffix}.csv')
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")

    # Print summary
    if len(result_df) > 0:
        print(f"Significant genes (padj < 0.05): {len(result_df)}")
        print("\nTop 5 genes:")
        top_genes = result_df.head(5)[['Gene', 'Cluster', 'logFC', 'padj', 'FC_percDiff']]
        print(top_genes.to_string(index=False))

    return result_df


def run_de_on_neighborhoods_mixed_effects(neighborhood_adata, group_col='median_NEAT1_norm',
                                          fov_col='FOV', save_folder=""):
    """
    Run DE analysis using mixed-effects models with FOV as random effect

    Parameters
    ----------
    neighborhood_adata : AnnData
        Neighborhood expression data
    group_col : str
        Column to group by
    fov_col : str
        FOV column name (random effect)
    save_folder : str
        Save directory

    Returns
    -------
    DataFrame with mixed-effects DE results
    """

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import mixedlm
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("Please install statsmodels: pip install statsmodels")

    print(f"\n=== MIXED-EFFECTS DE ANALYSIS ON NEIGHBORHOODS ===")
    print(f"Grouping by: {group_col}, Random effect: {fov_col}")

    # Get expression matrix
    if hasattr(neighborhood_adata.X, 'toarray'):
        expr_matrix = neighborhood_adata.X.toarray()
    else:
        expr_matrix = neighborhood_adata.X

    # Filter cells with neighbors
    neighbor_col = 'n_neighbors' if 'n_neighbors' in neighborhood_adata.obs.columns else 'n_target_neighbors'
    if neighbor_col in neighborhood_adata.obs.columns:
        has_neighbors = neighborhood_adata.obs[neighbor_col] > 0
        expr_matrix = expr_matrix[has_neighbors]
        obs_data = neighborhood_adata.obs[has_neighbors].copy()
    else:
        obs_data = neighborhood_adata.obs.copy()

    print(f"Testing {len(neighborhood_adata.var_names)} genes across {len(obs_data)} cells")

    results = []

    for i, gene in enumerate(neighborhood_adata.var_names):
        if i % 500 == 0:
            print(f"Processing gene {i + 1}/{len(neighborhood_adata.var_names)}")

        gene_expr = expr_matrix[:, i]

        # Skip if no variation
        if np.var(gene_expr) < 1e-6:
            continue

        # Create data for model
        model_data = obs_data.copy()
        model_data['gene_expr'] = gene_expr

        try:
            # Fit mixed model: expression ~ group + (1|FOV)
            formula = f"gene_expr ~ C({group_col})"
            model = mixedlm(formula, model_data, groups=model_data[fov_col])
            result = model.fit(reml=False)

            # Extract group effects
            params = result.params
            pvals = result.pvalues

            for param_name in params.index:
                if param_name.startswith('C('):  # Group comparison parameters
                    results.append({
                        'Gene': gene,
                        'parameter': param_name,
                        'effect_size': params[param_name],
                        'p_value': pvals[param_name],
                        'aic': result.aic,
                        'n_obs': result.nobs,
                        'n_groups': len(model_data[fov_col].unique())
                    })

        except Exception:
            # Skip problematic genes silently
            continue

    if not results:
        print("No successful mixed-effects models")
        return None

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_adjusted'] = p_adj

    # Add significance stars
    def stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    results_df['significance'] = results_df['p_adjusted'].apply(stars)

    # Sort by significance
    results_df = results_df.sort_values('p_adjusted')

    # Save results
    save_path = os.path.join(save_folder, 'de_results_mixed_effects.csv')
    results_df.to_csv(save_path, index=False)
    print(f"Mixed-effects results saved to: {save_path}")

    # Print summary
    n_sig = (results_df['p_adjusted'] < 0.05).sum()
    print(f"Significant genes (padj < 0.05): {n_sig}")

    if n_sig > 0:
        print("\nTop 5 significant genes:")
        top_genes = results_df.head(5)[['Gene', 'parameter', 'effect_size', 'p_adjusted', 'significance']]
        print(top_genes.to_string(index=False))

    return results_df


def _aggregate_by_fov(adata, group_col, fov_col):
    """
    Aggregate expression data by FOV and group (mean expression per FOV-group)
    """
    import scanpy as sc

    # Get expression matrix
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X

    # Create DataFrame
    expr_df = pd.DataFrame(expr_matrix, columns=adata.var_names, index=adata.obs.index)
    expr_df[fov_col] = adata.obs[fov_col].values
    expr_df[group_col] = adata.obs[group_col].values

    # Add neighbor count if exists
    neighbor_col = 'n_neighbors' if 'n_neighbors' in adata.obs.columns else 'n_target_neighbors'
    if neighbor_col in adata.obs.columns:
        expr_df[neighbor_col] = adata.obs[neighbor_col].values

    # Group by FOV and group, take mean
    gene_cols = list(adata.var_names)
    groupby_cols = [fov_col, group_col]

    agg_dict = {gene: 'mean' for gene in gene_cols}
    if neighbor_col in expr_df.columns:
        agg_dict[neighbor_col] = 'mean'

    aggregated_df = expr_df.groupby(groupby_cols).agg(agg_dict).reset_index()

    print(f"Aggregated from {len(adata)} cells to {len(aggregated_df)} FOV-group combinations")

    # Create new AnnData
    obs_cols = [col for col in aggregated_df.columns if col not in gene_cols]

    aggregated_adata = sc.AnnData(
        X=aggregated_df[gene_cols].values,
        obs=aggregated_df[obs_cols].reset_index(drop=True),
        var=adata.var.copy()
    )

    return aggregated_adata


def _extract_de_results(scanpy_result):
    """
    Extract DE results from scanpy rank_genes_groups_filtered result
    """

    clusters = []
    genes = []
    logFC = []
    pval = []
    padj = []
    perc_cluster = []
    perc_others = []

    for group in scanpy_result['names'].dtype.names:
        notna_idxs = pd.Series(scanpy_result['names'][group]).notna()
        genes_group = pd.Series(scanpy_result['names'][group]).dropna()

        clusters.extend([group] * len(genes_group))
        genes.extend(genes_group)
        logFC.extend(scanpy_result['logfoldchanges'][group][notna_idxs])
        pval.extend(scanpy_result['pvals'][group][notna_idxs])
        padj.extend(scanpy_result['pvals_adj'][group][notna_idxs])
        perc_cluster.extend(scanpy_result['pts'][group].loc[genes_group])
        perc_others.extend(scanpy_result['pts_rest'][group].loc[genes_group])

    result_df = pd.DataFrame({
        'Cluster': clusters,
        'Gene': genes,
        'logFC': logFC,
        'pval': pval,
        'padj': padj,
        'perc_pos_cells': perc_cluster,
        'perc_pos_cells_others': perc_others
    })

    # Calculate additional metrics
    result_df['perc_diff'] = result_df['perc_pos_cells'] - result_df['perc_pos_cells_others']
    result_df['FC_percDiff'] = result_df['logFC'] * result_df['perc_diff']

    # Filter and sort
    result_df = result_df[result_df['padj'] < 0.05]
    result_df = result_df.sort_values('FC_percDiff', ascending=False)

    return result_df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_neighborhood_de_summary(de_results, title_suffix="", save_folder=None):
    """
    Create summary plots for neighborhood DE results

    Parameters
    ----------
    de_results : dict
        Dictionary with keys like 'pooled', 'fov_adjusted', 'mixed_effects'
        Each containing DE results DataFrame
    title_suffix : str
        Add to plot titles (e.g., cell type name)
    save_folder : str
        Directory to save plots
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Neighborhood DE Analysis Summary {title_suffix}', fontsize=16)

    methods = list(de_results.keys())
    colors = ['blue', 'red', 'green'][:len(methods)]

    # 1. Number of significant genes by method
    n_sig = []
    method_names = []
    for method, results in de_results.items():
        if results is not None and len(results) > 0:
            n_sig.append(len(results))
            method_names.append(method)

    axes[0, 0].bar(method_names, n_sig, color=colors[:len(method_names)])
    axes[0, 0].set_title('Significant Genes by Method')
    axes[0, 0].set_ylabel('Number of Genes')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Effect size distributions
    for i, (method, results) in enumerate(de_results.items()):
        if results is not None and len(results) > 0:
            effect_col = 'logFC' if 'logFC' in results.columns else 'effect_size'
            if effect_col in results.columns:
                axes[0, 1].hist(results[effect_col], alpha=0.6,
                                label=method, color=colors[i], bins=20)

    axes[0, 1].set_title('Effect Size Distributions')
    axes[0, 1].set_xlabel('Log Fold Change / Effect Size')
    axes[0, 1].set_ylabel('Number of Genes')
    axes[0, 1].legend()
    axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)

    # 3. P-value distributions
    for i, (method, results) in enumerate(de_results.items()):
        if results is not None and len(results) > 0:
            p_col = 'padj' if 'padj' in results.columns else 'p_adjusted'
            if p_col in results.columns:
                axes[1, 0].hist(-np.log10(results[p_col] + 1e-300), alpha=0.6,
                                label=method, color=colors[i], bins=20)

    axes[1, 0].set_title('Significance Distributions')
    axes[1, 0].set_xlabel('-log10(Adjusted P-value)')
    axes[1, 0].set_ylabel('Number of Genes')
    axes[1, 0].legend()
    axes[1, 0].axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')

    # 4. Method concordance (if multiple methods)
    if len(de_results) >= 2:
        method_list = list(de_results.keys())
        method1, method2 = method_list[0], method_list[1]

        df1 = de_results[method1]
        df2 = de_results[method2]

        if df1 is not None and df2 is not None and len(df1) > 0 and len(df2) > 0:
            # Get common genes
            genes1 = set(df1['Gene']) if 'Gene' in df1.columns else set()
            genes2 = set(df2['Gene']) if 'Gene' in df2.columns else set()
            common_genes = genes1.intersection(genes2)

            if len(common_genes) > 0:
                # Compare effect sizes for common genes
                effect1 = []
                effect2 = []
                for gene in common_genes:
                    e1 = df1[df1['Gene'] == gene]['logFC'].iloc[0] if 'logFC' in df1.columns else \
                    df1[df1['Gene'] == gene]['effect_size'].iloc[0]
                    e2 = df2[df2['Gene'] == gene]['logFC'].iloc[0] if 'logFC' in df2.columns else \
                    df2[df2['Gene'] == gene]['effect_size'].iloc[0]
                    effect1.append(e1)
                    effect2.append(e2)

                axes[1, 1].scatter(effect1, effect2, alpha=0.6)
                axes[1, 1].plot([-2, 2], [-2, 2], 'r--', alpha=0.5)  # Unity line
                axes[1, 1].set_xlabel(f'{method1} Effect Size')
                axes[1, 1].set_ylabel(f'{method2} Effect Size')
                axes[1, 1].set_title(f'Method Concordance\n({len(common_genes)} common genes)')
            else:
                axes[1, 1].text(0.5, 0.5, 'No common\nsignificant genes',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Method Concordance')
    else:
        axes[1, 1].text(0.5, 0.5, 'Need ≥2 methods\nfor comparison',
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()

    if save_folder:
        save_path = f"{save_folder}/de_summary{title_suffix.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved: {save_path}")

    plt.show()


def plot_top_genes_heatmap(neighborhood_adata, de_results, group_col,
                           n_genes=20, save_folder=None, title_suffix=""):
    """
    Plot heatmap of top DE genes across groups
    """

    # Get top genes from any method
    all_genes = set()
    for method, results in de_results.items():
        if results is not None and len(results) > 0:
            top_genes = results.head(n_genes)['Gene'].tolist()
            all_genes.update(top_genes)

    if len(all_genes) == 0:
        print("No genes to plot")
        return

    # Filter genes that exist in data
    available_genes = [g for g in all_genes if g in neighborhood_adata.var_names][:n_genes]

    if len(available_genes) == 0:
        print("No available genes in data")
        return

    # Create expression matrix for heatmap
    if hasattr(neighborhood_adata.X, 'toarray'):
        expr_matrix = neighborhood_adata.X.toarray()
    else:
        expr_matrix = neighborhood_adata.X

    gene_indices = [list(neighborhood_adata.var_names).index(g) for g in available_genes]
    gene_expr = expr_matrix[:, gene_indices]

    # Group by condition and calculate means
    expr_df = pd.DataFrame(gene_expr, columns=available_genes)
    expr_df[group_col] = neighborhood_adata.obs[group_col].values

    group_means = expr_df.groupby(group_col).mean()

    # Plot heatmap
    plt.figure(figsize=(max(8, len(available_genes) * 0.4), max(6, len(group_means) * 0.5)))

    sns.heatmap(group_means.T, annot=False, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean Expression'})

    plt.title(f'Top DE Genes Heatmap {title_suffix}')
    plt.xlabel('Group')
    plt.ylabel('Genes')
    plt.tight_layout()

    if save_folder:
        save_path = f"{save_folder}/top_genes_heatmap{title_suffix.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved: {save_path}")

    plt.show()


def plot_method_comparison_volcano(de_results, save_folder=None, title_suffix=""):
    """
    Create volcano plots comparing different methods
    """

    methods = list(de_results.keys())
    n_methods = len(methods)

    if n_methods == 0:
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for i, (method, results) in enumerate(de_results.items()):
        if results is None or len(results) == 0:
            axes[i].text(0.5, 0.5, 'No significant\nresults',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{method}')
            continue

        # Get effect size and p-values
        effect_col = 'logFC' if 'logFC' in results.columns else 'effect_size'
        p_col = 'padj' if 'padj' in results.columns else 'p_adjusted'

        x = results[effect_col]
        y = -np.log10(results[p_col] + 1e-300)

        # Color by significance
        colors = ['red' if p < 0.05 else 'gray' for p in results[p_col]]

        axes[i].scatter(x, y, c=colors, alpha=0.6, s=20)
        axes[i].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        axes[i].axvline(0, color='black', linestyle='--', alpha=0.5)

        axes[i].set_xlabel('Effect Size')
        axes[i].set_ylabel('-log10(Adjusted P-value)')
        axes[i].set_title(f'{method}\n({len(results)} significant genes)')

        # Label top genes
        if len(results) > 0:
            top_5 = results.head(5)
            for _, row in top_5.iterrows():
                axes[i].annotate(row['Gene'],
                                 (row[effect_col], -np.log10(row[p_col] + 1e-300)),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, alpha=0.7)

    plt.tight_layout()

    if save_folder:
        save_path = f"{save_folder}/volcano_comparison{title_suffix.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Volcano plots saved: {save_path}")

    plt.show()


def create_neighborhood_de_report(cell_type, neighborhood_adata, de_results,
                                  group_col, save_folder):
    """
    Create a comprehensive report with all visualizations for one cell type
    """

    print(f"\n=== Creating DE Report for {cell_type} ===")

    title_suffix = f"({cell_type})"

    # 1. Summary plots
    plot_neighborhood_de_summary(de_results, title_suffix=title_suffix,
                                 save_folder=save_folder)

    # 2. Top genes heatmap
    plot_top_genes_heatmap(neighborhood_adata, de_results, group_col,
                           n_genes=20, save_folder=save_folder,
                           title_suffix=title_suffix)

    # 3. Volcano plots comparison
    plot_method_comparison_volcano(de_results, save_folder=save_folder,
                                   title_suffix=title_suffix)

    # 4. Basic scanpy plots if data available
    try:
        import scanpy as sc

        # Get some top genes for dotplot
        all_top_genes = set()
        for method, results in de_results.items():
            if results is not None and len(results) > 0:
                top_genes = results.head(10)['Gene'].tolist()
                all_top_genes.update(top_genes)

        available_genes = [g for g in all_top_genes if g in neighborhood_adata.var_names]

        if len(available_genes) > 0:
            # Set scanpy save directory
            sc.settings.figdir = save_folder

            # Dotplot
            sc.pl.dotplot(neighborhood_adata, available_genes[:15],
                          groupby=group_col, swap_axes=True,
                          save=f'_dotplot_{cell_type}.png')

            print(f"Scanpy dotplot saved for {cell_type}")

    except Exception as e:
        print(f"Could not create scanpy plots: {e}")

    print(f"Report complete for {cell_type}")


# Usage wrapper for your existing workflow
def run_de_with_visualizations(cell_neighborhoods, cell_type, column,
                               base_folder="results"):
    """
    Run all DE methods and create visualizations
    """

    # Create directories
    de_folder = f'{base_folder}/per_celltype_de/{cell_type}/'
    fov_folder = f'{base_folder}/per_celltype_fov_meta_analysis/{cell_type}/'
    os.makedirs(de_folder, exist_ok=True)
    os.makedirs(fov_folder, exist_ok=True)

    # Run all DE methods
    de_results = {}

    # Pooled
    de_results['pooled'] = NN.run_de_on_neighborhoods(
        cell_neighborhoods, column, save_folder=de_folder
    )

    # FOV-adjusted
    de_results['fov_adjusted'] = NN.run_de_on_neighborhoods(
        cell_neighborhoods, column, account_for_fov=True,
        fov_col='FOV', save_folder=de_folder
    )

    # Mixed effects
    de_results['mixed_effects'] = NN.run_de_on_neighborhoods_mixed_effects(
        cell_neighborhoods, column, fov_col='FOV', save_folder=fov_folder
    )

    # Create comprehensive report
    create_neighborhood_de_report(cell_type, cell_neighborhoods, de_results,
                                  column, de_folder)

    return de_results