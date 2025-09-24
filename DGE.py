import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


def de_analysis(adata,column, save_folder='results/de_analysis'):
    adata_epi = adata[adata.obs['label'] == 'epithelial'].copy()
    sc.tl.rank_genes_groups(adata_epi,column ,method='wilcoxon', pts=True)

    sc.tl.filter_rank_genes_groups(adata_epi, groupby=column,
                                  min_fold_change=1)

    # Plot results
    sc.settings.figdir = save_folder  # directory to save figures
    sc.pl.rank_genes_groups(adata_epi, n_genes=50, sharey=False,
                            save=f'rank_genes_groups_{column}.png')

    # Extract detailed results
    result = adata_epi.uns['rank_genes_groups_filtered']

    # Parse results into DataFrame
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
        clusters.extend([group]*genes_group.size)
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
    result_df.to_csv(f'{save_folder}/de_results_{column}.csv', index=False)

    best_genes1 = list(np.unique(result_df.Gene))

    sc.settings.figdir = save_folder  # directory to save figures
    sc.pl.dotplot(adata_epi, best_genes1, groupby=column,
                  swap_axes=True, save=f'dotplot_topDE_{column}.png')
    sc.pl.dotplot(adata_epi, ['NEAT1', 'VEGFA', 'MALAT1', 'CD55'],
                  groupby=column, swap_axes=True, save=f'dotplot_specific_genes_{column}.png')

    return result_df



################################
# The problem is that the effects may be driven by differences in cell type composition rather than true expression changes within cell types.
# based on FOV types


def de_analysis_per_fov(adata, neat1_column, fov_column='FOV', save_folder='results/de_analysis_per_fov'):
    """
    Run DE analysis within each FOV separately, then combine results
    Returns detailed log fold change information
    """
    print(f"=== PER-FOV DE ANALYSIS ===")
    print(f"Testing {neat1_column} across FOVs")

    adata_epi = adata[adata.obs['label'] == 'epithelial'].copy()
    # Check FOV distribution
    fov_neat1_counts = adata_epi.obs.groupby([fov_column, neat1_column]).size().unstack(fill_value=0)
    print(f"\nFOV-NEAT1 distribution:")
    print(fov_neat1_counts)

    # Find FOVs with sufficient cells in each NEAT1 group
    min_cells_per_group = 3
    valid_fovs = []

    for fov in fov_neat1_counts.index:
        if (fov_neat1_counts.loc[fov] >= min_cells_per_group).all():
            valid_fovs.append(fov)

    print(f"\nFOVs with ≥{min_cells_per_group} cells per group: {len(valid_fovs)}")
    print(f"Valid FOVs: {valid_fovs}")

    if len(valid_fovs) < 2:
        print("ERROR: Need at least 2 FOVs with sufficient cells per group")
        return None

    # Run DE analysis for each valid FOV
    all_fov_results = []

    for fov in valid_fovs:
        print(f"\n--- Analyzing FOV: {fov} ---")

        # Subset to this FOV
        fov_mask = adata_epi.obs[fov_column] == fov
        adata_fov = adata_epi[fov_mask].copy()

        # Check group sizes
        group_counts = adata_fov.obs[neat1_column].value_counts()
        print(f"Group sizes: {group_counts.to_dict()}")

        # Run your original DE analysis function on this FOV
        try:
            sc.tl.rank_genes_groups(adata_fov, neat1_column, method='wilcoxon', pts=True)
            sc.tl.filter_rank_genes_groups(adata_fov, groupby=neat1_column, min_fold_change=1)

            # Extract detailed results using your approach
            result = adata_fov.uns['rank_genes_groups_filtered']

            # Parse results into DataFrame
            for group in result['names'].dtype.names:
                notna_idxs = pd.Series(result['names'][group]).notna()
                genes_group = pd.Series(result['names'][group]).dropna()

                for i, gene in enumerate(genes_group):
                    if pd.isna(gene):
                        continue

                    idx = notna_idxs[notna_idxs].index[i]  # Get original index

                    all_fov_results.append({
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

            print(f"Found {len([r for r in all_fov_results if r['FOV'] == fov])} significant gene-group pairs")

        except Exception as e:
            print(f"DE analysis failed for FOV {fov}: {e}")
            continue

    if not all_fov_results:
        print("No successful DE results across FOVs")
        return None

    # Convert to DataFrame
    fov_results_df = pd.DataFrame(all_fov_results)
    fov_results_df['FC_percDiff'] = fov_results_df['logFC'] * fov_results_df['perc_diff']
    print(f"\nTotal significant results across all FOVs: {len(fov_results_df)}")
    fov_results_df.to_csv(f'{save_folder}/de_results_per_fov_{neat1_column}.csv', index=False)

    # ANALYSIS 1: Genes significant in multiple FOVs
    gene_fov_counts = fov_results_df.groupby(['Gene', 'Group']).agg({
        'FOV': 'count',
        'logFC': ['mean', 'std'],
        'pval': lambda x: stats.combine_pvalues(x, method='fisher')[1],  # Fisher's combined p-value
        'perc_diff': 'mean',
        'FC_percDiff': 'mean'
    }).round(4)

    # Flatten column names
    gene_fov_counts.columns = ['n_fovs', 'mean_logFC', 'std_logFC', 'combined_pval', 'mean_perc_diff',
                               'mean_FC_percDiff']
    gene_fov_counts = gene_fov_counts.reset_index()

    # Multiple testing correction on combined p-values
    _, padj_combined, _, _ = multipletests(gene_fov_counts['combined_pval'], method='fdr_bh')
    gene_fov_counts['padj_combined'] = padj_combined

    # Add consistency metrics
    gene_fov_counts['consistency_score'] = (gene_fov_counts['n_fovs'] / len(valid_fovs)) * \
                                           (1 / (gene_fov_counts[
                                                     'std_logFC'] + 0.1))  # Higher when more FOVs and less variable
    gene_fov_counts.to_csv(f'{save_folder}/gene_fov_summary_{neat1_column}.csv', index=False)

    # Filter and sort
    significant_combined = gene_fov_counts[gene_fov_counts['padj_combined'] < 0.05].copy()
    significant_combined = significant_combined.sort_values('consistency_score', ascending=False)

    print(f"\nGenes significant after combining across FOVs: {len(significant_combined)}")

    # ANALYSIS 2: Consistency analysis
    print(f"\n=== CONSISTENCY ANALYSIS ===")

    consistency_summary = []
    for n_fovs in range(1, len(valid_fovs) + 1):
        subset = significant_combined[significant_combined['n_fovs'] >= n_fovs]
        consistency_summary.append({
            'min_fovs': n_fovs,
            'n_genes': len(subset),
            'percent_of_total': len(subset) / len(significant_combined) * 100 if len(significant_combined) > 0 else 0
        })

    consistency_df = pd.DataFrame(consistency_summary)
    print(consistency_df)
    consistency_df.to_csv(f'{save_folder}/consistency_summary_{neat1_column}.csv', index=False)

    # ANALYSIS 3: Top consistent genes
    print(f"\n=== TOP CONSISTENT GENES ===")
    print("(Genes found in multiple FOVs with consistent direction)")

    top_consistent = significant_combined.head(20)
    display_cols = ['Gene', 'Group', 'n_fovs', 'mean_logFC', 'std_logFC', 'padj_combined', 'consistency_score']
    print(top_consistent[display_cols])

    # ANALYSIS 4: FOV-specific patterns
    print(f"\n=== FOV-SPECIFIC ANALYSIS ===")
    fov_summary = fov_results_df.groupby('FOV').agg({
        'Gene': 'nunique',
        'logFC': ['mean', 'std'],
        'FC_percDiff': 'mean'
    }).round(3)
    fov_summary.columns = ['unique_genes', 'mean_logFC', 'std_logFC', 'mean_FC_percDiff']
    print("Summary per FOV:")
    print(fov_summary)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Number of FOVs per gene
    axes[0, 0].hist(gene_fov_counts['n_fovs'], bins=range(1, len(valid_fovs) + 2),
                    alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Number of FOVs')
    axes[0, 0].set_ylabel('Number of Genes')
    axes[0, 0].set_title('Gene Consistency Across FOVs')

    # Plot 2: LogFC consistency
    for n_fovs in range(2, min(6, len(valid_fovs) + 1)):
        subset = gene_fov_counts[gene_fov_counts['n_fovs'] == n_fovs]
        if len(subset) > 0:
            axes[0, 1].scatter(subset['mean_logFC'], subset['std_logFC'],
                               label=f'{n_fovs} FOVs', alpha=0.6)
    axes[0, 1].set_xlabel('Mean Log Fold Change')
    axes[0, 1].set_ylabel('Std Log Fold Change')
    axes[0, 1].set_title('LogFC Consistency')
    axes[0, 1].legend()

    # Plot 3: P-value distribution
    axes[1, 0].hist(gene_fov_counts['combined_pval'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    axes[1, 0].set_xlabel('Combined P-value')
    axes[1, 0].set_ylabel('Number of Genes')
    axes[1, 0].set_title('Combined P-value Distribution')
    axes[1, 0].legend()

    # Plot 4: Heatmap of top genes across FOVs
    if len(significant_combined) > 0:
        top_genes = significant_combined.head(10)['Gene'].tolist()
        heatmap_data = []
        for gene in top_genes:
            gene_data = fov_results_df[fov_results_df['Gene'] == gene]
            row = {}
            for fov in valid_fovs:
                fov_gene = gene_data[gene_data['FOV'] == fov]
                if len(fov_gene) > 0:
                    row[fov] = fov_gene['logFC'].iloc[0]
                else:
                    row[fov] = 0
            row['Gene'] = gene
            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(heatmap_data).set_index('Gene')
        sns.heatmap(heatmap_df, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('LogFC Heatmap: Top Genes × FOVs')

    plt.savefig(f'{save_folder}/de_per_fov_summary_{neat1_column}.png', dpi=300)
    plt.tight_layout()
    plt.show()

    # Return comprehensive results
    results = {
        'per_fov_results': fov_results_df,
        'combined_results': significant_combined,
        'consistency_summary': consistency_df,
        'fov_summary': fov_summary,
        'valid_fovs': valid_fovs
    }

    return results



# Mixed Effects Differential Expression Analysis
# Accounts for patient (FOV) effects while testing NEAT1 associations
def mixed_effects_de_analysis(adata, neat1_column, patient_column='FOV', n_jobs=1,
                              save_folder='results/mixed_effects_de'):
    """
    Differential expression using mixed effects models
    Model: Gene_expression ~ NEAT1_group + (1|Patient)

    This accounts for patient-to-patient variability while testing NEAT1 effects
    """
    print("=== MIXED EFFECTS DE ANALYSIS ===")
    print("Model: Gene_expression ~ NEAT1_group + (1|Patient)")
    adata_epi = adata[adata.obs['label'] == 'epithelial'].copy()

    # Get expression data
    if hasattr(adata_epi.X, 'toarray'):
        X = adata_epi.X.toarray()
    else:
        X = adata_epi.X

    genes = adata_epi.var_names
    n_genes = len(genes)

    # Prepare metadata
    metadata = adata_epi.obs[[neat1_column, patient_column]].copy()
    metadata = metadata.dropna()

    # Filter expression data to match metadata
    valid_indices = metadata.index
    X_filtered = X[adata_epi.obs.index.isin(valid_indices)]

    print(f"Analyzing {len(valid_indices)} cells across {len(metadata[patient_column].unique())} patients")
    print(f"NEAT1 group distribution: {metadata[neat1_column].value_counts().to_dict()}")

    # Check patient distribution
    patient_neat1_dist = metadata.groupby([patient_column, neat1_column]).size().unstack(fill_value=0)
    print(f"\nPatient-NEAT1 distribution:")
    print(patient_neat1_dist)

    # Ensure we have enough patients with both groups
    patients_with_both_groups = (patient_neat1_dist > 0).all(axis=1).sum()
    print(f"Patients with both NEAT1 groups: {patients_with_both_groups}")

    if patients_with_both_groups < 3:
        print("WARNING: Few patients have both NEAT1 groups. Results may be unreliable.")

    # Run mixed effects models for each gene
    results = []
    failed_genes = []

    print(f"\nAnalyzing {n_genes} genes...")

    for i, gene in enumerate(genes):
        if i % 500 == 0:
            print(f"Progress: {i}/{n_genes} genes ({i / n_genes * 100:.1f}%)")

        try:
            # Prepare data for this gene
            gene_data = pd.DataFrame({
                'expression': X_filtered[:, i],
                'neat1_group': metadata[neat1_column].values,
                'patient': metadata[patient_column].values
            })

            # Skip genes with no variance
            if gene_data['expression'].std() == 0:
                continue

            # Fit mixed effects model
            # Random intercept for patient: (1|patient)
            model = mixedlm("expression ~ neat1_group", gene_data, groups=gene_data["patient"])
            fitted_model = model.fit(reml=False)  # Use ML for model comparison

            # Extract results for each NEAT1 group coefficient
            for param in fitted_model.params.index:
                if 'neat1_group' in param and param != 'Intercept':

                    # Get group name (remove 'neat1_group[T.' and ']' if present)
                    if '[T.' in param:
                        group_name = param.split('[T.')[1].rstrip(']')
                    else:
                        group_name = param.replace('neat1_group', '')

                    results.append({
                        'Gene': gene,
                        'NEAT1_Group': group_name,
                        'Coefficient': fitted_model.params[param],
                        'Std_Error': fitted_model.bse[param],
                        'P_value': fitted_model.pvalues[param],
                        'CI_lower': fitted_model.conf_int().loc[param, 0],
                        'CI_upper': fitted_model.conf_int().loc[param, 1],
                        'Random_Effect_Var': fitted_model.cov_re.iloc[0, 0],  # Patient variance
                        'Residual_Var': fitted_model.scale,  # Within-patient variance
                        'Log_Likelihood': fitted_model.llf
                    })

        except Exception as e:
            failed_genes.append({'Gene': gene, 'Error': str(e)})
            continue

    print(f"\nCompleted: {len(results)} gene-group results")
    print(f"Failed: {len(failed_genes)} genes")

    if len(results) == 0:
        print("No successful models. Check your data.")
        return None

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Multiple testing correction
    _, results_df['P_adj'], _, _ = multipletests(results_df['P_value'], method='fdr_bh')

    # Calculate effect size measures
    results_df['Effect_Size'] = results_df['Coefficient'] / results_df['Std_Error']  # z-score
    results_df['Variance_Explained_by_Patient'] = (
            results_df['Random_Effect_Var'] /
            (results_df['Random_Effect_Var'] + results_df['Residual_Var'])
    )

    # Sort by significance
    results_df = results_df.sort_values('P_adj')
    results_df.to_csv(f'{save_folder}/mixed_effects_de_results.csv', index=False)
    # Summary statistics
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Significant genes (P_adj < 0.05): {(results_df['P_adj'] < 0.05).sum()}")
    print(f"Significant genes (P_adj < 0.01): {(results_df['P_adj'] < 0.01).sum()}")

    # Patient variance contribution
    mean_patient_var = results_df['Variance_Explained_by_Patient'].mean()
    print(f"Average variance explained by patient effects: {mean_patient_var:.3f}")

    # Top results
    significant_results = results_df[results_df['P_adj'] < 0.05]

    if len(significant_results) > 0:
        print(f"\nTop 10 significant genes:")
        display_cols = ['Gene', 'NEAT1_Group', 'Coefficient', 'P_value', 'P_adj', 'Effect_Size']
        print(significant_results[display_cols].head(10))

        # Group-wise summary
        group_summary = significant_results.groupby('NEAT1_Group').agg({
            'Gene': 'count',
            'Coefficient': ['mean', 'std'],
            'Effect_Size': ['mean', 'std']
        }).round(3)
        group_summary.columns = ['N_Genes', 'Mean_Coef', 'Std_Coef', 'Mean_Effect', 'Std_Effect']
        print(f"\nSummary by NEAT1 group:")
        print(group_summary)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: P-value histogram
    axes[0, 0].hist(results_df['P_value'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    axes[0, 0].set_xlabel('P-value')
    axes[0, 0].set_ylabel('Number of Gene-Group Tests')
    axes[0, 0].set_title('P-value Distribution')
    axes[0, 0].legend()

    # Plot 2: Effect size vs significance
    axes[0, 1].scatter(results_df['Effect_Size'], -np.log10(results_df['P_adj']),
                       alpha=0.6, s=20)
    axes[0, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='FDR=0.05')
    axes[0, 1].set_xlabel('Effect Size (Coefficient/SE)')
    axes[0, 1].set_ylabel('-log10(Adjusted P-value)')
    axes[0, 1].set_title('Volcano Plot')
    axes[0, 1].legend()

    # Plot 3: Patient variance contribution
    axes[1, 0].hist(results_df['Variance_Explained_by_Patient'], bins=30,
                    alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Proportion of Variance from Patient Effects')
    axes[1, 0].set_ylabel('Number of Genes')
    axes[1, 0].set_title('Patient Effect Size Distribution')

    # Plot 4: Coefficient distribution by group
    if len(significant_results) > 0:
        for group in significant_results['NEAT1_Group'].unique():
            group_data = significant_results[significant_results['NEAT1_Group'] == group]
            axes[1, 1].hist(group_data['Coefficient'], alpha=0.6,
                            label=f'Group: {group}', bins=20)
        axes[1, 1].set_xlabel('Coefficient (Log Fold Change)')
        axes[1, 1].set_ylabel('Number of Genes')
        axes[1, 1].set_title('Coefficient Distribution by NEAT1 Group')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{save_folder}/mixed_effects_de_summary.png', dpi=300)
    plt.show()

    # Return results
    final_results = {
        'results_df': results_df,
        'significant_genes': significant_results,
        'failed_genes': pd.DataFrame(failed_genes) if failed_genes else None,
        'summary': {
            'n_significant_05': (results_df['P_adj'] < 0.05).sum(),
            'n_significant_01': (results_df['P_adj'] < 0.01).sum(),
            'mean_patient_variance': mean_patient_var,
            'n_patients': len(metadata[patient_column].unique()),
            'n_cells': len(metadata)}
        }

    return final_results


def compare_with_naive_analysis(adata, neat1_column, patient_column='FOV',
                                save_folder='results/comparison_naive'):
    """
    Compare mixed effects results with naive analysis (ignoring patient effects)
    """
    print("\n=== COMPARISON WITH NAIVE ANALYSIS ===")
    adata_epi = adata[adata.obs['label'] == 'epithelial'].copy()

    # Run mixed effects
    mixed_results = mixed_effects_de_analysis(adata_epi, neat1_column, patient_column)

    if mixed_results is None:
        return None

    # Run naive analysis (your original approach on all data)
    print("\nRunning naive analysis for comparison...")

    sc.tl.rank_genes_groups(adata_epi, neat1_column,
                            method='wilcoxon', pts=True)

    # Extract naive results
    naive_results = []
    for group in adata_epi.uns['rank_genes_groups']['names'].dtype.names:
        for i, gene in enumerate(adata_epi.uns['rank_genes_groups']['names'][group]):
            if pd.isna(gene):
                continue

            naive_results.append({
                'Gene': gene,
                'NEAT1_Group': group,
                'Naive_LogFC': adata_epi.uns['rank_genes_groups']['logfoldchanges'][group][i],
                'Naive_Pval': adata_epi.uns['rank_genes_groups']['pvals'][group][i],
                'Naive_Padj': adata_epi.uns['rank_genes_groups']['pvals_adj'][group][i]
            })

    naive_df = pd.DataFrame(naive_results)

    # Merge with mixed effects results
    comparison = mixed_results['results_df'].merge(
        naive_df, on=['Gene', 'NEAT1_Group'], how='inner'
    )

    print(f"\nComparison for {len(comparison)} gene-group pairs:")

    # Compare significance calls
    mixed_sig = (comparison['P_adj'] < 0.05)
    naive_sig = (comparison['Naive_Padj'] < 0.05)

    print(f"Mixed effects significant: {mixed_sig.sum()}")
    print(f"Naive analysis significant: {naive_sig.sum()}")
    print(f"Agreement: {(mixed_sig == naive_sig).mean():.3f}")
    print(f"Mixed only: {(mixed_sig & ~naive_sig).sum()}")
    print(f"Naive only: {(naive_sig & ~mixed_sig).sum()}")

    comparison.to_csv(f'{save_folder}/comparison_mixed_vs_naive_{neat1_column}.csv', index=False)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coefficient vs LogFC
    axes[0].scatter(comparison['Naive_LogFC'], comparison['Coefficient'], alpha=0.6)
    axes[0].set_xlabel('Naive Log Fold Change')
    axes[0].set_ylabel('Mixed Effects Coefficient')
    axes[0].set_title('Effect Size Comparison')

    # Add diagonal line
    min_val = min(comparison['Naive_LogFC'].min(), comparison['Coefficient'].min())
    max_val = max(comparison['Naive_LogFC'].max(), comparison['Coefficient'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    # P-value comparison
    axes[1].scatter(-np.log10(comparison['Naive_Pval']), -np.log10(comparison['P_value']), alpha=0.6)
    axes[1].set_xlabel('-log10(Naive P-value)')
    axes[1].set_ylabel('-log10(Mixed Effects P-value)')
    axes[1].set_title('Significance Comparison')

    # Add diagonal line
    max_log_p = max(-np.log10(comparison['Naive_Pval']).max(), -np.log10(comparison['P_value']).max())
    axes[1].plot([0, max_log_p], [0, max_log_p], 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{save_folder}/comparison_mixed_vs_naive_{neat1_column}.png', dpi=300)
    plt.show()

    return comparison