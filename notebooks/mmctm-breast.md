---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3.7.12 64-bit (conda)
    language: python
    name: python3712jvsc74a57bd0dc0fe05456373cce17991fe9e2e9264df4f4d1e972d77814a9700f21c9e7a8e2
---

# Setting

```python
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
```

```python
import ete3
import hdbscan
import pandas as pd
from scipy.cluster import hierarchy
import umap
```

## modules

```python
def plot_heatmap_with_annot(plot_data, annot, figsize=(20, 15), out_path=None):
    assert plot_data.index[-1] == 'cluster_id', plot_data.index
    assert plot_data.shape[1] == annot.shape[1]
    assert (plot_data.columns != annot.columns).sum() == 0
    annot_cnt = annot.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=figsize, 
                             gridspec_kw={'height_ratios':[20, 1, 30], 'hspace':0.01})
    axa = axes[0] # annot axis
    axc = axes[1] # cluster axis
    axh = axes[2] # heatmap axis
    
    sns.heatmap(plot_data.iloc[:-1, :], cmap="vlag", ax=axh, cbar=False, vmin=-1, vmax=1, center=0)
    
    axc_hm = sns.heatmap(plot_data.loc[['cluster_id']], ax=axc, cbar=False, cmap='tab20')
    axc.set_xticks([])
    axc_hm.set_yticklabels(['cluster'], rotation=0); 
    
    axa_hm = sns.heatmap(annot, ax=axa, cbar=False, cmap='Blues')
    axa.set_xticks([])
    
    # fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
```

```python
def plot_heatmap(plot_data, figsize=(20, 10), out_path=None):
    assert plot_data.index[-1] == 'cluster_id', plot_data.index
    fig, (ax1, axh) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios':[1, 30], 'hspace':0.01})
    sns.heatmap(plot_data.iloc[:-1, :], cmap="vlag", ax=axh, cbar=False, vmin=-1, vmax=1, center=0)
    ax1_hm = sns.heatmap(plot_data.loc[['cluster_id']], ax=ax1, cbar=False, cmap='tab20')
    ax1.set_xticks([])
    ax1_hm.set_yticklabels(['cluster'], rotation=0); 
    # fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
```

```python
def reduce_dimensions(props, n_neighbors):
    # large n_neighbors, low min_dist
    embedder = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.0, metric='correlation',
        random_state=3049821
    )
    embedding = embedder.fit_transform(props.values.T)
    embedding = pd.DataFrame(
        embedding, columns=['umap1', 'umap2'],
        index=pd.Index(props.columns, name='sample_id')
    )
    return embedding


def construct_internal_node_name(node, leaf_names):
    leaf_names = node.pre_order(lambda x: leaf_names[x.id])
    leaf_names.sort()
    return ''.join(leaf_names)


def _tree_to_newick(node, parent_dist, leaf_names, is_root=False):
    if node.is_leaf():
        return '{}:{:.10g}'.format(leaf_names[node.id], parent_dist - node.dist)

    left_newick = _tree_to_newick(node.get_left(), node.dist, leaf_names)
    right_newick = _tree_to_newick(node.get_right(), node.dist, leaf_names)

    node_name = construct_internal_node_name(node, leaf_names)

    if is_root:
        newick = '({},{}){};'.format(left_newick, right_newick, node_name)
    else:
        newick = '({},{}){}:{:.10g}'.format(
            left_newick, right_newick, node_name, parent_dist - node.dist
        )
    return newick


def tree_to_newick(tree, leaf_names):
    return _tree_to_newick(tree, tree.dist, leaf_names, True)


def cluster_sample_embeddings(embedding, min_samples, min_cluster_size,
        cluster_selection_epsilon):

    # low min_samples, quite large min_cluster_size
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size,
        approx_min_span_tree=False, gen_min_span_tree=True,
        allow_single_cluster=False, cluster_selection_method='leaf',
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    cluster_labels = clusterer.fit_predict(embedding)

    cluster_labels = pd.DataFrame(
        cluster_labels + 1, columns=['cluster_id'], index=embedding.index
    )

    print(cluster_labels['cluster_id'].sort_values().unique())

    tree = hierarchy.to_tree(clusterer.single_linkage_tree_.to_numpy())
    nw = tree_to_newick(tree, embedding.index)
    tree = ete3.Tree(nw, format=1)

    return cluster_labels, tree, clusterer


def plot_embeddings(embeddings, clusters, clusterer, pdf_path):
    plot_data = embeddings.reset_index().merge(clusters.reset_index())

    clusters = plot_data['cluster_id'].unique()
    clusters.sort()

    
    if clusters.max() > 15:
        colour_palette = sns.color_palette('husl', clusters.max())
    else:
        colour_palette = sns.color_palette([
            '#4292c6', # blue
            '#fd8d3c', # orange
            '#41ab5d', # green
            '#e31a1c', # red
            '#dcbeff', # lavendar
            '#88419d', # purple
            '#d94801', # dark orange
            '#b35806', # brown
            '#a6761d', # drab brown
            '#969696', # gray
            '#252525', # black
            '#006400', # dark green
            '#191970', # midnight blue
            '#ffb6c1', # light pink
            '#c71585' # medium violet red
        ])

    colour_palette = {
        x: (colour_palette[x - 1] if x > 0 else (0.5, 0.5, 0.5))
        for x in clusters
    }

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(
        x='umap1', y='umap2', hue='cluster_id', linewidth=0,
        data=plot_data, ax=ax, palette=colour_palette
    )

    plt.savefig(pdf_path)

```

# Data

```python
## SNV-SV-INDEL (n=780+40)
sigs_path =  '/juno/work/shah/users/chois7/tickets/breast-spore-mmctm/fit/results/analysis/signatures/mmctm_props_standardized.tsv'
clust_path = '/juno/work/shah/users/chois7/tickets/breast-spore-mmctm/fit/results/analysis/signatures/mmctm_props_hdbscan_clusters.tsv'

## SNV-SV-INDEL (n=780)
# sigs_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/signatures/_without_complex_svs/mmctm_props_standardized.tsv'
# clust_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/signatures/_without_complex_svs/mmctm_props_hdbscan_clusters.tsv'

## SNV-SV-SVComplex-INDEL (n=780)
# sigs_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/fit/results/analysis/signatures/mmctm_props_standardized.tsv'
# clust_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/fit/results/analysis/signatures/mmctm_props_hdbscan_clusters.tsv'

sigs = pd.read_table(sigs_path, index_col=0).T
clust = pd.read_table(clust_path, index_col=0)

sigs = sigs.join(clust).sort_values(['cluster_id']).T
```

## complex svs

```python
orig_path = '/work/shah/users/leej39/bc_evolution/mmctm/svtm.simplified.120524.v1.tsv'
orig = pd.read_table(orig_path, index_col=0)
```

```python
orig.index.name = 'term'
```

```python
dst = orig.T
```

```python
dst_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/counts/sv_counts.tsv'
dst.to_csv(dst_path, sep='\t')
```

```python
samples = list(dst.columns)
samples_path = '/juno/work/shah/users/chois7/tickets/mmctm-pipeline-gdan/resources/samples.txt'
with open(samples_path, 'w') as out:
    for sample in samples:
        out.write(sample + '\n')
```

## complex sv plot

```python
sv_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/counts/sv_counts.tsv'
sv = pd.read_table(sv_path, index_col=0)
```

```python
sv.T.describe()
```

```python
pdata = sv.iloc[10:-2]
```

```python
fig, ax = plt.subplots(figsize=(15, 3))
ax.bar(x=range(pdata.index.shape[0]), height=pdata.T.describe().loc['mean'])
ax.set_xticks(range(pdata.index.shape[0]));
ax.set_xticklabels(pdata.index);
# ax.set_yscale('log');
plt.xticks(rotation=90);
```

```python
fig, ax = plt.subplots(figsize=(15, 3))
ax.bar(x=range(pdata.index.shape[0]), height=pdata.T.describe().loc['max'])
ax.set_xticks(range(pdata.index.shape[0]));
ax.set_xticklabels(pdata.index);
# ax.set_yscale('log');
plt.xticks(rotation=90);
```

```python
def get_var_sig(var_sig_path, vartype="SNV", debug=False):
    var = pd.read_table(var_sig_path, index_col=3)
    var = var[var.modality == vartype] # SNV, SV, INDEL
    var = var.set_index(var.index.str.replace("->", ">"))
    if debug: print(var.shape)
    if debug: print(var.head())
    return var
```

```python
def plot_simple_sv_spectra(sv, title, ax):
    sv_colors = {
        "simple_del_1": "#B4E3FF",
        "simple_del_2": "#88B3E9",
        "simple_del_3": "#5B83D3",
        "simple_del_4": "#2F53BD",
        "simple_del_5": "#0223A7",
        "simple_dup_1": "#FFB5B5",
        "simple_dup_2": "#E98888",
        "simple_dup_3": "#D35C5C",
        "simple_dup_4": "#BC2F2F",
        "simple_dup_5": "#A60202",
        "foldback_inv": "tab:orange",
        "unbalanced_tra": "#D9ABFF",
        "reciprocal_inv": "#007527",
        "reciprocal_tra": "#7E11C7",
        "templated_ins": "#ABFFB5",
        "line1_ins": "tab:green",
        "incomplete": "tab:grey",
    }
    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    terms = sv.index
    for svtype, row in sv.iterrows():
        value = row['value']
        prob = row['probability']
        ax.bar(x=[value], height=[prob], color=sv_colors[svtype], label=svtype)

    ax.set_xticks(sv['value'])
    ax.set_xticklabels(terms, fontproperties=font, rotation=90)
    for xtl in ax.xaxis.get_ticklabels():
        if '_amp' in xtl.get_text():
            xtl.set_color('red')

    ax.set_xlim((0.5, sv.shape[0]+1))
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7, ncol=2)

    sns.despine(trim=True, ax=ax)
    return df
```

```python
def plot_complex_sv_spectra(sv, title, ax):
    sv_colors = {
        "double_minute": "tab:pink",
        "small_uni_not": "#FFDFA7",
        "small_uni_amp": "#EEBF90",
        "small_oligo_not": "#DD9F79",
        "small_oligo_amp": "#CB8063",
        "small_multi_not": "#BA604C",
        "small_multi_amp": "#A94035",
        "med_uni_not": "#6CC257",
        "med_uni_amp": "#58AB51",
        "med_oligo_not": "#44944B",
        "med_oligo_amp": "#317C44",
        "med_multi_not": "#1D653E",
        "med_multi_amp": "#094E38",
        "large_uni_not": "#7454DD",
        "large_uni_amp": "#6344C0",
        "large_oligo_not": "#5334A2",
        "large_oligo_amp": "#422585",
        "large_multi_not": "#321567",
        "large_multi_amp": "#21054A",
    }
    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    terms = sv.index
    for svtype, row in sv.iterrows():
        value = row['value']
        prob = row['probability']
        ax.bar(x=[value], height=[prob], color=sv_colors[svtype], label=svtype)

    ax.set_xticks(sv['value'])
    ax.set_xticklabels(terms, fontproperties=font, rotation=90)
    for xtl in ax.xaxis.get_ticklabels():
        if '_amp' in xtl.get_text():
            xtl.set_color('red')

    ax.set_xlim((0.5, sv.shape[0]+1))
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7, ncol=3)

    sns.despine(trim=True, ax=ax)
    return df
```

```python
def plot_sv_spectra(sv, title, ax):
    sv_colors = {
        "simple_del_1": "#B4E3FF",
        "simple_del_2": "#88B3E9",
        "simple_del_3": "#5B83D3",
        "simple_del_4": "#2F53BD",
        "simple_del_5": "#0223A7",
        "simple_dup_1": "#FFB5B5",
        "simple_dup_2": "#E98888",
        "simple_dup_3": "#D35C5C",
        "simple_dup_4": "#BC2F2F",
        "simple_dup_5": "#A60202",
        "foldback_inv": "tab:orange",
        "unbalanced_tra": "#D9ABFF",
        "reciprocal_inv": "#007527",
        "reciprocal_tra": "#7E11C7",
        "templated_ins": "#ABFFB5",
        "double_minute": "tab:brown",
        "small_uni_not": "#FFDFA7",
        "small_uni_amp": "#EEBF90",
        "small_oligo_not": "#DD9F79",
        "small_oligo_amp": "#CB8063",
        "small_multi_not": "#BA604C",
        "small_multi_amp": "#A94035",
        "med_uni_not": "#6CC257",
        "med_uni_amp": "#58AB51",
        "med_oligo_not": "#44944B",
        "med_oligo_amp": "#317C44",
        "med_multi_not": "#1D653E",
        "med_multi_amp": "#094E38",
        "large_uni_not": "#7454DD",
        "large_uni_amp": "#6344C0",
        "large_oligo_not": "#5334A2",
        "large_oligo_amp": "#422585",
        "large_multi_not": "#321567",
        "large_multi_amp": "#21054A",
        "line1_ins": "tab:green",
        "incomplete": "tab:grey",
    }
    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    terms = sv.index
    for svtype, row in sv.iterrows():
        value = row['value']
        prob = row['probability']
        ax.bar(x=[value], height=[prob], color=sv_colors[svtype], label=svtype)

    ax.set_xticks(sv['value'])
    ax.set_xticklabels(terms, fontproperties=font, rotation=90)
    for xtl in ax.xaxis.get_ticklabels():
        if '_amp' in xtl.get_text():
            xtl.set_color('red')

    ax.set_xlim((0.5, sv.shape[0]+1))
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7, ncol=3)

    sns.despine(trim=True, ax=ax)
    return df
```

```python
sigs_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/train/results/analysis/model/SNV10_SV4_SVComplex2_INDEL6/model_sigs.tsv'
sv = get_var_sig(sigs_path, vartype='SVComplex')
n_sv_topics = sv.topic.unique().shape[0]
fig, axes = plt.subplots(n_sv_topics, 1, figsize=(7.5, 3 * n_sv_topics))
for ix, topic in enumerate(sv.topic.unique()):
    ax = axes[ix]
    topic_sv = sv[sv['topic']==topic].copy()
    title = f"SV topic {topic}"
    ax.set_title(title)
    df = plot_complex_sv_spectra(topic_sv, title, ax)
    # break
plt.tight_layout()
```

```python
sigs_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/train/results/analysis/model/SNV10_SV8_SVComplex8_INDEL6/model_sigs.tsv'
sv = get_var_sig(sigs_path, vartype='SV')
n_sv_topics = sv.topic.unique().shape[0]
fig, axes = plt.subplots(n_sv_topics, 1, figsize=(6, 3 * n_sv_topics))
for ix, topic in enumerate(sv.topic.unique()):
    ax = axes[ix]
    topic_sv = sv[sv['topic']==topic].copy()
    title = f"SV topic {topic}"
    ax.set_title(title)
    df = plot_simple_sv_spectra(topic_sv, title, ax)
    # break
plt.tight_layout()
```

## match samples to 780

```python
snv_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/counts/782_samples/snv_counts.tsv'
snv_out_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/counts/snv_counts.tsv'
snv = pd.read_table(snv_path, index_col=0)
snv = snv[samples]
snv.to_csv(snv_out_path, sep='\t')
```

```python
indel_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/counts/782_samples/indel_counts.tsv'
indel_out_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm/train/results/analysis/counts/indel_counts.tsv'
indel = pd.read_table(indel_path, index_col=0)
indel = indel[samples]
indel.to_csv(indel_out_path, sep='\t')
```

## seperate complex svs from merged svs table

```python
svs_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/train/results/analysis/counts/_sv_counts.tsv'
svs = pd.read_table(svs_path, index_col=0)
```

```python
simples_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/train/results/analysis/counts/sv_counts.tsv'
simples = svs.iloc[list(range(15)) + [-2, -1]]
simples.to_csv(simples_path, sep='\t', index=True)
```

```python
complexes_path = '/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/train/results/analysis/counts/svcomplex_counts.tsv'
complexes = svs.iloc[list(range(15, svs.shape[0]-2))]
complexes.to_csv(complexes_path, sep='\t', index=True)
```

# Proc

```python
sigs.std(axis=1)
```

```python
plot_data = sigs.copy()
png_path = None #f'/juno/work/shah/users/chois7/tickets/breast-mmctm/fit/results/plots/SNV9_SV13_INDEL6/cluster.all.png'
# png_path = f'/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/fit/results/plots/SNV10_SV8_SVComplex10_INDEL6/cluster.all.png'
plot_heatmap(plot_data, out_path=png_path)
```

```python
plot_data = sigs.copy()
# png_path = f'/juno/work/shah/users/chois7/tickets/breast-mmctm/fit/results/plots/SNV9_SV13_INDEL6/cluster.all.png'
png_path = f'/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/fit/results/plots/SNV10_SV8_SVComplex10_INDEL6/cluster.all.png'
plot_heatmap(plot_data, out_path=png_path)
```

# Plot heatmap

```python
plot_data = sigs.copy()
png_path = None
# png_path = f'/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/fit/results/plots/SNV10_SV8_SVComplex10_INDEL6/cluster.all.png'
plot_heatmap(plot_data, out_path=png_path)
```

```python
plot_data = sigs.copy()
# png_path = f'/juno/work/shah/users/chois7/tickets/breast-mmctm-complex/fit/results/plots/SNV10_SV8_SVComplex10_INDEL6/cluster.all.png'
plot_heatmap(plot_data)
```

## make clinical annotation

```python
clin_path = '/juno/work/shah/users/leej39/bc_evolution/BC780/List.patients_summary.final.txt'
clin = pd.read_table(clin_path)
clin['submitted_sample_id'] = clin['submitted_sample_id'].str.slice(0, 12)
```

```python
hist_map = dict(zip(clin['submitted_sample_id'], clin['histology']))
```

```python
site_map = dict(zip(clin['submitted_sample_id'], clin['tumor_site']))
```

```python
gender_map = dict(zip(clin['submitted_sample_id'], clin['gender']))
```

```python
age_map = dict(zip(clin['submitted_sample_id'], clin['age']))
```

```python
er_map = dict(zip(clin['submitted_sample_id'], clin['er_ihc']))
pr_map = dict(zip(clin['submitted_sample_id'], clin['pr_ihc']))
her2_map = dict(zip(clin['submitted_sample_id'], clin['her2_ihc']))
```

```python
annot = pd.DataFrame(index=sigs.columns)
annot['age'] = annot.index.map(age_map)
annot['Sex'] = annot.index.map(gender_map)
annot['site'] = annot.index.map(site_map)
annot['histology'] = annot.index.map(hist_map).str.strip()
annot['ER'] = annot.index.map(er_map)
annot['PR'] = annot.index.map(pr_map)
annot['HER2'] = annot.index.map(her2_map)
```

```python
age_q25, age_q50, age_q75 = annot['age'].quantile(0.25), annot['age'].quantile(0.5), annot['age'].quantile(0.75)
annot.loc[annot['age'].isna(), 'Age'] = -1
annot.loc[annot['age'] < age_q25, 'Age'] = 0
annot.loc[annot['age'] >= age_q25, 'Age'] = 1
annot.loc[annot['age'] >= age_q50, 'Age'] = 2
annot.loc[annot['age'] >= age_q75, 'Age'] = 3
```

```python
hist_nans = ['unknown', 'no_data_supplied']
hist_ductal = ['ductal', 'Infiltrating duct carcinoma', 'Breast Invasive Ductal Carcinoma', 'Breast ductal carcinoma in situ',
               'Breast invasive ductal carcinoma', 'Breast Ductal Carcinoma In Situ', 'Invasive Breast Carcinoma']
hist_lobular = ['lobular', 'Lobular carcinoma', 'lobular_Pleomorphic', 'Breast Invasive Lobular Carcinoma']
hist_inflamm = ['unspecified inflammatory breast cancer']
hist_mucinous = ['mucinous', 'Mucinous adenocarcinoma', 'Breast Invasive Mixed Mucinous Carcinoma']
hist_metaplastic = ['Carcinoma with chondroid metaplasia', 'Carcinoma with osseous metaplasia', 'Epithelial type metaplastic breast cancer']
annot['Histology'] = annot['histology'].copy()
annot.loc[annot['histology'].isin(hist_nans), 'Histology'] = 'Unknown'
annot.loc[annot['histology'].isin(hist_ductal), 'Histology'] = 'Ductal'
annot.loc[annot['histology'].isin(hist_lobular), 'Histology'] = 'Lobular'
annot.loc[annot['histology'].isin(hist_inflamm), 'Histology'] = 'Inflammatory'
annot.loc[annot['histology'].isin(hist_mucinous), 'Histology'] = 'Mucinous'
annot.loc[annot['histology'].str.count('etaplas') > 0, 'Histology'] = 'Metaplastic'
annot.loc[annot['histology'].str.count('pocrine') > 0, 'Histology'] = 'Apocrine'
# annot.loc[annot['histology'].str.count('pocrine') > 0, 'Histology'] = 'Apocrine'
annot['Histology'] = annot['Histology'].str.capitalize()
annot['Histology'] = annot['Histology'].str.replace('_', ' ').str.replace(' carcinoma', '').str.replace(' carcinonoma', '')
```

```python
ihc_nans = ['unknown', 'no_data_supplied', 'indeterminate']
annot['ER IHC'] = annot['ER'].copy()
annot.loc[annot['ER'].isin(ihc_nans), 'ER IHC'] = 'Unknown'
annot['ER IHC'] = annot['ER IHC'].str.capitalize()

annot['PR IHC'] = annot['PR'].copy()
annot.loc[annot['PR'].isin(ihc_nans), 'PR IHC'] = 'Unknown'
annot['PR IHC'] = annot['PR IHC'].str.capitalize()

annot['HER2 IHC'] = annot['HER2'].copy()
annot.loc[annot['HER2'].isin(ihc_nans), 'HER2 IHC'] = 'Unknown'
annot.loc[annot['HER2'].str.count('equivocal') > 0, 'HER2 IHC'] = 'Unknown'
annot.loc[annot['HER2'].str.count('\+') > 0, 'HER2 IHC'] = 'Positive'
annot['HER2 IHC'] = annot['HER2 IHC'].str.capitalize()
```

```python
annot['Site'] = annot['site'].str.replace('_', ' ').str.capitalize()
```

```python
annot_cols = ['Age', 'Sex', 'Site', 'Histology', 'ER IHC', 'PR IHC', 'HER2 IHC']
annot = annot[annot_cols]
```

```python
annot.head()
```

## make germline mutation annotation

```python
germ_files = glob.glob('/juno/work/shah/users/leej39/bc_evolution/BC780/driver_germline/*.tsv')
```

```python
germ_genes = ['BRCA1', 'BRCA2']
for gene in germ_genes:
    annot[gene] = 0
    
for germ_file in germ_files:
    _, fname = os.path.split(germ_file)
    sample = fname.replace('.driver.catalog.germline.tsv', '')
    mut = pd.read_table(germ_file)
    if mut.shape[0] > 0: 
        for rix, row in mut.iterrows():
            gene = row['gene']
            if gene in germ_genes:
                annot.loc[sample, gene] = 1
```

```python
annot = annot.dropna()
```

## make somatic mutation annotation

```python
som_files = glob.glob('/juno/work/shah/users/leej39/bc_evolution/BC780/driver_somatic/*.tsv')
```

```python
som_genes = ["TP53", "PIK3CA", "ERBB2", "KMT2C", "PTEN", "GATA3", "CCND1", "MYC", "ZNF703", "ZNF217"]
for gene in som_genes:
    annot[gene] = 0
    
for som_file in som_files:
    _, fname = os.path.split(som_file)
    sample = fname.replace('.driver.catalog.somatic.tsv', '')
    mut = pd.read_table(som_file)
    mut = mut[mut['likelihoodMethod'] != 'DNDS']
    if mut.shape[0] > 0: 
        for rix, row in mut.iterrows():
            gene = row['gene']
            if gene in som_genes:
                annot.loc[sample, gene] = 1
```

```python
annot = annot.dropna()
```

## map string to numeric

```python
nmaps = {
    'Sex': {'female': 1, 'male': 2},
    'Site': {'Unknown': -2, 'Primary': 0, 'Local recurrence': 1, 'Metastasis': 2},
    'Histology': {
        'Unknown': -1,
        'Ductal': 1, 'Lobular': 2, 'Inflammatory': 3,
        'Mucinous': 4, 'Metaplastic': 5, 'Micropapillary': 6,
        'Cribriform tubular': 7, 'Apocrine': 8, 'Papillary': 9,
        'Duct micropapillary': 10, 'Breast mixed ductal and lobular': 11,
        'Medullary': 12, 'Adenoid cystic': 13, 'Neuroendocrine': 14
    },
    'ER IHC': {'Unknown': -1, 'Negative': 0, 'Positive': 1},
    'PR IHC': {'Unknown': -1, 'Negative': 0, 'Positive': 1},
    'HER2 IHC': {'Unknown': -1, 'Negative': 0, 'Positive': 1}
}
```

```python
for col in annot_cols:
    if col == 'Age': continue
    annot[col] = annot[col].replace(nmaps[col])
```

## make genomic annotation

```python
geno_path = '/juno/work/shah/users/leej39/bc_evolution/hrdf.chromothripsis.bridge.annotated.tsv'
geno = pd.read_table(geno_path, index_col=2)
```

```python
annot = annot.join(geno[['wctx', 'tbamp']]).fillna(-1)
annot = annot.rename(columns={'wctx': 'Chromothripsis', 
                              'tbamp': 'T-B Amp'})
```

## add annotation for SPORE

```python
annot.columns
```

```python
annot
```

### chromothripsis and tb-amp for spore

```python
tb_path = '/work/shah/users/leej39/bc_evolution/spore.chromothripsis.bridge.annotated.tsv'
tb = pd.read_table(tb_path)
tb = tb.set_index('study_id')
```

```python
tb.head()
```

### isabl metadata for spore

```python
meta_path = '/juno/work/shah/users/chois7/projects/breast-SPORE/resources/isabl_meta_20240109-132631.csv'
meta = pd.read_csv(meta_path)
meta = meta[['individual_identifier', 'individual_gender']].drop_duplicates().reset_index(drop=True)
meta.columns = ['patient', 'sex']
```

```python
meta = meta[meta['patient'].isin(sigs.columns)]
meta['sex'] = meta['sex'].replace({'FEMALE':1, 'MALE':2})
```

### SNV indel for spore


### CNA for spore

```python
cna_genes = ['ERBB2', 'PTEN', 'CCND1', 'MYC', 'ZNF703', 'ZNF217']
som_genes = ['TP53', 'PIK3CA', 'KMT2C', 'GATA3']
```

```python
cna_path = '/juno/work/shah/users/chois7/projects/breast-SPORE/resources/cna_renamed.tsv'
cna = pd.read_table(cna_path, index_col=0)
```

```python
cna = cna[cna.index.isin(cna_genes)].replace({1:0, -1:0, 2:1, -2:1})
```

```python
cna.columns = cna.columns.str.replace('_T$','').str.replace('T$', '')
```

```python
cna = cna[meta['patient']]
```

```python
cnat = cna.T
```

### create spore annot dataframe

```python
tannot = annot.T # row:feature, column:sample
```

```python
spore = pd.DataFrame(columns=annot.columns)
for rix, row in meta.iterrows():
    patient, sex = row.squeeze()
    field = [-1] * len(annot.columns)
    field[0] = 2
    field[1] = sex
    wctx = tb.loc[patient, 'wctx']
    tbamp = tb.loc[patient, 'tbamp']
    field[-2] = wctx
    field[-1] = tbamp
    spore.loc[patient] = field
```

```python
spore.head()
```

```python
for gene in cnat.columns:
    spore[gene] = cnat[gene]
```

```python
spore
```

## apply clinical annotation

```python
plot_annot = annot.T
plot_annot = plot_annot.div(plot_annot.max(axis=1), axis=0)
```

```python
plot_heatmap_with_annot(plot_data, plot_annot)
```

```python
annot['Site'].value_counts(dropna=False)
```

```python
df.values.ravel('K')
```

```python
df_numeric
```

```python

```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Example DataFrame
data = {'col1': ['apple', 'banana', 'cherry'],
        'col2': ['banana', 'apple', 'apple'],
        'col3': ['cherry', 'cherry', 'banana']}
df = pd.DataFrame(data)

# Convert string objects to numeric values
unique_strings = pd.unique(df.values.ravel('K'))
string_to_num = {string: i for i, string in enumerate(unique_strings)}

# Apply the mapping
df_numeric = df.applymap(lambda x: string_to_num[x])

# Normalize the numeric values
norm = Normalize(vmin=df_numeric.min().min(), vmax=df_numeric.max().max())

# Create the heatmap
plt.imshow(df_numeric, cmap='tab20', norm=norm)
plt.colorbar()
```

# Selection

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

```python
in_dir = '/juno/work/shah/users/chois7/tickets/breast-spore-mmctm/train/results/analysis/model'
```

## show signatures

```python
%matplotlib inline
snv_k, sv_k, indel_k = 10, 14, 8
tag = f'SNV{snv_k}_SV{sv_k}_INDEL{indel_k}'
snv_png = mpimg.imread(f'{in_dir}/{tag}/model_sigs.SNV.png')
sv_png = mpimg.imread(f'{in_dir}/{tag}/model_sigs.SV.png')
indel_png = mpimg.imread(f'{in_dir}/{tag}/model_sigs.INDEL.png')
```

```python
print(tag)
```

```python
fig, ax = plt.subplots(figsize=(15, 20))
ax.imshow(snv_png)
ax.set_frame_on(False)
ax.set_xticks([]); ax.set_yticks([]);
```

```python
fig, ax = plt.subplots(figsize=(15, 20))
ax.imshow(indel_png)
ax.set_frame_on(False)
ax.set_xticks([]); ax.set_yticks([]);
```

```python
fig, ax = plt.subplots(figsize=(15, 20))
ax.imshow(sv_png)
ax.set_frame_on(False)
ax.set_xticks([]); ax.set_yticks([]);
```

# Clust

```python
for cluster_id in range(1, 11+1):
    ssigs = sigs.loc[:, sigs.loc['cluster_id'] == cluster_id]
    ssigs.shape
    props = ssigs.iloc[:-1, :]

    n_neighbors = 4
    min_samples = 4
    cluster_selection_epsilon = 0.5

    embeddings = reduce_dimensions(props, n_neighbors)
    clusters, tree, clusterer = cluster_sample_embeddings(
        embeddings, min_samples, 5, cluster_selection_epsilon
    ) 
    props = props.T.join(clusters).sort_values(['cluster_id']).T
    png_path = f'/juno/work/shah/users/chois7/tickets/breast-mmctm/fit/results/plots/SNV9_SV13_INDEL6/cluster.{cluster_id}.png'
    plot_data = props.copy()
    nrow, ncol = props.shape
    figsize = (ncol / 3.3, nrow / 3)
    plot_heatmap(plot_data, figsize=figsize, out_path=png_path)
```

```python

```
