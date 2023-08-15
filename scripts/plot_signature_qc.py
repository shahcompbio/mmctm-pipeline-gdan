import matplotlib
import pandas as pd
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

def get_cosmic(cosmic_path, debug=False):
    """Get COSMIC data as melted dataframe
    """
    cosmic = pd.read_table(cosmic_path, index_col=0)
    cosmic = cosmic.sort_index()
    if debug: print(cosmic.index)
    cosmic = cosmic.melt(ignore_index=False, var_name="topic", value_name="probability")
    if debug: print(cosmic.shape)
    if debug: print(cosmic.head())
    return cosmic

def get_var_sig(var_sig_path, vartype="SNV", debug=False):
    var = pd.read_table(var_sig_path, index_col=3)
    var = var[var.modality == vartype] # SNV, SV, INDEL
    var = var.set_index(var.index.str.replace("->", ">"))
    if debug: print(var.shape)
    if debug: print(var.head())
    return var

def calc_cosine_similarity(tb1, tb2, debug=False):
    assert hasattr(tb1, "topic")
    assert hasattr(tb2, "topic")
    topics1 = tb1.topic.unique()
    topics2 = tb2.topic.unique()
    if debug: print(topics1)
    if debug: print(topics2)
    if debug: print(len(topics1), len(topics2))
    cosine_simils = np.zeros([len(topics1), len(topics2)])
    for ix, topic_i in enumerate(topics1):
        ttb1 = tb1[tb1.topic==topic_i].sort_index()
        prob1 = ttb1.probability.values
        if debug: print("prob1 shape", prob1.shape)
        for jx, topic_j in enumerate(topics2):
            ttb2 = tb2[tb2.topic==topic_j].sort_index()
            prob2 = ttb2.probability.values
            if debug: print("prob2 shape", prob2.shape)
            assert (ttb1.index == ttb2.index).all()
            cosine_simil = 1 - spatial.distance.cosine(prob1, prob2)
            cosine_simils[ix, jx] = cosine_simil
    return cosine_simils

def draw_heatmap_from_csdf(csdf, xlabel="previous SNV signature", ylabel="current SNV signature", 
                           figsize=(7,6), sort=False, round=True, red_xlabels=[], nrows=0, ncols=0):
    df = csdf.copy()
    plt.figure(figsize=figsize)
    if sort: 
        df = df.loc[:, df.max(axis=0).sort_values(ascending=False).index]
        df = df.loc[df.max(axis=1).sort_values(ascending=False).index, :]
    df = df.round(2)
    if df.max().max() > 1: vmax=100
    else: vmax=1.0
    if nrows == 0: nrows = df.shape[0]
    if ncols == 0: ncols = df.shape[1]
    plot_df = df.iloc[:nrows,:ncols]
    hm = sns.heatmap(plot_df, annot=True, vmin=0, vmax=vmax)
    
    if len(red_xlabels) > 0:
        for xtl in hm.axes.get_xticklabels():
            if xtl.get_text() in red_xlabels:
                xtl.set_color('red')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def get_sig_vs_cosmic_csdf(var_path, snv_cosmic, indel_cosmic, vartype="SNV"):
    """Get signatures vs COSMIC cosine similarity dataframe
    """
    if vartype == "SNV":
        cosmic_path = snv_cosmic
    elif vartype == "INDEL":
        cosmic_path = indel_cosmic
    else: print(f"ERROR: vartype={vartype}"); return None
    cosmic = get_cosmic(cosmic_path)

    current = get_var_sig(var_path, vartype=vartype, debug=False)
    current = current[current.modality==vartype]

    cosine_simils = calc_cosine_similarity(current, cosmic)
    current_vs_cosmic_df = pd.DataFrame(cosine_simils, 
                            index=[snv_topic for snv_topic in current.topic.unique()],
                            columns=[snv_topic for snv_topic in cosmic.topic.unique()])
    return current_vs_cosmic_df



def plot_sv_spectra(sv, title, ax):
    sv_colors = {
        'del:<10kb': '#d6e6f4',
        'del:10kb-100kb': '#abd0e6',
        'del:100kb-1Mb': '#6aaed6',
        'del:1Mb-10Mb': '#3787c0',
        'del:>10Mb': '#105ba4',

        'dup:<10kb': '#fedfc0',
        'dup:10kb-100kb': '#fdb97d',
        'dup:100kb-1Mb': '#fd8c3b',
        'dup:1Mb-10Mb': '#e95e0d',
        'dup:>10Mb': '#b63c02',

        'inv:<10kb': '#dbf1d6',
        'inv:10kb-100kb': '#aedea7',
        'inv:100kb-1Mb': '#73c476',
        'inv:1Mb-10Mb': '#37a055',
        'inv:>10Mb': '#0b7734',

        'fbi:<10kb': '#f14432', 
        'fbi:10kb-100kb': '#bc141a',

        'translocation': '#9467BD'
    }
    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    df = sv.copy()
    pat = r'(.+):(.*):(.+):(.+)' # del:100kb-1Mb:0-1:cl
    df['sv_type'] = df.index.str.replace(pat, r'\1', regex=True) 
    df['sv_length'] = df.index.str.replace(pat, r'\2', regex=True)
    df['sv_plot_ix'] = df.index.str.replace(pat, r'\3:\4', regex=True)
    df['sv_color_ix'] = df[['sv_type', 'sv_length']].agg(':'.join, axis=1)
    df['sv_color_ix'] = df['sv_color_ix'].replace('tr:', 'translocation')
    df['index'] = range(df.shape[0])
    
    for mut_type, mut_type_data in df.groupby(['sv_color_ix'], sort=False):
        ax.bar(data=mut_type_data, x='index', height='probability', label=mut_type,
               color=sv_colors[mut_type])
        
    ax.set_xticks(df['index'])
    ax.set_xticklabels(df['sv_plot_ix'], fontproperties=font, rotation=90)
    for xtl in ax.xaxis.get_ticklabels():
        if ':cl' in xtl.get_text(): 
            xtl.set_color('red')

    ax.set_xlim((-1, 109))
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7)
    
    sns.despine(trim=True, ax=ax)
    return df

def get_args():
    description = 'Make signature probability QC plots'
    p = ArgumentParser(description=description)

    p.add_argument('sigs', help='sigs file')
    p.add_argument('cosmic_snvs', help='cosmic SBS file')
    p.add_argument('cosmic_indels', help='COSMIC ID file')
    p.add_argument('out_snvs_plot', help='SNV plot path')
    p.add_argument('out_indels_plot', help='INDEL plot path')
    p.add_argument('out_svs_plot', help='SV plot path')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    sigs_path = args.sigs
    cosmic_snvs = args.cosmic_snvs
    cosmic_indels = args.cosmic_indels

    # SNV
    csdf = get_sig_vs_cosmic_csdf(sigs_path, cosmic_snvs, cosmic_indels, vartype="SNV")
    draw_heatmap_from_csdf(csdf, xlabel="COSMIC", ylabel="MMCTM", figsize=(12, 5), sort=True, ncols=20)
    plt.tight_layout()
    plt.savefig(args.out_snvs_plot)

    # INDEL
    csdf = get_sig_vs_cosmic_csdf(sigs_path, cosmic_snvs, cosmic_indels, vartype="INDEL")
    draw_heatmap_from_csdf(csdf, xlabel="COSMIC", ylabel="MMCTM", figsize=(12, 4.5), sort=True, ncols=20)
    plt.tight_layout()
    plt.savefig(args.out_indels_plot)

    # SV
    sv = get_var_sig(sigs_path, vartype='SV')
    n_sv_topics = sv.topic.unique().shape[0]
    fig, axes = plt.subplots(n_sv_topics, 1, figsize=(15, 3 * n_sv_topics))
    for ix, topic in enumerate(sv.topic.unique()):
        ax = axes[ix]
        topic_sv = sv[sv['topic']==topic].copy()
        title = f"SV topic {topic}"
        ax.set_title(title)
        df = plot_sv_spectra(topic_sv, title, ax)
    plt.tight_layout()
    fig.savefig(args.out_svs_plot)