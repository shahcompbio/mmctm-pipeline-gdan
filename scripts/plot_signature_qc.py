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
    return sv


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
    fig, axes = plt.subplots(n_sv_topics, 1, figsize=(10, 3 * n_sv_topics))
    for ix, topic in enumerate(sv.topic.unique()):
        ax = axes[ix]
        topic_sv = sv[sv['topic']==topic].copy()
        title = f"SV topic {topic}"
        ax.set_title(title)
        df = plot_sv_spectra(topic_sv, title, ax)
    plt.tight_layout()
    fig.savefig(args.out_svs_plot)
