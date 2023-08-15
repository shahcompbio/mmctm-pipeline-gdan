from argparse import ArgumentParser
import numpy as np
import pandas as pd

np.random.seed(3450809821)


def get_args():
    description = 'Count SVs'
    p = ArgumentParser(description=description)

    p.add_argument('svs', help='SV tsv file')
    p.add_argument('counts', help='output count txt file')

    return p.parse_args()


def construct_empty_count_series():
    svs = ['del', 'dup', 'inv']
    bkdists = ['<10kb', '10kb-100kb', '100kb-1Mb', '1Mb-10Mb', '>10Mb']
    homlens = ['0-1', '2-5', '>5']
    cls = ['ncl', 'cl']

    terms = [
        '{}:{}:{}:{}'.format(s, d, h, c)
        for s in svs for d in bkdists for h in homlens for c in cls
    ]
    terms += [
        'fbi:{}:{}:{}'.format(d, h, c)
        for d in bkdists[:2] for h in homlens for c in cls
    ]
    terms += ['tr::{}:{}'.format(h, c) for h in homlens for c in cls]

    return pd.Series(np.zeros(len(terms), dtype=int), index=terms)


def get_svtype_label(svtype):
    sv_map = {
        'deletion': 'del', 'duplication': 'dup', 'inversion': 'inv',
        'foldback': 'fbi', 'translocation': 'tr'
    }
    return sv_map[svtype]


def get_bkdist_label(bkdist):
    if bkdist < 10e3:
        return '<10kb'
    elif bkdist < 100e3:
        return '10kb-100kb'
    elif bkdist < 1000e3:
        return '100kb-1Mb'
    elif bkdist < 10000e3:
        return '1Mb-10Mb'
    else:
        return '>10Mb'


def get_homlen_label(homlen):
    if homlen <= 1:
        return '0-1'
    elif homlen <= 5:
        return '2-5'
    else:
        return '>5'


def get_cl_label(cl):
    return ['ncl', 'cl'][cl]


def construct_sv_label(svtype, bkdist, homlen, cl):
    svtype_label = get_svtype_label(svtype)
    bkdist_label = get_bkdist_label(bkdist)
    homlen_label = get_homlen_label(homlen)
    cl_label = get_cl_label(cl)

    if svtype == 'translocation':
        sv_label = 'tr::{}:{}'.format(homlen_label, cl_label)
    else:
        sv_label = '{}:{}:{}:{}'.format(
            svtype_label, bkdist_label, homlen_label, cl_label
        )
    return sv_label


def count_svs(svs):
    counts = construct_empty_count_series()

    for idx, row in svs.iterrows():
        svtype = row['type']
        bkdist = row['brk_dist']
        homlen = row['homlen']
        cl = row['cl']

        counts[construct_sv_label(svtype, bkdist, homlen, cl)] += 1
    return counts


if __name__ == '__main__':
    argv = get_args()

    svs = pd.read_csv(
        argv.svs, sep='\t', dtype={'chrom_1': str, 'chrom_2': str}
    )

    counts = count_svs(svs)
    counts.to_csv(argv.counts, sep='\t', header=False)
