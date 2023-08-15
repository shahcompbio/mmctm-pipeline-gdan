from argparse import ArgumentParser
import numpy as np
import pandas as pd
import warnings
from pyfaidx import Fasta

np.random.seed(3450809821)


def get_args():
    description = 'Count library SNVs'
    p = ArgumentParser(description=description)

    p.add_argument('snvs', help='SNV tsv file')
    p.add_argument('ref', help='Reference sequence fasta file')
    p.add_argument('counts', help='output count txt file')

    return p.parse_args()


def construct_empty_count_series():
    snvs = ['C->A', 'C->G', 'C->T', 'T->A', 'T->C', 'T->G']
    nts = ['A', 'C', 'G', 'T']
    terms = [
        '{}[{}]{}'.format(l, s, r) for s in snvs for l in nts for r in nts
    ]
    return pd.Series(np.zeros(len(terms), dtype=int), index=terms)


def normalize_snv(context, alt):
    ref = context.seq[1]

    if ref in ['A', 'G']:
        context = (-context).seq

        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        alt = complement[str(alt)]
    else:
        context = context.seq
        alt = str(alt)
    return context, alt


def construct_snv_label(context, alt):
    if len(context) != 3:
        warnings.warn('Warning: bad context length: {}'.format(str(context)))
        return None
    return '{}[{}->{}]{}'.format(context[0], context[1], alt, context[2])


def count_snvs(snvs, genome):
    counts = construct_empty_count_series()

    for idx, row in snvs.iterrows():
        # two flanking bases
        start = row['pos'] - 2
        end = row['pos'] + 1

        context = genome[row['chrom']][start:end]
        if 'N' in context.seq:
            warnings.warn(
                'Warning: N found in context sequence at {}:{}-{}'.format(
                    row['chrom'], start + 1, end
                )
            )
            continue

        context, alt = normalize_snv(context, row['alt'])
        counts[construct_snv_label(context, alt)] += 1

    return counts


if __name__ == '__main__':
    argv = get_args()

    snvs = pd.read_csv(argv.snvs, sep='\t', dtype={'chrom': str})
    genome = Fasta(argv.ref)

    counts = count_snvs(snvs, genome)
    counts.to_csv(argv.counts, sep='\t', header=False)
