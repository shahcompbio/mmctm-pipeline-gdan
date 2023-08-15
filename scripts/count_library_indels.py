from argparse import ArgumentParser
import pandas as pd
import dvartk


def get_args():
    description = 'Count library indels'
    p = ArgumentParser(description=description)

    p.add_argument('indels', help='indel tsv file')
    p.add_argument('ref_version', help='reference sequence version')
    p.add_argument('counts', help='output count txt file')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    indels = pd.read_csv(args.indels, sep='\t', dtype={'chrom': str})
    genome_version = args.ref_version

    counts = dvartk.count_indels(indels, genome_version)
    counts.to_csv(args.counts, sep='\t', header=False)
