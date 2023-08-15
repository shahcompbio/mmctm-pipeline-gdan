from argparse import ArgumentParser
import pandas as pd


def get_args():
    description = 'Transform VCF to TSV for snvs and indels'
    p = ArgumentParser(description=description)

    p.add_argument('vcf', help='SNV vcf file')
    p.add_argument('snvs', help='output snvs tsv file')
    p.add_argument('indels', help='output indels tsv file')

    return p.parse_args()

    
if __name__ == '__main__':
    args = get_args()

    bases = {'A', 'C', 'G', 'T'}
    vcf_cols = ['chrom', 'pos', 'ID', 'ref', 'alt', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 's1', 's2']
    out_cols = ['chrom', 'pos', 'ref', 'alt']
    df = pd.read_table(args.vcf, comment='#', names=vcf_cols)
    snv_ix = (df['ref'].isin(bases) & df['alt'].isin(bases))
    indel_ix = (~snv_ix & (df['ref'].str.len() != df['alt'].str.len()))

    snvs = df.loc[snv_ix, out_cols]
    indels = df.loc[indel_ix, out_cols]

    snvs.to_csv(args.snvs, sep='\t', index=False)
    indels.to_csv(args.indels, sep='\t', index=False)

