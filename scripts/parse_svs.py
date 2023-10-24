import sys
import numpy as np
import pandas as pd
import fire

def save_svs(sv, output_svs):
    sv.to_csv(output_svs, sep='\t', index=False)

def write_empty_output(output_svs, dst_cols):
    with open(output_svs, 'w') as out:
        line = '\t'.join(dst_cols)
        out.write(line + '\n')
    sys.exit(0)

def parse_svs(consensus_calls, output_svs, 
        brk_dist_cutoff=35, foldback_brk_dist_cutoff=30000):
    """ Process SVs from HMFtools to match destruct format.
    """
    dst_cols = [
        'chrom_1', 'brk_1', 'chrom_2', 'brk_2', 'homlen', 'brk_dist', 'type'
    ]
    dtypes = {'chrom1': str, 'chrom2': str}
    try:
        sv = pd.read_table(consensus_calls, dtype=dtypes)
    except OSError:
        sv = pd.read_table(consensus_calls, dtype=dtypes, compression=None)
    if 'chrom2' not in sv.columns:
        write_empty_output(output_svs, dst_cols)

    # classify foldback
    type_map = {'DEL':'deletion', 'DUP':'duplication', 't2tINV':'inversion',
                'h2hINV':'inversion', 'TRA':'translocation'}
    sv['type'] = sv['svclass'].map(type_map)
    sv.loc[
        (sv['type']=='inversion') & (sv['isFoldback']==True),
        'type'
    ] = 'foldback'

    # add brk_dist
    _sv = pd.DataFrame()
    for svtype, svdf in sv.groupby("type"):
        svdf['brk_dist'] = int(3e9)
        if svtype != 'translocation': # filter 1
            svdf['brk_dist'] = svdf['end2'] - svdf['start1']
            svdf = svdf[svdf['brk_dist'] >= brk_dist_cutoff] 
        if svtype == 'foldback': # filter 2
            svdf = svdf[svdf['brk_dist'] < foldback_brk_dist_cutoff]
        _sv = pd.concat([_sv, svdf])
    sv = _sv.copy()

    # subset cols
    sv = sv[[
        'chrom1', 'start1', 'chrom2', 'end2', 'homlen', 'brk_dist', 'type'
    ]].drop_duplicates()
    sv['start1'] += 1 # 0-based to 1-based
    sv.columns = dst_cols

    # filter svs
    save_svs(sv, output_svs)


if __name__ == '__main__':
    fire.Fire(parse_svs)
