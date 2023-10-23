import numpy as np
import pandas as pd
import fire


def parse_svs(consensus_calls, output_svs):
    """ Process SVs from HMFtools to match destruct format.
    """
    dtypes = {'chrom1': str, 'chrom2': str}
    try:
        sv = pd.read_table(consensus_calls, dtype=dtypes)
    except OSError:
        sv = pd.read_table(consensus_calls, dtype=dtypes, compression=None)

    # add brk_dist
    chrom_diff = (sv['chrom1'] != sv['chrom2'])
    sv['brk_dist'] = int(3e9)
    sv.loc[chrom_diff, 'brk_dist'] = np.abs(
            sv['end2'] - sv['start1'] + 1
        )

    # classify foldback
    type_map = {'DEL':'deletion', 'DUP':'duplication', 't2tINV':'inversion', 
                'h2hINV':'inversion', 'TRA':'translocation'}
    sv['type'] = sv['svclass'].map(type_map)
    sv.loc[sv['isFoldback']=='true', 'type'] = 'foldback'

    # subset cols
    sv = sv[[
        'chrom1', 'start1', 'chrom2', 'end2', 'homlen', 'brk_dist', 'type'
    ]].drop_duplicates()
    sv['start1'] += 1 # 0-based to 1-based
    sv.columns = [
        'chrom_1', 'brk_1', 'chrom_2', 'brk_2', 'homlen', 'brk_dist', 'type'
    ]
    sv.to_csv(output_svs, sep='\t', index=False)


if __name__ == '__main__':
    fire.Fire(parse_svs)
