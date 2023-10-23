import pandas as pd
import fire


def parse_svs(consensus_calls, output_svs):
    """ Filter SVs using remixt and thresholds.
    """

    dtypes = {'chromosome_1': str, 'chromosome_2': str}
    try:
        sv = pd.read_csv(consensus_calls, dtype=dtypes)
    except OSError:
        sv = pd.read_csv(consensus_calls, dtype=dtypes, compression=None)
    sv = sv[
        ~sv['is_germline'] &
        ~sv['is_filtered'] &
        # ~sv['is_low_mappability'] &
        pd.isnull(sv['dgv_ids']) &
        (sv['template_length_min'] >= 200) &
        (sv['num_unique_reads'] >= 5) &
        (sv['num_split'] >= 2)
    ]
    sv.loc[sv['rearrangement_type'] == 'foldback', 'type'] = 'foldback'
    sv = sv[[
        'chromosome_1', 'position_1', 'chromosome_2', 'position_2',
        'homology', 'break_distance', 'type'
    ]].drop_duplicates()
    sv.columns = [
        'chrom_1', 'brk_1', 'chrom_2', 'brk_2', 'homlen', 'brk_dist',
        'type'
    ]
    sv.to_csv(output_svs, sep='\t', index=False)


if __name__ == '__main__':
    fire.Fire(parse_svs)
