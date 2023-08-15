import os
import subprocess
import pandas as pd

if not os.path.exists(config['intermediate_dir']): subprocess.run(f'mkdir -p {config["intermediate_dir"]}', shell=True)
if not os.path.exists(config['log_dir']): subprocess.run(f'mkdir -p {config["log_dir"]}', shell=True)

samples = [s.strip() for s in open(config['samples_file']).readlines()]
types = ['snv', 'indel', 'sv']

wildcard_constraints:
    type='snv|indel|sv',

rule all:
    input:
        expand(os.path.join(config['results_dir'], 'analysis/counts/{type}_counts.tsv'), type=types)

rule vcf_to_tsv:
    input:
        os.path.join(config['source_dir'], 'vcf/{sample}.vcf'),
    output:
        snvs=os.path.join(config['intermediate_dir'], 'analysis/process_variants/snv/sample/{sample}.tsv'),
        indels=os.path.join(config['intermediate_dir'], 'analysis/process_variants/indel/sample/{sample}.tsv'),
    singularity: "docker://amcpherson/mmctm-pythonscripts:v0.1"
    shell: 'python scripts/vcf_to_tsv.py {input} {output.snvs} {output.indels}'

rule sample_snv_counts:
    input:
        tsv=os.path.join(config['intermediate_dir'], 'analysis/process_variants/snv/sample/{sample}.tsv'),
        ref=config['reference_fasta'],
    output: os.path.join(config['intermediate_dir'], 'analysis/counts/snv/sample/{sample}.tsv'),
    singularity: "docker://amcpherson/mmctm-pythonscripts:v0.1",
    shell: 'python scripts/count_library_snvs.py {input.tsv} {input.ref} {output}'

rule sample_indel_counts:
    input:
        os.path.join(config['intermediate_dir'], 'analysis/process_variants/indel/sample/{sample}.tsv'),
    output: os.path.join(config['intermediate_dir'], 'analysis/counts/indel/sample/{sample}.tsv'),
    params: ref_version="GRCh38",
    shell: 'python scripts/count_library_indels.py {input} {params.ref_version} {output}' # TODO: add singularity image


def _get_sv_input_paths(wildcards):
    meta_path = config['breakpointcalling_metadata']
    df = pd.read_table(meta_path)
    df = df[df['result_type']=='consensus_calls']
    df = df[df['isabl_sample_id']==wildcards.sample]
    assert df.shape[0] == 1, df
    path = df['result_filepath'].tolist()[0]
    return path

rule complete_svs:
    input: consensus_calls=_get_sv_input_paths,
    output: os.path.join(config['intermediate_dir'], 'analysis/process_variants/sv/complete/{sample}.tsv')
    singularity: "docker://amcpherson/mmctm-pythonscripts:v0.1"
    shell: 'python scripts/filter_svs.py {input.consensus_calls} {output}'

rule sv_cluster_annotations:
    input: os.path.join(config['intermediate_dir'], 'analysis/process_variants/sv/complete/{sample}.tsv')
    output: os.path.join(config['intermediate_dir'], 'analysis/process_variants/sv/sample/{sample}.tsv')
    singularity: "docker://amcpherson/mmctm-rscripts:v0.1"
    shell:
        '''
        Rscript scripts/annotate_clustered_svs.R \
            {input} {output}
        '''

rule sample_sv_counts:
    input: os.path.join(config['intermediate_dir'], 'analysis/process_variants/sv/sample/{sample}.tsv'),
    output: os.path.join(config['intermediate_dir'], 'analysis/counts/sv/sample/{sample}.tsv'),
    singularity: "docker://amcpherson/mmctm-pythonscripts:v0.1",
    shell: 'python scripts/count_library_svs.py {input} {output}'

rule group_counts:
    input:
        expand(
            os.path.join(config['intermediate_dir'], 'analysis/counts/{{type}}/sample/{sample}.tsv'),
            sample=samples,
        )
    output: os.path.join(config['results_dir'], 'analysis/counts/{type}_counts.tsv')
    run:
        pieces = []
        for f in input:
            sample_id = os.path.splitext(os.path.basename(f))[0]
            sample_counts = pd.read_csv(f, sep='\t', index_col=0, header=None)
            sample_counts.columns = [sample_id]
            pieces.append(sample_counts)
        counts = pd.concat(pieces, axis=1)
        counts.index.name = 'term'
        counts.to_csv(output[0], sep='\t')