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
        # expand(os.path.join(config['results_dir'], 'analysis/counts/{type}_counts.tsv'), type=types)
        os.path.join(config['results_dir'], 'analysis/model/model_props.tsv'),

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
#BSUB -n 30
#BSUB -o log/{{log_dir}}/{{snv_k}}-{{sv_k}}-{{indel_k}}.out
#BSUB -e log/{{log_dir}}/{{snv_k}}-{{sv_k}}-{{indel_k}}.err
#BSUB -J ac_{{snv_k}}-{{sv_k}}-{{indel_k}}
#BSUB -W 12:00
rule train_mmctm:
    input: 
        snv=os.path.join(config['results_dir'], 'analysis/counts/snv_counts.tsv'),
        sv=os.path.join(config['results_dir'], 'analysis/counts/sv_counts.tsv'),
        indel=os.path.join(config['results_dir'], 'analysis/counts/indel_counts.tsv'),
    output:
        jld=os.path.join(config['results_dir'], 'analysis/model/model.jld'),
        cor=os.path.join(config['results_dir'], 'analysis/model/model_cor.tsv'),
        mean=os.path.join(config['results_dir'], 'analysis/model/model_mean.tsv'),
        sigs=os.path.join(config['results_dir'], 'analysis/model/model_sigs.tsv'),
        props=os.path.join(config['results_dir'], 'analysis/model/model_props.tsv'),
    params:
        snv_k = config['snv_k'], 
        sv_k = config['sv_k'], 
        indel_k = config['indel_k'],
        modalities = "SNV SV INDEL",
        threads = 30,
    singularity: "library://soymintc/julia/mmctm-jl_1.6.5:latest",
    threads: 30,
    shell:
        'julia -p {params.threads} scripts/run_mmctm.jl '
        '{input.snv} {input.sv} {input.indel} '
        '-r 1500 -v --progress '
        '--modality-labels {params.modalities} '
        '-k {params.snv_k} {params.sv_k} {params.indel_k} '
        '--model {output.jld} '
        '--cor {output.cor} '
        '--mean {output.mean} '
        '--sigs {output.sigs} '
        '--props {output.props} '