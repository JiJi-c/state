import os
import fire
import asyncio
import logging

import torch
import h5py as h5
import pandas as pd
import urllib.request

from pathlib import Path
from functools import partial
from ast import literal_eval

from vci.data.preprocess import Preprocessor
from vci.preprocessing import ESMEmbedding, Evo2Embedding
from vci.data.gene_emb import parse_genome_for_gene_seq_map, resolve_genes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

ref_genome_file_type = 'cdna' # 'cdna'

scBasecamp_dir_species = {
    "arabidopsis_thaliana":     {"name": "Arabidopsis_thaliana",
                                 "bio_class": "plantae",
                                 "ref_genome": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/{ref_genome_file_type}/Arabidopsis_thaliana.TAIR10.{ref_genome_file_type}.all.fa.gz"},
    "bos_taurus":               {"name": "Bos_taurus",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/bos_taurus/{ref_genome_file_type}/Bos_taurus.ARS-UCD1.3.{ref_genome_file_type}.all.fa.gz"},
    "caenorhabditis_elegans":   {"name": "Caenorhabditis_elegans",
                                 "bio_class": "invertebrate",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/caenorhabditis_elegans/{ref_genome_file_type}/Caenorhabditis_elegans.WBcel235.{ref_genome_file_type}.all.fa.gz"},
    "callithrix_jacchus":       {"name": "Callithrix_jacchus",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/callithrix_jacchus/{ref_genome_file_type}/Callithrix_jacchus.mCalJac1.pat.X.{ref_genome_file_type}.all.fa.gz"},
    "danio_rerio":              {"name": "Danio_rerio",
                                 "bio_class": "fish",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/danio_rerio/{ref_genome_file_type}/Danio_rerio.GRCz11.{ref_genome_file_type}.all.fa.gz"},
    "drosophila_melanogaster":  {"name": "Drosophila_melanogaster",
                                 "bio_class": "insecta",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/drosophila_melanogaster/{ref_genome_file_type}/Drosophila_melanogaster.BDGP6.46.{ref_genome_file_type}.all.fa.gz"},
    "equus_caballus":           {"name": "Equus_caballus",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/equus_caballus/{ref_genome_file_type}/Equus_caballus.EquCab3.0.{ref_genome_file_type}.all.fa.gz"},
    "gallus_gallus":            {"name": "Gallus_gallus",
                                 "bio_class": "aves",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/gallus_gallus/{ref_genome_file_type}/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.{ref_genome_file_type}.all.fa.gz"},
    "gorilla_gorilla":          {"name": "Gorilla_gorilla",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/gorilla_gorilla/{ref_genome_file_type}/Gorilla_gorilla.gorGor4.{ref_genome_file_type}.all.fa.gz"},
    "heterocephalus_glaber":    {"name": "Heterocephalus_glaber",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/heterocephalus_glaber_female/{ref_genome_file_type}/Heterocephalus_glaber_female.Naked_mole-rat_maternal.{ref_genome_file_type}.all.fa.gz"},
    "homo_sapiens":             {"name": "Homo_sapiens",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/{ref_genome_file_type}/Homo_sapiens.GRCh38.{ref_genome_file_type}.all.fa.gz"},
    "macaca_mulatta":           {"name": "Macaca_mulatta",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/macaca_mulatta/{ref_genome_file_type}/Macaca_mulatta.Mmul_10.{ref_genome_file_type}.all.fa.gz"},
    "mus_musculus":             {"name": "Mus_musculus",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/{ref_genome_file_type}/Mus_musculus.GRCm39.{ref_genome_file_type}.all.fa.gz"},
    "oryctolagus_cuniculus":    {"name": "Oryctolagus_cuniculus",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/oryctolagus_cuniculus/{ref_genome_file_type}/Oryctolagus_cuniculus.OryCun2.0.{ref_genome_file_type}.all.fa.gz"},
    "oryza_sativa":             {"name": "Oryza_sativa",
                                 "bio_class": "plantae",
                                 "ref_genome": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/oryza_sativa/{ref_genome_file_type}/Oryza_sativa.IRGSP-1.0.{ref_genome_file_type}.all.fa.gz"},
    "ovis_aries":               {"name": "Ovis_aries",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/ovis_aries/{ref_genome_file_type}/Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.{ref_genome_file_type}.all.fa.gz"},
    "pan_troglodytes":          {"name": "Pan_troglodytes",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/pan_troglodytes/{ref_genome_file_type}/Pan_troglodytes.Pan_tro_3.0.{ref_genome_file_type}.all.fa.gz"},
    "schistosoma_mansoni":      {"name": "Schistosoma_mansoni",
                                 "bio_class": "trematoda",
                                 "ref_genome": f"http://ftp.ensemblgenomes.org/pub/metazoa/release-60/fasta/schistosoma_mansoni/{ref_genome_file_type}/Schistosoma_mansoni.Smansoni_v7.{ref_genome_file_type}.all.fa.gz"},
    "solanum_lycopersicum":     {"name": "Solanum_lycopersicum",
                                 "bio_class": "plantae",
                                 "ref_genome": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/solanum_lycopersicum/{ref_genome_file_type}/Solanum_lycopersicum.SL3.0.{ref_genome_file_type}.all.fa.gz"},
    "sus_scrofa":               {"name": "Sus_scrofa",
                                 "bio_class": "mammal",
                                 "ref_genome": f"https://ftp.ensembl.org/pub/release-113/fasta/sus_scrofa/{ref_genome_file_type}/Sus_scrofa.Sscrofa11.1.{ref_genome_file_type}.all.fa.gz"},
    "zea_mays":                 {"name": "Zea_mays",
                                 "bio_class": "plantae",
                                 "ref_genome": f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/zea_mays/{ref_genome_file_type}/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.{ref_genome_file_type}.all.fa.gz"},
}

exclude_kingdom = ["plantae"]
exclude_species = []
# exclude_species = ["Homo_sapiens", 'Bos_taurus']

summary_file = '/large_storage/ctc/ML/data/cell/embs/scBasecamp/scBasecamp_all.csv',
data_file_loc = '/scratch/ctc/ML/uce/scBasecamp'
embedding_file = '/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
geneome_loc = '/large_storage/ctc/projects/vci/ref_genome'
gene_seq_mapping_loc = '/large_storage/ctc/projects/vci/genes/'
emb_idx_file = '/scratch/ctc/ML/uce/model_files/gene_embidx_mapping.torch'


def download_ref_genome():
    if not os.path.exists(geneome_loc):
        os.makedirs(geneome_loc, exist_ok=True)

    for dataset, metadata in scBasecamp_dir_species.items():
        url = metadata['ref_genome']
        download_file = os.path.join(geneome_loc, Path(url).name)
        if not os.path.exists(download_file):
            logging.info(f"Downloading {url} to {download_file}...")
            urllib.request.urlretrieve(url, download_file)


def inferESM2(ref_genome=None,
              geneome_loc=geneome_loc,
              gene_seq_mapping_loc=gene_seq_mapping_loc):
    if ref_genome is None:
        ref_genomes = [f.name for f in Path(geneome_loc).iterdir() if f.is_file()]
    else:
        ref_genomes = [ref_genome]

    for genome in ref_genomes:
        logging.info(f'Generating ESM2 embedding for {genome}')
        emb_generator = ESMEmbedding(genome, geneome_loc=geneome_loc)
        emb_generator.generate_gene_emb_mapping(gene_seq_mapping_loc)


def inferEvo2(ref_genome=None):
    if ref_genome is None:
        ref_genomes = [f.name for f in Path(geneome_loc).iterdir() if f.is_file()]
    else:
        ref_genomes = [ref_genome]

    for genome in ref_genomes:
        logging.info(f'Generating Evo2 embedding for {genome}')
        emb_generator = Evo2Embedding(genome, geneome_loc=geneome_loc)
        emb_generator.generate_gene_emb_mapping(gene_seq_mapping_loc)


# TODO: This is meant to fix a bug without having to reprocess the entire dataset. Remove it after the bug is fixed
def fix_numpy_to_tensor_issue(
        gene_emb_mapping_file='/large_storage/ctc/ML/data/cell/embs/scBasecamp/scBasecamp.gene_symbol_to_embedding_ESM2.pt',
        fixed_gene_emb_mapping_file='/large_storage/ctc/ML/data/cell/embs/scBasecamp/scBasecamp.gene_symbol_to_embedding_ESM2_fixed.pt'):
    gene_emb_mapping = torch.load(gene_emb_mapping_file)
    for k, v in gene_emb_mapping.items():
        if isinstance(v, torch.Tensor):
            continue
        gene_emb_mapping[k] = torch.tensor(v, dtype=torch.float64)
    torch.save(gene_emb_mapping, fixed_gene_emb_mapping_file)


def preprocess_scbasecamp(data_path='/large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS',
                          dest_path=data_file_loc,
                          summary_file = summary_file,
                          species_dirs=None):
    if species_dirs is None:
        species_dirs = [item.name for item in Path(data_path).iterdir() if item.is_dir()]
        assert len(species_dirs) > 0, 'No species specific data found'
    else:
        species_dirs = [species_dirs]

    for species in species_dirs:
        mdata = scBasecamp_dir_species.get(species.lower())
        if mdata["bio_class"] in exclude_kingdom:
            logging.warning(f"Skipping {species} as it belongs to {mdata['bio_class']}")
            continue
        if species in exclude_species:
            logging.warning(f"Skipping {species} as it is in exclude list")
            continue
        log.info(f'Species {species}...')
        species_dir = Path(data_path) / species

        os.makedirs(dest_path, exist_ok=True)

        preprocess = Preprocessor(species,
                                  species_dir,
                                  dest_path,
                                  summary_file,
                                  embedding_file,
                                  emb_idx_file)
        preprocess.process()


def _dataset_gene_iter(emb_type='ESM2',
                       species=None,
                       h5ad_file=None,
                       data_path=None,
                       feature_field=None,
                       output_dir=None):
    h5ad_file = Path(os.path.join(data_path, species, h5ad_file))
    with h5.File(h5ad_file, mode='r') as h5f:
        gene_symbols = h5f[f'var/{feature_field}/categories'][:]
        total_genes = len(gene_symbols)
        chunk_size = 3

        output_file = os.path.join(output_dir, f'{emb_type}_emb_{species.lower()}.torch')
        gene_emb_mapping = {}
        if os.path.exists(output_file):
            gene_emb_mapping = torch.load(output_file)

        for i in range(0, total_genes, chunk_size):
            gene_symbols_chunk = gene_symbols[i:i+chunk_size]
            gene_symbols_chunk = [gene_symbol.decode('utf-8') for gene_symbol in gene_symbols_chunk]

            filtered_gene_symbols = []
            for gene_symbol in gene_symbols_chunk:
                if gene_symbol in gene_emb_mapping:
                    logging.info(f"Skipping {gene_symbol}...")
                    continue
                filtered_gene_symbols.append(gene_symbol)
            if len(filtered_gene_symbols) == 0:
                continue

            logging.info(f"Processing {i} of {total_genes}: {filtered_gene_symbols}...")
            sequences = resolve_genes(filtered_gene_symbols, return_type = 'dna')
            if sequences is None:
                logging.warning(f"Could not resolve {filtered_gene_symbols}")
                continue

            for gene_symbol, sequence in sequences:
                if sequence is None:
                    logging.warning(f"Could not resolve {gene_symbol}")
                    continue
                with open(f'/large_storage/ctc/projects/vci/ncbi/{species}.cvs', 'a') as file:
                    file.write(f'{gene_symbol},{str(sequence)}\n')

                gene_emb_mapping[gene_symbol] = sequence
                if emb_type == 'ESM2':
                    sequence = sequence.translate()

                sequence = str(sequence)
                if emb_type == 'ESM2':
                    if len(sequence) > 16559:
                        sequence = sequence[:16559]

                yield species, gene_symbol, [str(sequence)]


def resolve_gene_symbols(
        feature_field='gene_symbols',
        species_dir=None,
        data_path=data_file_loc,
        gene_seq_mapping_loc=gene_seq_mapping_loc,
        emb_type='ESM2'):

    if species_dir is None:
        species_dirs = [item.name for item in Path(data_path).iterdir() if item.is_dir()]
        assert len(species_dirs) > 0, 'No species specific data found'
    else:
        species_dirs = [species_dir]

    for species in species_dirs:
        h5ad_files = [f.name for f in Path(os.path.join(data_path, species)).iterdir() if f.is_file()]
        for h5ad_file in h5ad_files:
            _dataset_gene_iter_fn = partial(_dataset_gene_iter,
                                            emb_type='ESM2',
                                            species=species,
                                            h5ad_file=h5ad_file,
                                            data_path=data_path,
                                            feature_field=feature_field,
                                            output_dir=gene_seq_mapping_loc)
            logging.info(f'Generating {emb_type} embedding for {h5ad_file}')
            if emb_type == 'ESM2':
                emb_generator = ESMEmbedding(None,
                                             geneome_loc=geneome_loc,
                                             seq_generator_fn=_dataset_gene_iter_fn)
            else:
                emb_generator = Evo2Embedding(None,
                                            geneome_loc=geneome_loc,
                                            seq_generator_fn=_dataset_gene_iter_fn)
            emb_generator.species = species
            emb_generator.generate_gene_emb_mapping(gene_seq_mapping_loc)


if __name__ == '__main__':
    fire.Fire()
