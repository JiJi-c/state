import os
import fire
import logging
import shutil

import torch
import pandas as pd
import urllib.request

from pathlib import Path
from ast import literal_eval
from transformers import AutoTokenizer, AutoModel

from vci.data.preprocess import Preprocessor
from vci.data.gene_emb import create_genename_sequence_map


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

ref_genome_file_type = 'cds' # 'cdna'

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

# summary_file = '/scratch/ctc/ML/vci/scBasecamp_all.csv'
summary_file = '/home/rajesh.ilango/scBasecamp_all.csv'
embedding_file = '/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
geneome_loc = '/large_storage/ctc/projects/vci/ref_genome'
gene_emb_mapping_file = '/large_storage/ctc/ML/data/cell/emb/ESM/gene_chrom_sequence_mapping.tsv'
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


def generate_gene_sequence_mapping():
    ref_genomes = [f.name for f in Path(geneome_loc).iterdir() if f.is_file()]
    for ref_genome in ref_genomes:
        ref_genome = Path(os.path.join(geneome_loc, ref_genome))
        create_genename_sequence_map(ref_genome, output_file=gene_emb_mapping_file)


def preprocess_scbasecamp(data_path='/large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS',
                          dest_path='/scratch/ctc/ML/uce/scBasecamp',
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


def inferESM2(gene_emb_mapping_file='/large_storage/ctc/ML/data/cell/emb/ESM/gene_chrom_sequence_mapping.tsv',
          output_file="/large_storage/ctc/ML/data/cell/emb/ESM/scBasecamp.gene_symbol_to_embedding_ESM2.pt"):

    batch_size=1 # PLEASE DO NOT CHANGE THIS VALUE. After changes for addressing splices, this is the only value that works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the ESM2 model from Hugging Face
    model_name = "facebook/esm2_t33_650M_UR50D"  # You can also use other ESM2 variants like "facebook/esm2_t12_35M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    seq_df = pd.read_csv(gene_emb_mapping_file, sep='\t', header=None)
    seq_df.columns = ['gene', 'chrom', 'sequence']

    gene_emb_map = {}
    if os.path.exists(output_file):
        gene_emb_map= torch.load(output_file)

    for i in range(0, len(seq_df), batch_size):
        batch = seq_df.iloc[i:i+batch_size]

        sequences = literal_eval(batch['sequence'].iloc[0])
        seq_len = sum([len(s) for s in sequences])
        gene = batch['gene'].iloc[0]
        if gene in gene_emb_map:
            logging.info(f"Skipping {gene} {seq_len}")
            continue
        else:
            logging.info(f"Processing {gene} {seq_len}...")

        if seq_len >= 13084:
            logging.info(f"Too large sequence {gene} {seq_len}")
            continue

        # Tokenize the sequence
        inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the embeddings (representations)
        # There are different ways to get embeddings from ESM2:
        # Using the last hidden state (token embeddings)
        gene_emb_map[gene] = outputs.last_hidden_state.mean(1).mean(0).cpu().numpy()

        if i % (10 * batch_size) == 0:
            logging.info(f'Saving after {i//batch_size} batches...')
            torch.save(gene_emb_map, output_file)

        if i % (1000 * batch_size) == 0:
            logging.info(f'creating checkpoint {i//batch_size}...')
            checkpoint_file = output_file.replace('.pt', f'.{i}.pt')
            shutil.copyfile(output_file, checkpoint_file)

        del outputs

        # For per-residue embeddings (excluding special tokens)
        # The first token is <cls> and the last is <eos>
        # per_residue_embeddings = token_embeddings[0, 1:-1, :]
        torch.cuda.empty_cache()
    torch.save(gene_emb_map, output_file)

if __name__ == '__main__':
    fire.Fire()
