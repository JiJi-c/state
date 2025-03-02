import os
import fire
import logging
import h5py as h5

from pathlib import Path
import urllib.request

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

scBasecamp_dir_species = {
    "arabidopsis_thaliana":     {"name": "Arabidopsis_thaliana",
                                 "bio_class": "plantae",
                                 "ref_genome": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/cdna/Arabidopsis_thaliana.TAIR10.cdna.all.fa.gz"},
    "bos_taurus":               {"name": "Bos_taurus",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/bos_taurus/cdna/Bos_taurus.ARS-UCD1.3.cdna.all.fa.gz"},
    "caenorhabditis_elegans":   {"name": "Caenorhabditis_elegans",
                                 "bio_class": "invertebrate",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/caenorhabditis_elegans/cdna/Caenorhabditis_elegans.WBcel235.cdna.all.fa.gz"},
    "callithrix_jacchus":       {"name": "Callithrix_jacchus",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/callithrix_jacchus/cdna/Callithrix_jacchus.mCalJac1.pat.X.cdna.all.fa.gz"},
    "danio_rerio":              {"name": "Danio_rerio",
                                 "bio_class": "fish",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/danio_rerio/cdna/Danio_rerio.GRCz11.cdna.all.fa.gz"},
    "drosophila_melanogaster":  {"name": "Drosophila_melanogaster",
                                 "bio_class": "insecta",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/drosophila_melanogaster/cdna/Drosophila_melanogaster.BDGP6.46.cdna.all.fa.gz"},
    "equus_caballus":           {"name": "Equus_caballus",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/equus_caballus/cdna/Equus_caballus.EquCab3.0.cdna.all.fa.gz"},
    "gallus_gallus":            {"name": "Gallus_gallus",
                                 "bio_class": "aves",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/gallus_gallus/cdna/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.cdna.all.fa.gz"},
    "gorilla_gorilla":          {"name": "Gorilla_gorilla",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/gorilla_gorilla/cdna/Gorilla_gorilla.gorGor4.cdna.all.fa.gz"},
    "heterocephalus_glaber":    {"name": "Heterocephalus_glaber",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/heterocephalus_glaber_female/cdna/Heterocephalus_glaber_female.Naked_mole-rat_maternal.cdna.all.fa.gz"},
    "homo_sapiens":             {"name": "Homo_sapiens",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz"},
    "macaca_mulatta":           {"name": "Macaca_mulatta",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/macaca_mulatta/cdna/Macaca_mulatta.Mmul_10.cdna.all.fa.gz"},
    "mus_musculus":             {"name": "Mus_musculus",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz"},
    "oryctolagus_cuniculus":    {"name": "Oryctolagus_cuniculus",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/oryctolagus_cuniculus/cdna/Oryctolagus_cuniculus.OryCun2.0.cdna.all.fa.gz"},
    "oryza_sativa":             {"name": "Oryza_sativa",
                                 "bio_class": "plantae",
                                 "ref_genome": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/oryza_sativa/cdna/Oryza_sativa.IRGSP-1.0.cdna.all.fa.gz"},
    "ovis_aries":               {"name": "Ovis_aries",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/ovis_aries/cdna/Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.cdna.all.fa.gz"},
    "pan_troglodytes":          {"name": "Pan_troglodytes",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/pan_troglodytes/cdna/Pan_troglodytes.Pan_tro_3.0.cdna.all.fa.gz"},
    "schistosoma_mansoni":      {"name": "Schistosoma_mansoni",
                                 "bio_class": "trematoda",
                                 "ref_genome": "http://ftp.ensemblgenomes.org/pub/metazoa/release-60/fasta/schistosoma_mansoni/cdna/Schistosoma_mansoni.Smansoni_v7.cdna.all.fa.gz"},
    "solanum_lycopersicum":     {"name": "Solanum_lycopersicum",
                                 "bio_class": "plantae",
                                 "ref_genome": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/solanum_lycopersicum/cdna/Solanum_lycopersicum.SL3.0.cdna.all.fa.gz"},
    "sus_scrofa":               {"name": "Sus_scrofa",
                                 "bio_class": "mammal",
                                 "ref_genome": "https://ftp.ensembl.org/pub/release-113/fasta/sus_scrofa/cdna/Sus_scrofa.Sscrofa11.1.cdna.all.fa.gz"},
    "zea_mays":                 {"name": "Zea_mays",
                                 "bio_class": "plantae",
                                 "ref_genome": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/zea_mays/cdna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.cdna.all.fa.gz"},
}

exclude_kingdom = ["plantae"]

# summary_file = '/scratch/ctc/ML/vci/scBasecamp_all.csv'
summary_file = '/home/rajesh.ilango/scBasecamp_all.csv'
embedding_file = '/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
geneome_loc = '/large_storage/ctc/projects/vci/ref_genome'
gene_emb_mapping_file = '/large_storage/ctc/ML/data/cell/emb/ESM/gene_chrom_sequence_mapping.tsv'
emb_idx_file = '/scratch/ctc/ML/uce/model_files/gene_embidx_mapping.torch'


def download_ref_genome():
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

if __name__ == '__main__':
    fire.Fire()