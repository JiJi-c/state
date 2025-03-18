import gzip
import logging
import requests

from functools import partial
from typing import List

from Bio import SeqIO, Entrez
from Bio.Seq import Seq


def resolve_genes(ensemble_id,
                  return_type = 'dna',
                  email="rajesh.ilango@arcinstitute.org"):
    Entrez.email = email
    Entrez.api_key = "6015ba109c4bd986f4a2947a90462ba53708"
    Entrez.tool = "vci_gene_symbol_resolver"

    logging.info(f"Processing {ensemble_id}...")
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{ensemble_id}"

    r = requests.get(server + ext, headers={"Content-Type": "text/plain"})
    if not r.ok:
        return None, None

    if return_type == 'dna':
        return ensemble_id, r.text, None
    else:
        return ensemble_id, str(Seq(r.text).translate()), None


def parse_genename_seq(fasta_file, return_type='dna'):
    gene_dict = {} # gene_name -> (chromosome, sequence)
    gene_name_map = {} # gene_name -> gene_id
    with gzip.open(fasta_file, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if return_type == 'dna':
                seq = str(record.seq)
            else:
                seq = str(record.translate().seq)
            if len(seq) == 0:
                continue

            header_parts = record.description.split()
            chromosome = None
            gene = None
            for part in header_parts:
                if part.startswith('gene:'):
                    gene = part.split(':')[1]
                elif part.startswith('chromosome:'):
                    chromosome = part.replace('chromosome:', '')
                elif part.startswith('chr:'):
                    chromosome = part.replace('chr:', '')
                if gene and chromosome:
                    break

            if gene:
                if gene in gene_dict:
                    chroms, prot_seqs = gene_dict[gene]
                else:
                    chroms, prot_seqs = [], []

                chroms.append(chromosome)
                prot_seqs.append(seq)
                gene_dict[gene] = chroms, prot_seqs

    return gene_dict, gene_name_map


def parse_genome_for_gene_seq_map(species,
                                  fasta_file,
                                  output_file=None,
                                  return_type='dna'):
    gene_dict, gene_name_map = parse_genename_seq(fasta_file, return_type=return_type)
    logging.info(f'{len(gene_dict)} genes in {fasta_file}')
    if output_file is not None:
        with open(output_file, 'a') as f:
            for gene, info in gene_dict.items():
                f.write(f'{species}\t{gene}\t{info[0]}\t{info[1]}\n')
    return gene_dict, gene_name_map