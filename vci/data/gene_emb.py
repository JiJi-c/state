import gzip
import logging
from Bio import SeqIO


def parse_genename_seq(fasta_file):
    gene_dict = {}
    with gzip.open(fasta_file, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            header_parts = record.description.split()
            gene_name = None
            chromosome = None
            i = 0
            for part in header_parts:
                if part.startswith('gene_symbol:'):
                    gene_name = part.split(':')[1]
                    i += 1
                elif part.startswith('chromosome:') or part.startswith('chr:'):
                    chromosome = part.split(':')[1]
                    i += 1
                if i == 2:
                    break
            gene_dict[gene_name] = chromosome, str(record.seq)
    return gene_dict


def create_genename_sequence_map(fasta_file, output_file=None):
    gene_dict = parse_genename_seq(fasta_file)
    logging.info(f'{len(gene_dict)} genes found in {fasta_file}')
    if output_file is not None:
        with open(output_file, 'a') as f:
            for gene, info in gene_dict.items():
                f.write(f'{gene}\t{info[0]}\t{info[1]}\n')
    return gene_dict