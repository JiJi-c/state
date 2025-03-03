import gzip
import logging
from Bio import SeqIO


def parse_genename_seq(fasta_file):
    gene_dict = {}
    without_chromosome = 0
    with gzip.open(fasta_file, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            header_parts = record.description.split()
            gene_name = None
            chromosome = None
            for part in header_parts:
                if part.startswith('gene_symbol:'):
                    gene_name = part.split(':')[1]
                elif part.startswith('chromosome:'):
                    chromosome = part.replace('chromosome:', '')
                elif part.startswith('chr:'):
                    chromosome = part.replace('chr:', '')
                if gene_name and chromosome:
                    break

            if gene_name is not None:
                if chromosome is None:
                    without_chromosome += 1
                gene_dict[gene_name] = chromosome, str(record.translate().seq)

    return gene_dict, without_chromosome


def create_genename_sequence_map(fasta_file, output_file=None):
    gene_dict, without_chromosome = parse_genename_seq(fasta_file)
    logging.info(f'{len(gene_dict)} genes found in {fasta_file}. {without_chromosome} genes without chromosome information.')
    if output_file is not None:
        with open(output_file, 'a') as f:
            for gene, info in gene_dict.items():
                f.write(f'{gene}\t{info[0]}\t{info[1]}\n')
    return gene_dict