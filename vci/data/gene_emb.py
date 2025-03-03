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
            for part in header_parts:
                if part.startswith('gene_symbol:'):
                    gene_name = part.split(':')[1]
                elif part.startswith('chromosome:'):
                    chromosome = part.replace('chromosome:', '')
                elif part.startswith('chr:'):
                    chromosome = part.replace('chr:', '')
                if gene_name and chromosome:
                    break

            if gene_name:
                if gene_name in gene_dict:
                    chroms, prot_seqs = gene_dict[gene_name]
                else:
                    chroms, prot_seqs = [], []

                chroms.append(chromosome)
                prot_seqs.append(str(record.translate().seq))
                gene_dict[gene_name] = chroms, prot_seqs

    return gene_dict


def create_genename_sequence_map(fasta_file, output_file=None):
    gene_dict = parse_genename_seq(fasta_file)
    logging.info(f'{len(gene_dict)} genes in {fasta_file}')
    if output_file is not None:
        with open(output_file, 'a') as f:
            for gene, info in gene_dict.items():
                f.write(f'{gene}\t{info[0]}\t{info[1]}\n')
    return gene_dict