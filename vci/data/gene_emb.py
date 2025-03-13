import gzip
import logging
import asyncio

from functools import partial
from typing import List

from Bio import SeqIO, Entrez


def sequence_from_gene_symbol(gene_name:str,
                              return_type = 'dna',
                              email="rajesh.ilango@arcinstitute.org"):
    Entrez.email = email
    Entrez.api_key = "6015ba109c4bd986f4a2947a90462ba53708"
    Entrez.tool = "vci_gene_symbol_resolver"

    logging.info(f"Processing {gene_name}...")
    # Find the gene ID
    try:
        seq = None
        if gene_name.startswith('ENSG') or '.' in gene_name:
            handle = Entrez.efetch(db="nucleotide", id=gene_name, rettype="fasta", retmode="text")
            for record in SeqIO.parse(handle, 'fasta'):
                seq = record
                break
        else:
            gene_search = Entrez.esearch(db="gene", term=f"{gene_name}[GENE]")
            gene_record = Entrez.read(gene_search)
            if len(gene_record["IdList"]) == 0:
                return None, None
            gene_id = gene_record["IdList"][0]

            # Link to nucleotide database
            link_results = Entrez.read(Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id))
            nuccore_ids = [link["LinkSetDb"][0]["Link"][0]["Id"] for link in link_results if link["LinkSetDb"]]

            # Fetch the sequence
            if nuccore_ids:
                handle = Entrez.efetch(db="nuccore", id=nuccore_ids[0], rettype="fasta", retmode="text")
                for record in SeqIO.parse(handle, 'fasta'):
                    seq = record
        if seq is None:
            return None, None
        elif return_type == 'dna':
            return gene_name, seq.seq
        else:
            return gene_name, seq.translate().seq
    except Exception as e:
        logging.error(f"Error processing {gene_name}: {e}")
        return None, None


def resolve_genes(gene_symbols:List[str],
                  return_type = 'dna',
                  email="rajesh.ilango@arcinstitute.org"):
    resolve_gene_fn = partial(sequence_from_gene_symbol, return_type=return_type, email=email)
    tasks = []
    for gene_symbol in gene_symbols:
        tasks.append(resolve_gene_fn(gene_symbol) )
    return tasks


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
            gene_name = None
            chromosome = None
            gene = None
            for part in header_parts:
                if part.startswith('gene:'):
                    gene = part.split(':')[1]
                if part.startswith('gene_symbol:'):
                    gene_name = part.split(':')[1]
                elif part.startswith('chromosome:'):
                    chromosome = part.replace('chromosome:', '')
                elif part.startswith('chr:'):
                    chromosome = part.replace('chr:', '')
                if gene_name and chromosome:
                    break

            if gene_name is not None and gene is not None:
                gene_name_map[gene_name] = gene

            if gene_name is None and gene is not None:
                gene_name = gene

            if gene_name:
                if gene_name in gene_dict:
                    chroms, prot_seqs = gene_dict[gene_name]
                else:
                    chroms, prot_seqs = [], []

                chroms.append(chromosome)
                prot_seqs.append(seq)
                gene_dict[gene_name] = chroms, prot_seqs

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