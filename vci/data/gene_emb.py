import gzip
import logging
import requests
import mygene
import numpy as np

from Bio import SeqIO, Entrez
from Bio.Seq import Seq
import time


GENE_NAME_ENSEMPLE_MAP = {
    'GATD3A': 'ENSMUSG00000053329',
    'GATD3B': 'ENSG00000160221',
    'PBK': 'ENSG00000168078',
    'NEPRO': 'ENSG00000163608'
    'HSPA14-1': 'ENSG00000187522',
}

def convert_gene_symbols_to_ensembl_rest(gene_symbols, species="human"):

    server = "https://grch37.rest.ensembl.org"

    # Map species to its scientific name
    species_map = {
        "human": "homo_sapiens",
        "mouse": "mus_musculus",
        "rat": "rattus_norvegicus"
    }

    species_name = species_map.get(species.lower(), species)

    gene_to_ensembl = {}

    for symbol in gene_symbols:
        # Construct the URL for the API request
        ext = f"/lookup/symbol/{species_name}/{symbol}?"

        # Make the request
        r = requests.get(server + ext, headers={"Content-Type": "application/json"})

        # Check if the request was successful
        if r.status_code != 200:
            logging.warning(f"[REST API] Failed to retrieve data for {symbol}: {r.status_code}")
            gene_to_ensembl[symbol] = None  
            continue


        # Parse the JSON response
        decoded = r.json()

        # Extract the Ensembl ID
        if "id" in decoded:
            gene_to_ensembl[symbol] = decoded["id"]

        # Sleep briefly to avoid overloading the server
        time.sleep(0.1)

    return gene_to_ensembl


def convert_symbols_to_ensembl(adata):
    gene_symbols = adata.var_names.tolist()

    mg = mygene.MyGeneInfo()
    results = mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene', species='human')

    symbol_to_ensembl = {}
    for result in results:
        if 'ensembl' in result and not result.get('notfound', False):
            if isinstance(result['ensembl'], list):
                symbol_to_ensembl[result['query']] = result['ensembl'][0]['gene']
            else:
                symbol_to_ensembl[result['query']] = result['ensembl']['gene']

    for symbol in gene_symbols:
        if symbol_to_ensembl.get(symbol) is None:
            sym_results = convert_gene_symbols_to_ensembl_rest([symbol])
            if symbol in sym_results and sym_results[symbol] is not None:
                symbol_to_ensembl[symbol] = sym_results[symbol]
                logging.info(f"Converted {symbol} to {symbol_to_ensembl[symbol]} using REST API")
            else:
                logging.warning(f"Could not retrieve Ensembl ID for {symbol} using REST — fallback will be used")


    logging.info(f"Done...")
    # for symbol in gene_symbols:
    #     if symbol_to_ensembl.get(symbol) is None:
    #         logging.info(f"{symbol} -> {symbol_to_ensembl.get(symbol, np.nan)}")
    #         symbol_to_ensembl[symbol] = GENE_NAME_ENSEMPLE_MAP[symbol]

    for symbol in gene_symbols:
        value = symbol_to_ensembl.get(symbol, None)
        if value is None or str(value).lower() == 'nan' or value == '':
            logging.info(f"{symbol} -> {value}")
            if symbol in GENE_NAME_ENSEMPLE_MAP:
                symbol_to_ensembl[symbol] = GENE_NAME_ENSEMPLE_MAP[symbol]
                logging.info(f"Manually assigned {symbol} to {GENE_NAME_ENSEMPLE_MAP[symbol]}")
            else:
                logging.warning(f"{symbol} not found in GENE_NAME_ENSEMPLE_MAP — assigned np.nan")
                symbol_to_ensembl[symbol] = np.nan


    # Add the remaining or errored ones manually
    symbol_to_ensembl['PBK'] = 'ENSG00000168078'
    adata.var['gene_symbols'] = [symbol_to_ensembl.get(symbol, np.nan) for symbol in gene_symbols]
    return adata


def resolve_genes(ensemble_id,
                  return_type = 'dna',
                  email="rajesh.ilango@arcinstitute.org"):
    Entrez.email = email
    Entrez.api_key = "6015ba109c4bd986f4a2947a90462ba53708"
    Entrez.tool = "vci_gene_symbol_resolver"

    logging.info(f"Processing {ensemble_id}...")
    server = "https://grch37.rest.ensembl.org"
    ext = f"/sequence/id/{ensemble_id}"

    r = requests.get(server + ext, headers={"Content-Type": "text/plain"})
    if not r.ok:
        logging.info(f"Response: {r.status_code} {r.text}")
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
