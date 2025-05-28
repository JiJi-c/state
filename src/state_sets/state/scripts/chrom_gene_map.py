import pickle
import pandas as pd
import fire


def generate_chrom_gene_map(ds_chrom_gene_map_file='/scratch/ctc/ML/uce/dataset_to_chroms_new.pkl',
                            chrom_gene_map_file='/scratch/ctc/ML/uce/gene_to_chrom_map.pkl'):

    ds_chrom_gene_map = pickle.load(open(ds_chrom_gene_map_file, 'rb'))
    gene_chrom_map = {}
    invalid_mappings = 0
    for ds, chrom_gene_map in ds_chrom_gene_map.items():
        # print(f'Processing {ds}')
        for genes, chrom in chrom_gene_map.items():
            if genes in gene_chrom_map:
                if gene_chrom_map[genes] != chrom:
                    invalid_mappings += 1
                    print(f'Gene {genes} mapped to {chrom} is appearing in chrom {gene_chrom_map[genes]} ds {ds}')
            else:
                gene_chrom_map[genes] = chrom

    print(f'Invalid mappings: {invalid_mappings} out of {len(gene_chrom_map)}')
    pickle.dump(gene_chrom_map, open(chrom_gene_map_file, 'wb'))


if __name__ == '__main__':
    fire.Fire(generate_chrom_gene_map)