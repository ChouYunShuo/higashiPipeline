import h5py
import numpy as np
import os
from tqdm import trange
from multiprocessing import Pool
from utils import create_mask

def process_raw_cell(args):
    origin_sparse, cell_id, size = args
    cell_coo_origin = origin_sparse[cell_id].tocoo()
    xs, ys, proba = cell_coo_origin.row, cell_coo_origin.col, cell_coo_origin.data
    proba = process_proba(proba)
    m = np.zeros((size, size))
    np.add.at(m, (xs, ys), proba)
    np.add.at(m, (ys, xs), proba)
    return m

def process_imputed_cell(args):
    hdf, cell_id, xs, ys, size = args
    proba = np.array(hdf[f"cell_{cell_id}"])
    proba = process_proba(proba)
    m = np.zeros((size, size))
    np.add.at(m, (xs, ys), proba)
    np.add.at(m, (ys, xs), proba)
    return m

def process_proba(proba):
    proba /= np.sum(proba)
    proba[proba <= 1e-6] = 0.0
    proba *= 1e6
    return proba

class GroupMatrixParser:
    def __init__(self, impute_dir, raw_dir, chrom_list, chrom_offset, cell_ids, neighbors, res, cytoband_path, embedding_name="exp1", process_cnt=1, is_raw=False):
        self.impute_dir = impute_dir
        self.raw_dir = raw_dir
        self.chrom_list = chrom_list
        self.chrom_offset = chrom_offset
        self.cell_ids = cell_ids
        self.neighbors = neighbors
        self.res = res
        self.cytoband_path = cytoband_path
        self.embedding_name = embedding_name
        self.process_cnt = process_cnt
        self.is_raw = is_raw

    def __iter__(self):
        range_iter = trange(len(self.chrom_list), desc="Processing Chromosomes")

        for idx in range_iter:
            chrom = self.chrom_list[idx].decode('utf-8')
            file_path = os.path.join(self.raw_dir, f"chr{chrom}_sparse_adj.npy")
            origin_sparse = np.load(file_path, allow_pickle=True)
            size = origin_sparse[0].shape[0]
            mask = 1 - create_mask(self.res, self.cytoband_path, -1, chrom, origin_sparse)

            m = np.zeros((size, size))

            if self.is_raw:
                cell_args = [(origin_sparse, cell_id, size) for cell_id in self.cell_ids]
                with Pool(self.process_cnt) as pool:
                    for cell_m in pool.imap(process_raw_cell, cell_args):
                        m += cell_m
            else:
                with h5py.File(os.path.join(self.impute_dir, f"chr{chrom}_{self.embedding_name}_nbr_{self.neighbors}_impute.hdf5"), "r") as hdf:
                    coordinates = hdf['coordinates']
                    xs, ys = coordinates[:, 0], coordinates[:, 1]
                    cell_args = [(hdf, cell_id, xs, ys, size) for cell_id in self.cell_ids]
                    with Pool(self.process_cnt) as pool:
                        for cell_m in pool.imap(process_imputed_cell, cell_args):
                            m += cell_m

            m *= mask

            xs, ys = np.nonzero(m)
            proba = m[xs, ys]
            proba1 = m[ys, xs]

            x = np.concatenate([xs, ys]) + self.chrom_offset[idx]
            y = np.concatenate([ys, xs]) + self.chrom_offset[idx]
            proba = np.concatenate([proba, proba1])

            sorter = np.lexsort((y, x))
            x = x[sorter]
            y = y[sorter]
            proba = proba[sorter]

            yield {
                "bin1_id": x,
                "bin2_id": y,
                "count": proba,
            }
