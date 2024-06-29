import h5py
import numpy as np
import os
from tqdm import trange
import multiprocessing as mp
from utils import create_mask

def process_raw_cell(q, start, end, origin_sparse, size):
    m = np.zeros((size, size))
    for cell_id in range(start, end):
        cell_coo_origin = origin_sparse[cell_id].tocoo()
        xs, ys, proba = cell_coo_origin.row, cell_coo_origin.col, cell_coo_origin.data
        proba = process_proba(proba)
        np.add.at(m, (xs, ys), proba)
        np.add.at(m, (ys, xs), proba)
    q.put(m)

def process_imputed_cell(q, start, end, hdf, xs, ys, size):
    m = np.zeros((size, size))
    for cell_id in range(start, end):
        proba = np.array(hdf[f"cell_{cell_id}"])
        proba = process_proba(proba)
        np.add.at(m, (xs, ys), proba)
        np.add.at(m, (ys, xs), proba)
    q.put(m)

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
        cells_per_process = (len(self.cell_ids) + self.process_cnt - 1) // self.process_cnt
        for idx in range_iter:
            chrom = self.chrom_list[idx].decode('utf-8')
            file_path = os.path.join(self.raw_dir, f"chr{chrom}_sparse_adj.npy")
            origin_sparse = np.load(file_path, allow_pickle=True)
            size = origin_sparse[0].shape[0]
            mask = 1 - create_mask(self.res, self.cytoband_path, -1, chrom, origin_sparse)

            m = np.zeros((size, size))
            q = mp.Queue()
            processes = []

            cells_per_process = len(self.cell_ids) // self.process_cnt

            if self.is_raw:
                for process_id in range(self.process_cnt):
                    start = process_id * cells_per_process
                    end = min((process_id + 1) * cells_per_process, len(self.cell_ids))
                    if start < end:
                        p = mp.Process(target=process_raw_cell, args=(q, start, end, origin_sparse, size))
                        processes.append(p)
                        p.start()
            else:
                with h5py.File(os.path.join(self.impute_dir, f"chr{chrom}_{self.embedding_name}_nbr_{self.neighbors}_impute.hdf5"), "r") as hdf:
                    coordinates = hdf['coordinates']
                    xs, ys = coordinates[:, 0], coordinates[:, 1]
                    for process_id in range(self.process_cnt):
                        start = process_id * cells_per_process
                        end = min((process_id + 1) * cells_per_process, len(self.cell_ids))
                        if start < end:
                            p = mp.Process(target=process_imputed_cell, args=(q, start, end, hdf, xs, ys, size))
                            processes.append(p)
                            p.start()

            for _ in processes:
                cell_m = q.get()
                m += cell_m

            for p in processes:
                p.join()

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
