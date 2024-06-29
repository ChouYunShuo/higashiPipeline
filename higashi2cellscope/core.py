import h5py
import numpy as np
import json
import os
import sys
import pickle
from sklearn.decomposition import PCA
from umap import UMAP
from collections import defaultdict
from utils import rlencode, fileType, copyDataset, merge_temp_h5_files, print_hdf5_structure, check_hdf5_structure, get_celltype_dict
from matrixParser import MatrixParser
from groupMatrixParser import GroupMatrixParser
import multiprocessing as mp

CHROM_DTYPE = np.dtype("S")
CHROMID_DTYPE = np.int32
CHROMSIZE_DTYPE = np.int32
COORD_DTYPE = np.int32
BIN_DTYPE = np.int64
COUNT_DTYPE = np.float32
OFFSET_DTYPE = np.int64

"""
├── meta
│    ├── label
├── embed
│    ├── pca
│    └── umap
└── resolutions
    ├── 10000
    │   ├── bins
    |   |   ├── chrom
    |   |   ├── start
    |   |   └── end
    │   ├── chroms
    |   |   ├── name
    |   |   └── length
    │   └── layers
    │         ├── raw
    │         │   ├── cell_0
    │         │   │   ├── pixels
    │         │   │   │   ├── bin1_id
    │         │   │   │   ├── bin2_id
    │         │   │   │   └── count
    │         │   │   └── indexes
    │         │   │       ├── chrom_offset
    │         │   │       └── bin1_offset
    │         │   ├── cell_1
    │         │   ├── cell_2
    │         │   ├── group_0
    │         │   └── group_1
    │         ├── imputed_0neighbor
    │         │   ├── cell_0
    │         │   │   ├── pixels
    │         │   │   │   ├── bin1_id
    │         │   │   │   ├── bin2_id
    │         │   │   │   └── count
    │         │   │   └── indexes
    │         │   │       ├── chrom_offset
    │         │   │       └── bin1_offset
    │         │   ├── cell_1
    │         │   └── cell_2
    │         ├── imputed_5neighbor
    │         │   ├── cell_0
    │         │   │   ├── pixels
    │         │   │   │   ├── bin1_id
    │         │   │   │   ├── bin2_id
    │         │   │   │   └── count
    │         │   │   └── indexes
    │         │   │       ├── chrom_offset
    │         │   │       └── bin1_offset
    │         │   ├── cell_1
    │         │   └── cell_2
    │         ├── tracks
    |         │    └── insulation
    │         │       ├── cell_0
    │         │       ├── cell_1
    │         │       ├── group_0
    │         │       └── group_1
    │   
    ├── 50000

"""
def sort_key(item):
    name = item[0]
    if name.isdigit():
        return int(name)
    
def get_chroms_from_txt(filename):
    chromosome_data = []
    
    with open(filename, 'r') as file:
        for line in file:
            name, length = line.split()
            if name.startswith("chr") and (name[3:].isdigit()):#or name[3] in ["X"]):
                chromosome_data.append((name[3:], int(length)))

    chromosome_data.sort(key=sort_key)
    return chromosome_data

def write_chroms(grp, fname, h5_opts):
    chrom_dict = defaultdict(list)
    all_chrom = get_chroms_from_txt(fname)
    for chrom in all_chrom:
        chrom_dict["name"].append(chrom[0])
        chrom_dict["length"].append(chrom[1])

    n_chroms = len(chrom_dict["name"])

    names = np.array(chrom_dict["name"], dtype=CHROM_DTYPE)
    lengths = np.array(chrom_dict["length"], dtype=CHROMSIZE_DTYPE)

    grp.create_dataset('name', shape=(n_chroms,),
                       dtype=names.dtype, data=names, **h5_opts)
    grp.create_dataset("length", shape=(n_chroms,),
                       dtype=lengths.dtype, data=lengths, **h5_opts)

    return names, lengths

def write_bins(res, chroms_names, chrom_lens, grp, h5_opts):
    bin_dict = defaultdict(list)
    for index in range(len(chroms_names)):
        for i in range(int((chrom_lens[index])/(res))+1):
            bin_dict["chrom"].append(chroms_names[index])
            bin_dict["start"].append(i*res)
            bin_dict["end"].append((i+1)*res)

    n_bins = len(bin_dict["chrom"])

    chroms = np.array(bin_dict["chrom"], dtype=CHROM_DTYPE)
    starts = np.array(bin_dict["start"], dtype=COORD_DTYPE)
    ends = np.array(bin_dict["end"], dtype=COORD_DTYPE)

    grp.create_dataset('chrom', shape=(n_bins,),
                       dtype=chroms.dtype, data=chroms, **h5_opts)
    grp.create_dataset("start", shape=(n_bins,),
                       dtype=starts.dtype, data=starts, **h5_opts)
    grp.create_dataset("end", shape=(n_bins,),
                       dtype=ends.dtype, data=ends, **h5_opts)

def setup_pixels(grp, nbins, h5_opts):
    max_shape = nbins*nbins
    grp.create_dataset('bin1_id', shape=(max_shape,),
                       dtype=BIN_DTYPE, **h5_opts)
    grp.create_dataset("bin2_id", shape=(max_shape,),
                       dtype=BIN_DTYPE, **h5_opts)
    grp.create_dataset("count", shape=(max_shape,),
                       dtype=COUNT_DTYPE, **h5_opts)

def write_pixels(grp, impute_dir, raw_dir, chrom_list, chrom_offset, cell_id, neighbors, columns,res, cytoband_path, embedding_name, process_cnt, is_raw):
    cellMatrixParser = MatrixParser(
         impute_dir, raw_dir, chrom_list, chrom_offset, cell_id, neighbors, res, cytoband_path, embedding_name, process_cnt, is_raw)
    m_size = 0
    for chunk in cellMatrixParser:
        dsets = [grp[col] for col in columns]
        n = len(chunk[columns[0]])
        for col, dset in zip(columns, dsets):
            dset.resize((m_size + n,))
            dset[m_size: m_size + n] = chunk[col]
        m_size += n

def write_group_pixels(grp, impute_dir, raw_dir, chrom_list, chrom_offset, cell_ids, neighbors, columns, res, cytoband_path, embedding_name, process_cnt, is_raw):
    matrixParser = GroupMatrixParser(impute_dir, raw_dir, chrom_list, chrom_offset, cell_ids, neighbors, res, cytoband_path, embedding_name, process_cnt, is_raw)
    m_size = 0
    for chunk in matrixParser:
        dsets = [grp[col] for col in columns]
        n = len(chunk[columns[0]])
        for col, dset in zip(columns, dsets):
            dset.resize((m_size + n,))
            dset[m_size: m_size + n] = chunk[col]
        m_size += n

def get_bin_index(grp, n_chroms, n_bins):
    chrom_ids = grp["chrom"]
    chrom_offset = np.zeros(n_chroms + 1, dtype=OFFSET_DTYPE)
    index = 0
    for start, length, value in zip(*rlencode(chrom_ids)):
        chrom_offset[index] = start
        index += 1
    chrom_offset[index] = n_bins

    return chrom_offset

def get_pixel_index(grp, n_bins, n_pixels):
    bin1 = np.array(grp["bin1_id"])
    bin1_offset = np.zeros(n_bins + 1, dtype=OFFSET_DTYPE)
    curr_val = 0

    for start, length, value in zip(*rlencode(bin1, 1000000)):
        bin1_offset[curr_val: value + 1] = start
        curr_val = value+1

    bin1_offset[curr_val:] = n_pixels

    return bin1_offset

def write_index(grp, chrom_offset, bin_offset, h5_opts):
    grp.create_dataset(
        "chrom_offset",
        shape=(len(chrom_offset),),
        dtype=OFFSET_DTYPE,
        data=chrom_offset, **h5_opts
    )
    grp.create_dataset(
        "bin1_offset",
        shape=(len(bin_offset),),
        dtype=OFFSET_DTYPE,
        data=bin_offset, **h5_opts
    )

# append functions
def write_embed(grp, embed, h5_opts):
    vec_pca = PCA(n_components=2).fit_transform(embed)
    vec_umap = UMAP(n_components=2).fit_transform(embed)

    grp.create_dataset("pca", shape=(len(vec_pca),2), data= vec_pca, **h5_opts)
    grp.create_dataset("umap", shape=(len(vec_umap),2), data= vec_umap, **h5_opts)
    #grp.create_dataset("label", shape=(len(label),), data= label, **h5_opts)

def write_meta(grp, data, label_name, h5_opts):
    cell_type = np.array(data[label_name])
    ascii_label = np.char.encode(cell_type, 'ascii')
    grp.create_dataset("label", shape=(len(ascii_label),), data= ascii_label, **h5_opts)

def write_track(source_dataset, cur_grp, track_type: str):
    if track_type in cur_grp:
        del cur_grp[track_type]
    copyDataset(source_dataset, cur_grp, track_type)

def process_cells_range(start, end, process_id, temp_folder, neighbor_num, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file, embedding_name, h5_opts, n_bins, progress, process_cnt):
    temp_h5_path = os.path.join(temp_folder, f"temp_cells_{process_id}.h5")
    
    def process_single_cell(parent_group, cell_index, neighbor_num, is_raw):
        cur_cell_grp = parent_group.create_group(f"cell_{cell_index}")
        cell_grp_pixels = cur_cell_grp.create_group("pixels")
        setup_pixels(cell_grp_pixels, n_bins, h5_opts)
        write_pixels(
            cell_grp_pixels, contact_map_file, raw_map_file, np_chroms_names,
            chrom_offset, cell_index, neighbor_num, list(cur_cell_grp["pixels"]),
            res, cytoband_file, embedding_name, process_cnt, is_raw
        )
        n_pixels = len(cell_grp_pixels.get("bin1_id"))
        bin_offset = get_pixel_index(cell_grp_pixels, n_bins, n_pixels)
        grp_index = cur_cell_grp.create_group("indexes")
        write_index(grp_index, chrom_offset, bin_offset, h5_opts)
        return n_pixels

    def update_progress(progress, idx, num, process_id):
        with progress[idx].get_lock():
            progress[idx].value += 1
            if progress[idx].value % 10 == 0:  # Print progress every 10 cells
                print(f"Process {process_id} has completed {progress[idx].value} cells for neighbor {num}")

    with h5py.File(temp_h5_path, 'w') as hdf:
        # Write raw data
        raw_grp = hdf.create_group("raw")
        for i in range(start, end):
            process_single_cell(raw_grp, i, neighbor_num[0], True)

        # Write imputed data
        for idx, num in enumerate(neighbor_num):
            neigh_group = hdf.create_group(f"imputed_{num}neighbor")
            for i in range(start, end):
                print(f"Process {process_id} processing cell {i} neighbor {num}")
                process_single_cell(neigh_group, i, num, False)
                update_progress(progress, idx, num, process_id)

class SCHiCGenerator:
    def __init__(self, config_path):
        self.data_folder = ""
        self.output_path = ""
        self.contact_map_path = ""
        self.raw_map_path = ""
        self.embed_file_name = ""
        self.meta_file_name = ""
        self.embed_label = ""
        self.tracks = []
        self.neighbor_num = 0
        self.embedding_name = ""
        self.chrom_size_file_name = ""
        self.cytoband_file_name = ""
        self.cell_cnt = 0
        self.resolutions = []
        self.h5_opts = {}
        self.process_cnt = 0
        self.load_base_config(config_path)
    
    def load_base_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
            for key, default in self.__dict__.items():
                if not key.startswith('_'):
                    setattr(self, key, config.get(key, default))
            print("Config loaded")
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {config_path}")

    def create_all_h5(self):
        try:
            self.validate_paths()
            print("all data folders and files validated")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error: {e}")
            os.remove(self.output_path)
            sys.exit(1)
        with h5py.File(self.output_path, 'w') as hdf:
            hdf.create_group("resolutions")

        for res in self.resolutions:
            print("Creating resolution: "+str(res))
            self.create_res_h5(res) 
   
    def create_res_h5(self, res):
        contact_map_file = os.path.join(self.data_folder, self.contact_map_path)
        raw_map_file = os.path.join(self.data_folder, self.raw_map_path)
        cytoband_file = os.path.join(self.data_folder, self.cytoband_file_name)
        n_bins = n_chroms = 0
        chrom_offset = []
        with h5py.File(self.output_path, 'r+') as hdf:
            res_grp = hdf.create_group("resolutions/"+str(res))
           
            # write res/chrom
            grp_chroms = res_grp.create_group("chroms")
            chrom_size_file = os.path.join(self.data_folder, self.chrom_size_file_name)
            np_chroms_names, np_chroms_length = write_chroms(
                grp_chroms, chrom_size_file, self.h5_opts)

            # write res/bin
            grp_bins = res_grp.create_group("bins")
            write_bins(res, np_chroms_names,
                       np_chroms_length, grp_bins, self.h5_opts)

            layer_groups = res_grp.create_group("layers")

            n_bins = len(res_grp["bins"].get("chrom"))
            n_chroms = len(res_grp["chroms"].get("length"))
            chrom_offset = get_bin_index(
                    res_grp["bins"], n_chroms, n_bins)
            if self.process_cnt == 1:
                self.process_cells(layer_groups, n_bins, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file)
        if self.process_cnt > 1:
            self.parallel_process_cells(contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file, n_bins)
        
        with h5py.File(self.output_path, 'a') as hdf:
            layer_groups = hdf[f"resolutions/{res}/layers"]
            self.process_groups(layer_groups, n_bins, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file)
          
    def parallel_process_cells(self, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file, n_bins):
        # Determine the range of cells each process should handle
        cells_per_process = (self.cell_cnt + self.process_cnt - 1) // self.process_cnt
        temp_folder = "temp_h5"
        os.makedirs(temp_folder, exist_ok=True)
        
        # Create a shared progress counter
        progress = [mp.Value('i', 0) for _ in self.neighbor_num]
        
        # Create a pool of worker processes
        processes = []
        for process_id in range(self.process_cnt):
            start = process_id * cells_per_process
            end = min((process_id + 1) * cells_per_process, self.cell_cnt)
            if start < end:
                p = mp.Process(target=process_cells_range, args=(start, end, process_id, temp_folder, self.neighbor_num, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file, self.embedding_name, self.h5_opts, n_bins, progress, self.process_cnt))
                processes.append(p)
                p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        # Merge the temporary HDF5 files into the original HDF5 file
        merge_temp_h5_files(self.output_path, temp_folder, self.process_cnt, res, self.neighbor_num)

    def process_cells(self, layer_groups, n_bins, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file):
        print("Processing cells...")
        def process_single_cell(parent_group, cell_index, neighbor_num, is_raw):
            cur_cell_grp = parent_group.create_group(f"cell_{cell_index}")
            cell_grp_pixels = cur_cell_grp.create_group("pixels")
            setup_pixels(cell_grp_pixels, n_bins, self.h5_opts)
            write_pixels(
                cell_grp_pixels, contact_map_file, raw_map_file, np_chroms_names,
                chrom_offset, cell_index, neighbor_num, list(cur_cell_grp["pixels"]),
                res, cytoband_file, self.embedding_name, self.process_cnt, is_raw
            )
            n_pixels = len(cell_grp_pixels.get("bin1_id"))
            bin_offset = get_pixel_index(cell_grp_pixels, n_bins, n_pixels)
            grp_index = cur_cell_grp.create_group("indexes")
            write_index(grp_index, chrom_offset, bin_offset, self.h5_opts)

        ## Write raw data
        print("Writing raw data...")
        raw_grp = layer_groups.create_group("raw")
        for i in range(self.cell_cnt):
            process_single_cell(raw_grp, i, self.neighbor_num[0], True)

        ## Write imputed data
        for num in self.neighbor_num:
            print(f"Writing impute data for neighbor={num}...")
            neigh_group = layer_groups.create_group(f"imputed_{num}neighbor")
            for i in range(self.cell_cnt):
                print(f"Processing cell {i} for neighbor {num}...")
                process_single_cell(neigh_group, i, num, False)

    def process_groups(self, layer_grp, n_bins, contact_map_file, raw_map_file, np_chroms_names, chrom_offset, res, cytoband_file):
        print("processing psuedo-bulk data...")
        meta_file = os.path.join(self.data_folder, self.meta_file_name)
        if fileType(meta_file) !="pkl":
            raise FileNotFoundError(f"The file '{meta_file}' is not a pickle file.")
        
        cell_type_dict = get_celltype_dict(meta_file, self.embed_label)
        
        def process_group(parent_group, cell_type, cells, neighbor_num, is_raw):
            cur_celltype_grp = parent_group.create_group(cell_type)
            cell_celltype_pixels = cur_celltype_grp.create_group("pixels")
            setup_pixels(cell_celltype_pixels, n_bins, self.h5_opts)
            write_group_pixels(
                cell_celltype_pixels, contact_map_file, raw_map_file, np_chroms_names,
                chrom_offset, cells, neighbor_num, list(cur_celltype_grp["pixels"]),
                res, cytoband_file, self.embedding_name, self.process_cnt, is_raw
            )
            n_raw_pixels = len(cell_celltype_pixels.get("bin1_id"))
            raw_bin_offset = get_pixel_index(cell_celltype_pixels, n_bins, n_raw_pixels)
            raw_grp_index = cur_celltype_grp.create_group("indexes")
            write_index(raw_grp_index, chrom_offset, raw_bin_offset, self.h5_opts)

        for cell_type, cells in cell_type_dict.items():
            raw_grp = layer_grp["raw"]
            print(f"processing raw group for type {cell_type}")
            process_group(raw_grp, cell_type, cells, self.neighbor_num[0], True)

            for num in self.neighbor_num:
                print(f"processing imputed_{num}neighbor group for type {cell_type}")
                impute_grp = layer_grp[f"imputed_{num}neighbor"]
                process_group(impute_grp, cell_type, cells, num, False)

    def append_h5(self, atype: str):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " not exists")
        if(atype=='embed'):
            print("appending cell embeddings...")
            embed_file = os.path.join(self.data_folder, self.embed_file_name)
            cell_embeddings = np.load(embed_file)

            with h5py.File(self.output_path, 'a') as hdf:
                if 'embed' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['embed']

                emb_grp = hdf.create_group('embed')
                write_embed(emb_grp, cell_embeddings, self.h5_opts)
        elif(atype=='meta'):
            print("appending cell meta data...")
            meta_file = os.path.join(self.data_folder, self.meta_file_name)
            if fileType(meta_file) !="pkl":
                raise FileNotFoundError(f"The file '{meta_file}' is not a pickle file.")
            
            label_info = pickle.load(open(meta_file, "rb"))
            with h5py.File(self.output_path, 'a') as hdf:
                if 'meta' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['meta']

                meta_grp = hdf.create_group('meta')
                write_meta(meta_grp, label_info, self.embed_label, self.h5_opts)
        elif(atype=='1dtrack'):
            print("appending 1d track data...")
            for res_tracks in self.tracks:
                for track_type, track_f_name in res_tracks["track_object"].items():   
                    track_file = os.path.join(self.data_folder, track_f_name)
                    if not os.path.exists(track_file):
                        raise FileNotFoundError(f"The file '{track_file}' does not exist.")
            
            with h5py.File(self.output_path, 'a') as hdf:
                for res_tracks in self.tracks:
                    cur_res = res_tracks[f"resolution"]
                    res_grp = hdf[f"resolutions/{cur_res}/layers"]
                    # Ensure the 'tracks' group exists
                    if 'tracks' in res_grp:
                        del res_grp['tracks']
                    tracks_grp = res_grp.create_group('tracks')
                    # Open track files outside the inner loops to minimize the number of open calls
                    track_files = {}
                    for track_type, track_f_name in res_tracks["track_object"].items():
                        track_file_path = os.path.join(self.data_folder, track_f_name)
                        track_files[track_type] = h5py.File(track_file_path, 'r')
                    for track_type, f in track_files.items():
                        if track_type in tracks_grp:
                            del tracks_grp[track_type]
                        track_grp = tracks_grp.create_group(track_type)
                        for cell_id in range(self.cell_cnt):
                            tgt_grp = track_grp.create_group(f"cell_{cell_id}")
                            source_grp = f[f"insulation/cell_{cell_id}"]
                            write_track(source_grp, tgt_grp, track_type)
                    for f in track_files.values():
                        f.close()
        else:
            raise ValueError("Invalid atype provided. Only 'embed, meta, 1dtrack' is supported.")

    def validate_paths(self):
        if os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " already exists")
        
        contact_map_folder = os.path.join(self.data_folder, self.contact_map_path)
        if not os.path.exists(contact_map_folder):
            raise FileNotFoundError(f"The folder '{contact_map_folder}' does not exist.")

        raw_map_folder = os.path.join(self.data_folder, self.raw_map_path)
        if not os.path.exists(contact_map_folder):
            raise FileNotFoundError(f"The folder '{raw_map_folder}' does not exist.")

        embed_file = os.path.join(self.data_folder, self.embed_file_name)
        if not os.path.exists(embed_file):
            raise FileNotFoundError(f"The file '{embed_file}' does not exist.")

        meta_file = os.path.join(self.data_folder, self.meta_file_name)
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"The file '{meta_file}' does not exist.") 

        chrom_size_file = os.path.join(self.data_folder, self.chrom_size_file_name)
        if not os.path.exists(chrom_size_file):
            raise FileNotFoundError(f"The file '{chrom_size_file}' does not exist.")
        
        cytoband_file = os.path.join(self.data_folder, self.cytoband_file_name)
        if not os.path.exists(cytoband_file):
            raise FileNotFoundError(f"The file '{cytoband_file}' does not exist.")

    def print_schema(self):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " does not exist")
        print_hdf5_structure(self.output_path)
    
    def check_schema(self):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " does not exist")
        is_valid, message = check_hdf5_structure(self.output_path)
        if is_valid:
            print("HDF5 structure is valid.")
        else:
            print(f"Invalid HDF5 structure: {message}")

def generate_hic_file(config_path, mode, types=[]):
    generator = SCHiCGenerator(config_path)
    if mode == 'create':
        generator.create_all_h5()
    elif mode == 'append':
        for t in types:
            generator.append_h5(t)
    elif mode == 'print':
        generator.print_schema()
    elif mode == 'check':
        generator.check_schema()

if __name__ == "__main__":
    generator =  SCHiCGenerator("../config.JSON")
    # try:
    generator.create_all_h5()
    generator.append_h5("embed")
    generator.append_h5("1dtrack")
    generator.append_h5("meta")
    # except Exception as e:
    #     print(repr(e))
    #     os.remove(generator.output_path)
