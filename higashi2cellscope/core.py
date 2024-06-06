import h5py
import numpy as np
import json
import os
import sys
import pickle
from sklearn.decomposition import PCA
from umap import UMAP
from collections import defaultdict
from utils import rlencode, fileType, copyDataset
from matrixParser import MatrixParser
import multiprocessing as mp
import tables as pt

# Constants
sentinel = None

CHROM_DTYPE = np.dtype("S")
CHROMID_DTYPE = np.int32
CHROMSIZE_DTYPE = np.int32
COORD_DTYPE = np.int32
BIN_DTYPE = np.int64
COUNT_DTYPE = np.float32
OFFSET_DTYPE = np.int64

"""
├── meta
├── embeddings
│    ├── PCA
│    └── UMAP
└── resolutions
     ├── 10000
     │   ├── bins
     |   |   ├── chrom
     |   |   ├── start
     |   |   └── end
     │   ├── chroms
     |   |   ├── name
     |   |   └── length
     │   └── cells
     │       ├── cell_1
     │       │   ├── pixels
     |       |   |   ├── bin1_id
     |       |   |   ├── bin2_id
     |       |   |   └── count
     │       │   ├── indexes
     │       │   │   ├── chrom_offset
     |       |   │   └── bin1_offset
     │       │   ├── tracks
     |       |       └── insulation
     │       ├── cell_2
     │       │   ├── pixels
     │       │   └── indexes
     │   
     ├── 50000
     │   

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

def write_pixels(grp, impute_dir, raw_dir, chrom_list, chrom_offset, cell_id, neighbors, columns,res, cytoband_path, embedding_name):
    cellMatrixParser = MatrixParser(
         impute_dir, raw_dir, chrom_list, chrom_offset, cell_id, neighbors, res, cytoband_path, embedding_name)
    m_size = 0
    for chunk in cellMatrixParser:
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

def write_meta(grp, data, h5_opts):
    cell_type = np.array(data['cluster label'])
    ascii_label = np.char.encode(cell_type, 'ascii')
    grp.create_dataset("label", shape=(len(ascii_label),), data= ascii_label, **h5_opts)

def write_track(source_dataset, cur_grp, track_type: str):
    if track_type in cur_grp:
        del cur_grp[track_type]
    copyDataset(source_dataset, cur_grp, track_type)

# Worker function to process each cell
def process_cell(inqueue, output, cell_groups, contact_map_file, raw_map_file, np_chroms_names,
                 chrom_offset, neighbor_num, res, cytoband_file, embedding_name, h5_opts, n_bins):
    for i in iter(inqueue.get, sentinel):
        output.put(('start_cell', i))
        cur_cell_grp = cell_groups.create_group(f"cell_{i}")
        
        cell_grp_pixels = cur_cell_grp.create_group("pixels")
        setup_pixels(cell_grp_pixels, n_bins, h5_opts)
        
        write_pixels(cell_grp_pixels, contact_map_file, raw_map_file, np_chroms_names,
                     chrom_offset, i, neighbor_num, list(cur_cell_grp["pixels"]), res, cytoband_file, embedding_name)
        
        n_pixels = len(cur_cell_grp["pixels"].get("bin1_id"))
        bin_offset = get_pixel_index(cur_cell_grp["pixels"], n_bins, n_pixels)
        
        grp_index = cur_cell_grp.create_group("indexes")
        write_index(grp_index, chrom_offset, bin_offset, h5_opts)
        output.put(('end_cell', i))

def handle_output(output, hdf_filename):
    with pt.openFile(hdf_filename, mode='w') as hdf:
        while True:
            args = output.get()
            if args:
                method, args = args
                if method == 'start_cell':
                    print(f"Starting cell {args}")
                elif method == 'end_cell':
                    print(f"Ending cell {args}")
                else:
                    getattr(hdf, method)(*args)
            else:
                break

class SCHiCGenerator:
    def __init__(self, config_path):
        self.data_folder = ""
        self.output_path = ""
        self.contact_map_path = ""
        self.raw_map_path = ""
        self.embed_file_name = ""
        self.meta_file_name = ""
        self.tracks = []
        self.neighbor_num = 0
        self.embedding_name = ""
        self.chrom_size_file_name = ""
        self.cytoband_file_name = ""
        self.cell_cnt = 0
        self.resolutions = []
        self.h5_opts = {}
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

            cell_groups = res_grp.create_group("cells")

            n_bins = len(res_grp["bins"].get("chrom"))
            n_chroms = len(res_grp["chroms"].get("length"))
            chrom_offset = get_bin_index(
                    res_grp["bins"], n_chroms, n_bins)

            self.parallel_process_cells(self.cell_cnt, cell_groups, contact_map_file, raw_map_file, np_chroms_names,
                        chrom_offset, self.neighbor_num, res, cytoband_file, self.embedding_name, self.h5_opts, n_bins)
            
            # for i in range(self.cell_cnt):
            #     print("cell "+str(i)+":")
            #     cur_cell_grp = cell_groups.create_group(
            #         "cell_"+str(i))
                
            #     cell_grp_pixels = cur_cell_grp.create_group("pixels")
            #     setup_pixels(cell_grp_pixels, n_bins, self.h5_opts)
            #     write_pixels(cell_grp_pixels,  contact_map_file, raw_map_file, np_chroms_names,
            #                  chrom_offset, i, self.neighbor_num, list(cur_cell_grp["pixels"]), res, cytoband_file, self.embedding_name)
            #     n_pixels = len(cur_cell_grp["pixels"].get("bin1_id"))
            #     bin_offset = get_pixel_index(
            #         cur_cell_grp["pixels"], n_bins, n_pixels)
            #     grp_index = cur_cell_grp.create_group("indexes")

            #     write_index(grp_index, chrom_offset, bin_offset, self.h5_opts)
    
    def append_h5(self, atype: str):
        if not os.path.exists(self.output_path):
            raise RuntimeError("sc-HiC file: " +  self.output_path + " not exists")
        if(atype=='embed'):
            print("appeding cell embeddings...")
            embed_file = os.path.join(self.data_folder, self.embed_file_name)
            cell_embeddings = np.load(embed_file)

            with h5py.File(self.output_path, 'a') as hdf:
                if 'embed' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['embed']

                emb_grp = hdf.create_group('embed')
                write_embed(emb_grp, cell_embeddings, self.h5_opts)
        elif(atype=='meta'):
            print("appeding cell meta data...")
            meta_file = os.path.join(self.data_folder, self.meta_file_name)
            if fileType(meta_file) !="pkl":
                raise FileNotFoundError(f"The file '{meta_file}' is not a pickle file.")
            
            label_info = pickle.load(open(meta_file, "rb"))
            with h5py.File(self.output_path, 'a') as hdf:
                if 'meta' in hdf:
                    # Delete the group 'rgrp'
                    del hdf['embed']

                meta_grp = hdf.create_group('meta')
                write_meta(meta_grp, label_info, self.h5_opts)
        elif(atype=='1dtrack'):
            print("appeding 1d track data...")
            for res_tracks in self.tracks:
                for track_type, track_f_name in res_tracks["track_object"].items():   
                    track_file = os.path.join(self.data_folder, track_f_name)
                    if not os.path.exists(track_file):
                        raise FileNotFoundError(f"The file '{track_file}' does not exist.")
            
            with h5py.File(self.output_path, 'a') as hdf:
                for res_tracks in self.tracks:
                    cur_res = res_tracks["resolution"]
                    cell_groups = [f"resolutions/{cur_res}/cells/cell_{i}" for i in range(self.cell_cnt)]
                    
                    # Open track files outside the inner loops to minimize the number of open calls
                    track_files = {}
                    for track_type, track_f_name in res_tracks["track_object"].items():
                        track_file_path = os.path.join(self.data_folder, track_f_name)
                        track_files[track_type] = h5py.File(track_file_path, 'r')

                    for cell_id, cell_group in enumerate(cell_groups):
                        cur_grp = hdf[cell_group]
                        if "tracks" in cur_grp:
                            del cur_grp["tracks"]
                        track_grp = cur_grp.create_group('tracks')
                        for track_type, f in track_files.items():
                            source_grp = f[f"insulation/cell_{cell_id}"]
                            write_track(source_grp, track_grp, track_type)
                    
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
