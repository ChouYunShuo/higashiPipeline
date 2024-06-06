import numpy as np
import pandas as pd
import h5py
OFFSET_DTYPE = np.int64


def rlencode(array, chunksize=None):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    array = np.array(array)
    n = len(array)
    if n == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=array.dtype),
        )

    if chunksize is None:
        chunksize = n

    starts, values = [], []
    last_val = np.nan
    for i in range(0, n, chunksize):
        x = array[i: i + chunksize]
        locs = where(x[1:] != x[:-1]) + 1
        if x[0] != last_val:
            locs = np.r_[0, locs]
        starts.append(i + locs)
        values.append(x[locs])
        last_val = x[-1]
    starts = np.concatenate(starts)
    lengths = np.diff(np.r_[starts, n])
    values = np.concatenate(values)

    return starts, lengths, values


def create_mask(res,c_path,  k=30, chrom="chr1", origin_sparse=None):
	final = np.array(np.sum(origin_sparse, axis=0).todense())
	size = origin_sparse[0].shape[-1]
	a = np.zeros((size, size))
	if k > 0:
		for i in range(min(k, len(a))):
			for j in range(len(a) - i):
				a[j, j + i] = 1
				a[j + i, j] = 1
		a = np.ones_like((a)) - a
	
	gap = np.sum(final, axis=-1, keepdims=False) == 0
	
	gap_tab = pd.read_table(c_path, sep="\t", header=None)
	gap_tab.columns = ['chrom', 'start', 'end', 'sth', 'type']
	gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
	start = np.floor((np.array(gap_list['start'])) / res).astype('int')
	end = np.ceil((np.array(gap_list['end'])) / res).astype('int')
	
	for s, e in zip(start, end):
		a[s:e, :] = 1
		a[:, s:e] = 1
	a[gap, :] = 1
	a[:, gap] = 1
	
	return a

def copyGroup(source_group, dest_group):
    for key, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            # Create a new dataset in dest_group with the same name, shape, type and data
            dest_group.create_dataset(key, data=item[...], dtype=item.dtype, shape=item.shape)
        elif isinstance(item, h5py.Group):
            # Create a new subgroup in dest_group with the same name
            dest_subgroup = dest_group.create_group(key)
            # Recursively copy the items in this subgroup to the new subgroup
            copyGroup(item, dest_subgroup)

def copyDataset(item, dest_group, track_type):
    if isinstance(item, h5py.Dataset):
        # Create a new dataset in dest_group with the same name, shape, type and data
        dest_group.create_dataset(track_type, data=item[...], dtype=item.dtype, shape=item.shape)
        

def fileType(fname: str):
    return 'npy' if fname.endswith('npy') else 'pkl' if fname.endswith('pkl') else 'csv' if fname.endswith('csv') else 'na'

if __name__ == "__main__":
    with h5py.File("../../../scHDF5_data/Lee_et_al_001.h5", 'r') as f:
        dataset = f["resolutions/500000/cells/cell_1/tracks"]
        print(list(dataset.keys()))
        # print(len(f[f'resolutions/100000/bins'].get("chrom")))
        # dataset = f["resolutions/100000/cells/cell_id1/pixels/count"]
        # print(f"Dataset shape: {dataset.shape}")
        # print(f"Dataset size: {dataset.size}")
        # cells = ["cell_0"]