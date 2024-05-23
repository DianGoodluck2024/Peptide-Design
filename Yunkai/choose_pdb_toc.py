import h5py

# 打开 HDF5 文件
h5_file = "output.h5"
with h5py.File(h5_file, 'r') as h5f:
    # 列出文件中的所有组
    for pdb_id in h5f.keys():
        print(f"PDB ID: {pdb_id}")
        group = h5f[pdb_id]
        
        # 打印每个组中的数据集
        for dataset_name in group.keys():
            dataset = group[dataset_name]
            print(f"  Dataset: {dataset_name}")
            print(f"  Shape: {dataset.shape}")
            print(f"  Data: {dataset[()]}")
