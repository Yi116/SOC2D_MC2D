# In[]
import importlib
import numpy as np
import h5py
import time
from contextlib import closing

# In[]
import spin.spin_texture as spin_texture
importlib.reload(spin_texture)
from spin.spin_texture import *

# In[]

data = np.load("/home/jyin/workspace/MoS2/spindata.npz")
spins = data['spins']

# In[]
file_name_h5 = "/home/jyin/workspace/gene_data/MoS2/data_MoS2_electrons.hdf5"
f1 = h5py.File(file_name_h5, 'r')
# In[]
try:
    with closing(open_hdf5_with_retry(file_name_h5, 'r')) as f1:
        # Your file operations here
        qemesh = np.array(f1['mesh'])
        rots_recip = np.array(f1['rots'])
        kpoints = np.array(f1['kpoints'])
        bands0 = np.array(f1['transpbands26'])
        bands1 = np.array(f1['transpbands27'])

        pass
except Exception as e:
    print(f"Error opening HDF5 file: {e}")

