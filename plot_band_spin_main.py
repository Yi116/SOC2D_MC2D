# In[]
import numpy as np
import h5py

#%%


if __name__ == '__main__':
# In[]
    """
    # Load the data
    """
    

# In[]
    path_npy_spin_MoS2 = "/home/jyin/workspace/gene_data/MoS2/spindata.npz"
    path_h5_MoS2 = "/home/jyin/workspace/h5py_storage/MoS2_test2/data_MoS2_electrons.hdf5"

# In[]
    path_npy_spin_LuIO = "/home/jyin/workspace/gene_data/LuIO/spinsLUIO.npy"
    path_npz_LuIO = "/home/jyin/workspace/gene_data/LuIO/LuIOFET.npz"


# %%


    
# In[]
    path_spin = path_npy_spin_MoS2
    file_name_h5 = path_h5_MoS2
# In[]
    path_spin = path_npy_spin_LuIO
    file_npz = path_npz_LuIO

# %%
    import importlib
    import spin.spin_texture as spin_texture
    importlib.reload(spin_texture)
    from spin.spin_texture import *





# In[]
    data_spin = np.load(path_spin, allow_pickle=True)
    # spins = data_spin['spins']
# In[]
    data_f1 = h5py.File(file_name_h5, 'r')
    data = data_f1
# In[]
    data_npz = np.load(file_npz, allow_pickle=True)
    data = data_npz

# In[]
    qemesh = data['mesh']
    bands_init = data['bands']
    BZ_pg_rots = data['rots']
    kpoints = data['kpoints']
    equiv = data['equiv']
# In[]
    key_band0, key_band1 = bands_init.item().keys()
# In[]
    band0 = bands_init.item()[key_band0]
    band1 = bands_init.item()[key_band1]

# In[]
    if band0.shape[0] != kpoints.shape[0]:
        band0_reshaped = reshape_energy(band0, index_max_kpoint=data_spin.shape[1] -1)
    if band1.shape[0] != kpoints.shape[0]:
        band1_reshaped = reshape_energy(band1, index_max_kpoint=data_spin.shape[1] -1)

# In[]
    ks, eq, fromrot, spinflip = checked_k_points_syms(qemesh=qemesh,
                                                      rots_recip=BZ_pg_rots[:,0])

# In[]
    bndsBZ, spinBZ = list_found_spin_texture_energy(bands0=band0_reshaped,
                                                    bands1=band1_reshaped,
                                                    equiv=eq,
                                                    kpoints=ks,
                                                    rots_recip=BZ_pg_rots[:,0],
                                                    fromrot=fromrot,
                                                    spinflip=spinflip,
                                                    spins=data_spin)

# In[]
    bands_to_plot = np.stack((bndsBZ[0][:,2], bndsBZ[1][:,2]), axis=1)

# In[]
    kpoints_to_plot = bndsBZ[0][:,:2]
    kpoints_to_plot_3D = bndsBZ[0]
# In[]
    spinBZ_to_plot = np.stack((spinBZ[0],spinBZ[1] ), axis=1)

# In[]
    ### RGBA color for the bands and spin textures.
    bands_color = np.array([[0.,0.,1.,1.],
                            [1.,0.,0.,1.]])

    spins_color = np.array([[0.,0.,0.,1.],
                            [1.,1.,0.,1.]])
# In[]
    fig_3D_band_spin = plot_bands_2D_with_spin_interac(bands=bands_to_plot,
                                                       spin_vectors_each_band=spinBZ_to_plot,
                                                       kpoints=kpoints_to_plot,
                                                       vector_density=50,
                                                       want_plot_spin=False,
                                                       scale_factor_spin=1.0,
                                                       bands_color=bands_color,
                                                       spins_color=spins_color,
                                                       grid_size=300,
                                                       factor_energy=3,
                                                       )  # Increased from 100 to 200


# In[]
    path_save = './bands_inter_plot/LuIO_1Rashba.html'

# In[]

    fig_3D_band_spin.write_html(path_save)

# In[]
    
    fig_spin_proj = plot_spin_xy_projection(kpoints=kpoints_to_plot,
                                          spin_vectors_each_band=spinBZ_to_plot,
                                          bands=bands_to_plot,  # Add bands
                                          target_energy=-2.8,    # Choose energy level 
                                          energy_tolerance=0.05,
                                          spins_color=spins_color,
                                          vector_density=200,
                                          scale_factor_spin=1.5)
    
    fig_spin_proj.show()

# In[] 
############### Try to calculated the gradient of a given band.

# In[]
# 1st step: organize the data to a 2D dimension + 1 dimension who is [ka, kb, energy]
# 2nd step: calculate the gradient of the band
# 3rd step: plot the gradient of the band
# 4th step: find local minimum and maximum and defint some windows of energy depend on those local minimum and maximum.


# In[]
# 1st step: organize the data to a 2D dimension + 1 dimension
    band_3D_array = sort_Ek_in_2D_plan(bndsBZ[1])
    


# In[]
###### Calculate the gradient of the band with the function np.gradient
    gradient = calculate_gradient_vNumpy(band_3D_array)
