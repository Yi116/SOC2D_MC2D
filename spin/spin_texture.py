
# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
# from band_analyse import *
from aiida.orm import BandsData, KpointsData, StructureData
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
import plotly
import plotly.graph_objects as go
import h5py
import time
import plotly.io as pio
import json
import re
from IPython.core.debugger import set_trace
import math

# from aiida.tools.data.array.kpoints.seekpath import get_explicit_kpoints_path
from aiida.tools.data.array.kpoints.legacy import get_kpoints_path, get_explicit_kpoints_path

from analyse_symmetry import Atoms_symmetry_group_direct_space

from find_groups import change_basis

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from copy import deepcopy

from find_groups import find_rotation_axis

# %%
def qe_nint(vec):
    """
    Convert find the nealest integer of each element of a vector.
    Note: 1e-6 is used as the tolerance to treat the elements in interval [0.5 - 1e-6 * component, and 0.5 + 1e-6 * componenent] as 0.5
    """
    return np.rint(vec + 1e-6*np.sign(vec))


def open_hdf5_with_retry(filename, mode='r', max_attempts=3, delay=1):
    for attempt in range(max_attempts):
        try:
            return h5py.File(filename, mode)
        except BlockingIOError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                raise

    
def create_hexagonal_2Dbrillouin_zone(n_points : int):
    """
    We know in hexagonal 1st BZ's basis choise, the angle between a and b is 60 degree. and a and b 's direction are from center of hexagon to the middle of edge of hexagone. 
    """
    # Define basis vectors (60° between them)
    a = np.array([1.0, 0.0])
    b = np.array([0.5, np.sqrt(3)/2])
    
    # Normalize basis vectors
    a = a / 2  # This ensures coordinates range from -0.5 to 0.5
    b = b / 2
    
    # Create a square grid of normalized coordinates
    u = np.linspace(0, 1, n_points)
    v = np.linspace(0, 1, n_points)

    points_scaled = []
    for i in u:
        for j in v:
            points_scaled.append([i, j])
    points_scaled = np.array(points_scaled)



    points_scaled_irrBZ = []
    for point in points_scaled:
        pent = point[0] >= point[1]
        edge = point[1] <= -2 * point[0] + 1
        base = point[0] >= 0.
        if pent and edge and base:
            points_scaled_irrBZ.append(point)
    points_scaled_irrBZ = np.array(points_scaled_irrBZ)

    #### just add a 0 as the third component to make operation on rotations.
    points_scaled_irrBZ = np.hstack((points_scaled_irrBZ, np.zeros((points_scaled_irrBZ.shape[0], 1))))


    #### rotate by C6, and for anthoer set of mirror on a = b. than rotation this set by C6. than combine them. you will get the whole hexagonal BZ.
    #### just the set of the mirror, the mirro set shouldn't contain the pent a = b and a = 0. if not you repeat some point after C6 rotation.

    rotation_C6 = np.array([[ 0., -1.,  0.],
                            [ 1.,  1.,  0.],
                            [ 0.,  0.,  1.]])
    mirror_a_egal_b = np.array([[0.,1.,0.],
                                [1.,0.,0.],
                                [0.,0.,1.]])

    mirror_on_irrBZ = []
    for point in points_scaled_irrBZ:
        
        if point[0] != point[1] and point[0] != 0 :
            point_m = mirror_a_egal_b @ point
            mirror_on_irrBZ.append(point_m)
    
    full_BZ_points = np.concatenate((points_scaled_irrBZ, mirror_on_irrBZ), axis=0)
    iter_1 = points_scaled_irrBZ
    iter_2 = mirror_on_irrBZ

    
    for i in range(5):
        iter_1_rotated = []
        for point in iter_1:
            point_rotated = rotation_C6 @ point
            iter_1_rotated.append(point_rotated)
        iter_1_rotated = np.array(iter_1_rotated)
        full_BZ_points = np.concatenate((full_BZ_points, iter_1_rotated), axis=0)
        iter_1 = iter_1_rotated
        iter_2_rotated = []
        for point in iter_2:
            point_rotated = rotation_C6 @ point
            iter_2_rotated.append(point_rotated)
        
        iter_2_rotated = np.array(iter_2_rotated)
        full_BZ_points = np.concatenate((full_BZ_points, iter_2_rotated),axis=0)
        iter_2 = iter_2_rotated

    ### below are commanded cause we don't need to set the coordinates in cartesian
    # basis_ab_angle_120 = set_a_hex_cell()
    # basis_ab_angle_60 = basis_ab_angle_120.copy()
    # cos_pro = basis_ab_angle_120[0,1]
    # basis_ab_angle_60[0,1] = -1 * cos_pro
    # # Convert to Cartesian coordinates
    # points_cart = []
    # for point in points_scaled:
    #     point_cart = basis_ab_angle_60 @ point
    
    
    
    return full_BZ_points, points_scaled_irrBZ, mirror_a_egal_b



def checked_k_points_syms(qemesh, rots_recip,  tol = 1e-5):
    """
    This function do the following:
        1, generated an k point mesh by argument qemesh. Find the symmetry equivalent k points' link
    Arg :
    -----
        qemesh : the grid number of the full BZ on each dimension of ka, kb, kc where a, b, c denote the reciprocal lattice vectors in the reciprocal space.
        rots_recip is a list of list of [[rot, t_rev] for rot in rotations] (result of get_qe_recip_syms)

    Return :
    -------
       ks : the k_point in the BZ (not the first BZ) 
       equiv : equiv[i]=j where j is the index of the irreducible k point to which i'th kpoint in full BZ is equivalent.
        fromrot : the ith k point have ir th rotation stored in the rots_recip
        spinflip : at each ks point, what's the spin flip operation -1 or 1.

    Note:
    ------
        #### equiv[i]=j where j is the index of the irreducible k point to which i'th kpoint in full BZ is equivalent.
        Here we suppose the time_reversal is always true. ##### TODO find thibault's code to check if it's non time_reversal.
        tol = 1e-5 is QE's tolerence
    
    
    
    """
    # nscf is the aiida nscf calculation
    # rots recip is a list of list of [[rot, t_rev] for rot in rotations] (result of get_qe_recip_syms)
    # return the k_point in the first BZ and equiv, which is an array
    # equiv[i]=j where j is the index of the irreducible k point to which i'th kpoint is equivalent.
    
    nk1, nk2, nk3 = qemesh
    time_reversal = True
    trev = 0 ### the same argument as in time_reversal
    #
    nkarr = np.array([nk1, nk2, nk3])
    # make the grid
    nk = nk1*nk2*nk3

    ### give ks the kpoints in the full BZ (Note,not the 1st BZ, but the BZ defined by surface of a and b)
    ##### here we have not the negative value but at the end of the function there is a ks = ks -np.rint(ks) to mapp the kpoints' component between -0.5 and 0.5
    ####  
    ks = np.zeros((nk,3)) #### the ks fullfilled are in scaled unit (divied by nk1, nk2, nk3)
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j*nk3 + i*nk2*nk3
                ks[n][0] = float(i)/nk1
                ks[n][1] = float(j)/nk2
                ks[n][2] = float(k)/nk2
    #
    #### init of some arrays
    equiv = np.arange(nk)
    fromrot = np.zeros(nk, dtype=int)
    spinflip = np.ones(nk, dtype=int) ### 
    detrots = np.array([np.linalg.det(r) for r in rots_recip]) ### determinate of rotations in rots_recip

    #### Note : we need detrots cause spin are pseudovectors. If you rotate a pseudovectors with a rotation matrix, it's not like rotate a normal vector.
    #### Cause a pseudovector is usually define by a vector cross product another vector, you need to inverse the sign of the rotated pseudovecteur if there are rotation inversion, like mirror.
    #### If the rotation is not rotaion inversion, you dont need to inverse it.

    #
    for i, k in enumerate(ks):## for each k point in the full BZ
        if equiv[i] == i: ### check if the k point is still equivalent to itself, if yes, need to find other equivalent k points by applying symmetry operations
            for ir, rot in enumerate(rots_recip): ## for each rotations (nonsymmorphic translation doesn't matter in reciprocal space)
                ### here we basicly apply rots in point group of the material. Then we can find for each k in full BZ, the correspond equivalent irrBZ point 
                xkr = np.dot(rot, k) ### apply the rotation, xkr is the rotated k point
                xkr = xkr - qe_nint(xkr) #np.array([x - int(round(x)) for x in xkr]) set xkr into the first BZ.
                #
                if trev==1: ### why if having time reversal symmetry, you set the xkr to -xkr
                    xkr = -xkr
                #
                vec = xkr * nkarr # this is just make xkr in the scaled coor (1/nki ... nki/nki for each component i) to the vec who have the N coord (1, 2, .. nki for each component i)
                in_the_list = np.all(np.abs(vec-qe_nint(vec))<= tol)#min([ abs(vec-int(round(x)))<= 1e-5 for x in vec])
                
                #### the in_the_list boolean is to identify if the vec (since it's from (rot @ k_scaled) and convert to N coordinate) Need to check if its components are near a int with toolerance 
                #### if yes, the vec (rotated k) is still in the grid of k mesh. Mean it's validated
                
                if in_the_list :
                    idxs = [int(x) for x in  np.remainder(qe_nint(xkr * nkarr + 2* nkarr), nkarr)]
                    j = int(idxs[2] + idxs[1]*nk3 + idxs[0]*nk2*nk3)
                    if j>i and equiv[j]==j:
                        equiv[j] = i
                        fromrot[j] = ir
                        spinflip[j] = detrots[ir]
                    else:
                        if equiv[j]!=i or j<i: print("something went wrong", j,i)
                if time_reversal:
                    vec = -xkr * nkarr
                    in_the_list = min(np.abs(vec-qe_nint(vec))<= tol)#min([ abs(vec-int(round(x)))<= 1e-5 for x in vec])
                    #
                    if in_the_list :
                        idxs = [int(x) for x in  np.remainder(qe_nint(-xkr * nkarr + 2* nkarr), nkarr)]
                        j = int(idxs[2] + idxs[1]*nk3 + idxs[0]*nk2*nk3)
                        if j>i and equiv[j]==j:
                            equiv[j] = i
                            fromrot[j] = ir
                            spinflip[j] = - detrots[ir]
                        else:
                            if equiv[j]!=i or j<i: print("something went wrong", j,i)
    #
    # Here we "slide to the first BZ" according to QE.
    # Actually, we're just centering the reciprocall cell around zero
    # This need to be done AFTER the above process of defining equiv,
    # otherwise I'm not sure we get the same result as QE
    ks = ks - np.rint(ks) 
    # Here, using the qe_nint below will get you exactly the same irr points as qe
    # but that is not something we want, because some points on zone
    # border will be separated from the rest of the wedge.
    #np.array([k - qe_nint(k) for k in ks])
    #
    #
    #
    # Note kpoints are in bohr-1 units because of line below
    return ks , equiv, fromrot, spinflip



def list_found_spin_texture_energy(bands0, bands1, equiv, kpoints, rots_recip, spinflip, fromrot,spins, tol=1e-5):
    """
    from 2 bands (should be degenerated without term SOC) calculated by DFT with SOC spin splitting in the irreducible BZ, genera

    Para:
        bands0, bands1, : DFT calulate bands energy with shape array([np.void('index': index of kpoint in para kpoints,
                                                                                'energy' : energy)])
                        Note : by comparing the energy in these 2 bands, we can know where is the degenerated point.

        
        equiv : equiv[i]=j where j is the index of the irreducible k point to which i'th kpoint in full BZ is equivalent.
    
    """
    list_bands = [0,1]
    irridxs = np.where(equiv==np.arange(len(kpoints)))[0]
    xrots = rots_recip
    detrots = np.array([np.linalg.det(rot) for rot in xrots])

    bands = np.column_stack((bands0['energy'], bands1['energy']))
    bndsBZ = {ibnd: np.zeros_like(kpoints) for ibnd in list_bands}
    spinBZ = {ibnd: np.zeros_like(kpoints) for ibnd in list_bands}

    for i0, ibnd in enumerate(list_bands): ### iteration of bands
        for iBZ, k in enumerate(kpoints): ### kpoints is in full BZ
            ir = fromrot[iBZ] # index of the rotation of irreducible k point
            iirr = equiv[iBZ] # index of the irreducible k point
            iwedge = np.where(irridxs == iirr)[0][0] # index of the wedge 
            bndsBZ[ibnd][iBZ] = np.array([k[0],k[1],bands[iwedge,ibnd]])
            ### spin polarisation after polarisation is equal to rotation apply to a spin polarisation * spinflip who is the determinate of the rotation.
            spinBZ[ibnd][iBZ] = np.dot(xrots[abs(ir)], spins[i0, iwedge])*spinflip[iBZ] 


    return bndsBZ, spinBZ


def refind_correct_spin_state_2bands(
                                     kpoints, 
                                     spins,
                                     tol=1e-5):
    """
    In fact, in some DFT result data. if you calculate the spin degenerated band and the result is band0 band1, they are not distingueshed by spin up and down. But just band0 > band1. 
    So, with the direction of spin polarisation and direction of k points, we can refind the exate index in kpoints which is up and which is down, by k @ spin = 0


    Note , this function work just in case k @ spin polarization = 0. So Dresselhaus's in plan spin texture can't use this function.
    
    ### TODO : finish this.
    """




















#############
############# 
############# convert the kpoints' coordinates between cartesian and ab basis of reciprocal lattice vectors





















##################
##################
##################
##################
################## the plots of the spin texture

def plot_bands_2D_with_spin(bands: np.ndarray, 
                            kpoints: np.ndarray, 
                            spin_vectors_each_band: np.ndarray = None, 
                            max_N_bands: int = 5, 
                            vector_density: int = 5,
                            elev=30, 
                            azim=45, 
                            xlabel = 'ka', 
                            ylabel = 'kb',
                            transparency = 0.5,
                            want_plot_spin = False,
                            scale_factor_spin = 1,
                            scale_energy = 2,
                            factor_zoom = 1,
                            reci_cell : np.array = np.identity(3),
                            grid_size = 100,
                            bands_color = None,
                            spins_color = None):
    """
    Plot the energy bands with respect to kpoints in 3D with spin polarization vectors.
    Works with both single band (N,) and multiple bands (N, m).

    Plot spin vector with color inverse to the band's color
    
    Parameters:
    -----------
    bands : np.ndarray
        Shape (N,) for a single band or (N, m) array for multiple bands
    kpoints : np.ndarray
        Shape (N, 2) array containing the k-points coordinates
    spin_vectors1 : np.ndarray
        Shape (N, 3) array containing the first group of spin polarization vectors
    spin_vectors2 : np.ndarray
        Shape (N, 3) array containing the second group of spin polarization vectors
    max_N_bands : int, optional
        Maximum number of bands to plot for visibility, default=5
    vector_density : int, optional
        Controls the density of vectors to display (higher = fewer vectors), default=5
    elev :
        Angle from plan to the view.
    azim :
        Angle from x-axis to the view.
    transparency : float, transparency of the bands, default=0.5
    want_plot_spin : bool. If you want to plot the spin vectors, default=False
    scale_factor : float > 0 Adjust this value to make vectors more/less visible
    grid_size : int, optional: grid of kpoints plotting, default=100
    bands_color : list of RGBA (between 0 to 1) colors, it's first dime should be the number of bands, default=None
    spins_color : list of RGBA (between 0 to 1) colors, it's first dime should be the number of bands, default=None

    if bands_color and spins_color are None, the color will be generated by the jet colormap on divide 0 and 1 by number of bands. spin_color will have the inverse color of bands_color.
    Returns:
        None
    """
    shape_data_kpoints = kpoints.shape
    
    # Check if we have a single band or multiple bands
    single_band = bands.ndim == 1
    n_bands = -1
    if single_band:
        # For a single band, we treat it differently without reshaping
        n_kpoints = bands.shape[0]
        n_bands = 1
        # Ensure we have the right number of k-points and band values
        assert n_kpoints == shape_data_kpoints[0], "bands and kpoints don't have the same number of points"
    else:
        # For multiple bands, we can use the shape directly
        shape_data_bands = bands.shape
        n_kpoints = shape_data_bands[0]
        n_bands = shape_data_bands[1]
        # Perform full assertions
        assert shape_data_bands[0] == shape_data_kpoints[0], "bands and kpoints don't have the same number of points"
        assert shape_data_bands[1] <= max_N_bands, "don't give too many bands to plot"
        assert shape_data_bands[1] > 0, "don't give too few bands"
    
    # Common assertions for both cases
    assert shape_data_kpoints[1] == 2, "kpoints should be in 2D"
    # assert spin_vectors1.shape[0] == shape_data_kpoints[0], "spin_vectors1 must have same number of points as kpoints"
    # assert spin_vectors2.shape[0] == shape_data_kpoints[0], "spin_vectors2 must have same number of points as kpoints"
    # assert spin_vectors1.shape[1] == 3, "spin_vectors1 must be 3D vectors"
    # assert spin_vectors2.shape[1] == 3, "spin_vectors2 must be 3D vectors"
    assert spin_vectors_each_band.shape[-1] == 3, "spin_vectors must be 3D vectors"
    
    # Create a figure with a 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the extent of k-points for plotting
    x_min, x_max = np.min(kpoints[:, 0]), np.max(kpoints[:, 0])
    y_min, y_max = np.min(kpoints[:, 1]), np.max(kpoints[:, 1])
    
    # Create a grid for interpolation
    
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Define distinct colors for each band
    if bands_color is None:
        distinct_colors = plt.cm.jet(np.linspace(0, 1, n_bands + 1))
    else :
        distinct_colors = bands_color

    # assert vector_density != 0," vector_density should be no zero"


    #### define a set of index of kpoints to plot the spin vectors with a decalage of vector_density on dim 0 and dim 1
    step_plot_spin_dim_0 = np.linalg.norm(reci_cell[0])/vector_density
    step_plot_spin_dim_1 = np.linalg.norm(reci_cell[1])/vector_density
    # plot_atol = 4e-3
    index_plot_spin = []
    # Define the grid points
    grid_points_0 = np.arange(np.min(kpoints[:, 0]), np.max(kpoints[:, 0]) + step_plot_spin_dim_0, step_plot_spin_dim_0)
    grid_points_1 = np.arange(np.min(kpoints[:, 1]), np.max(kpoints[:, 1]) + step_plot_spin_dim_1, step_plot_spin_dim_1)

    # For each grid point, find the closest k-point
    for x0 in grid_points_0:
        for x1 in grid_points_1:
            # Calculate distances to this grid point
            distances = np.sqrt((kpoints[:, 0] - x0)**2 + (kpoints[:, 1] - x1)**2)
            # Find the index of the closest point
            closest_index = np.argmin(distances)
            index_plot_spin.append(closest_index)

    index_plot_spin = np.array(index_plot_spin)
    
    # for i in range(vector_density):
    #     for j in range(vector_density):
    #         set_grill_plot_spin.append(i*int(grill_plot_spin) * j*int(grill_plot_spin))

    

    ## Create solid colors for each band
    for i in range(n_bands):
            # Interpolate the band energies onto the grid - handle single band case
        if single_band:
            band_values = bands
        else:
            band_values = bands[:, i]
            
        grid_z = griddata(kpoints, band_values, (Xi, Yi), method='cubic', fill_value=np.nan)
        
        # Get a single color for this band
        band_color = distinct_colors[i]
        
        
        # Plot the interpolated band as a surface with a solid color
        surf = ax.plot_surface(Xi, Yi, grid_z, 
                            color=band_color,
                            alpha=transparency,
                            linewidth=0,
                            antialiased=True)
        if want_plot_spin :
            # Plot spin polarization vectors for this band
            # We'll only plot vectors at a subset of points to avoid clutter
            if len(spin_vectors_each_band.shape) != 3: ### this is to avoid the out of range error if we have only one band so one set of spin.
                spins = spin_vectors_each_band
            else :
                spins = spin_vectors_each_band[:,i,:] ### the ith band's spin polarisation vectors

            

            for j in index_plot_spin:
                
                
                # Calculate the sum of the two spin vectors
                # spin_sum = spin_vectors1[j] + spin_vectors2[j]
                
                # # Skip points where spin vectors cancel out
                # if np.linalg.norm(spin_sum) <= 1e-5:
                #     continue
                    
                # Position vectors at the energy level of the band
                x, y = kpoints[j]
                # Handle z-coordinate for single or multiple bands
                if single_band:
                    z = bands[j]
                else:
                    z = bands[j , i]
                
                if spins_color is None:
                    spin_color = distinct_colors[-i]
                else :
                    spin_color = spins_color[i]
                u1, v1, w1 = scale_factor_spin * spins[j] / np.linalg.norm(spins[j])
                ax.quiver(x, y, z, u1, v1, w1, color= spin_color, length=0.2, arrow_length_ratio=0.3)
                

    # Set labels and title

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel('Energy', fontsize=14)
    ax.set_title('Energy Bands with Spin Polarization', fontsize=16)

    # Set an oblique view angle
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(x_min/factor_zoom, x_max/factor_zoom)  # Adjust these values to zoom in/out
    ax.set_ylim(y_min/factor_zoom, y_max/factor_zoom)  # Adjust these values to zoom in/out
    # ax.set_zlim(np.min(bands) / scale_energy, np.max(bands)/scale_energy)  # Adjust these values to zoom in/out

    z_min, z_max = np.min(bands)
    z_center = (z_max + z_min) / 2  # Midpoint of the energy range
    z_range = (z_max - z_min) / scale_energy / 2  # Scaled half-range

    ax.set_zlim(z_center - z_range, z_center + z_range)

    

    

    
    # Adjust the aspect ratio to make the visualization clearer
    ax.set_box_aspect([1, 1, 0.5])
    
    # Add a custom legend for bands
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = []
    for i in range(n_bands):
        legend_elements.append(Patch(facecolor=distinct_colors[i], alpha=0.7, label=f'Band {i+1}') )
        # Add legend elements for spin vectors
        legend_elements.append(Line2D([0], [0], color=spins_color[i], lw=2, label=f'Spin band {i+1}'))
    
    
    

    
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()

    return fig, ax



def zoom_in_bottom_global(bands_to_plot_init : np.ndarray, factor_energy = 3):
    """
    plot the bottom to range / factor_energy's plot
    new_max = bottom - (top - bottom) / factor_energy
    plot of energy above this new_max to new_max

    Args : 
        bands_to_plot_init : np.ndarray : shape (N,n) or (N,) where N represent the number of kpoint. n represent the number of band
    """
    #### plot the bottom to range / factor_energy's plot
    bands_to_plot =  bands_to_plot_init.copy()
    
    top = np.nanmax(bands_to_plot)
    bottom = np.nanmin(bands_to_plot)
    new_max = bottom + (top - bottom) / factor_energy ### to flatten the plot of energy above this new_max to new_max
    # indexs_3 = np.argwhere(bands_to_plot[:,0] > bottom + (top-bottom) /factor_energy).flatten()
    # index2_3 = np.argwhere(bands_to_plot[:,1] > bottom + (top-bottom) /factor_energy).flatten()
    # # bands_to_plot[indexs_3[:,0], index2_3[:,1]] = np.max(bands_to_plot) /3

    # ### flatten the value of > bottom + (top-bottom) /factor_energy
    # bands_to_plot[indexs_3, 0] = bottom + (top-bottom) /factor_energy
    # bands_to_plot[index2_3, 1] = bottom + (top-bottom) /factor_energy

    if len(np.shape(bands_to_plot_init)) == 2:
        for i in range(np.shape(bands_to_plot)[1]):
            index_above_new_max = np.argwhere(bands_to_plot[:,i] > new_max).flatten()
            bands_to_plot[index_above_new_max, i] = new_max
    else:
        index_above_new_max = np.argwhere(bands_to_plot > new_max).flatten()
        bands_to_plot[index_above_new_max] = new_max
    return bands_to_plot, new_max




def plot_bands_2D_with_spin_interac(bands: np.ndarray,
                                   kpoints: np.ndarray,
                                   spin_vectors_each_band: np.ndarray = None,
                                   max_N_bands: int = 5,
                                   vector_density: int = 5,
                                   xlabel='ka',
                                   ylabel='kb',
                                   transparency=0.5,
                                   want_plot_spin=False,
                                   scale_factor_spin=0.5,
                                   reci_cell: np.array = np.identity(3),
                                   grid_size=100,
                                   bands_color=None,
                                   spins_color=None,
                                   factor_energy = 1
                                    ):
    """
    Interactive version of plot_bands_2D_with_spin using Plotly
    """
    
    
    
    # Input validation (same as plot_bands_2D_with_spin)
    shape_data_kpoints = kpoints.shape
    single_band = bands.ndim == 1
    
    if single_band:
        n_kpoints = bands.shape[0]
        n_bands = 1
        assert n_kpoints == shape_data_kpoints[0]
    else:
        shape_data_bands = bands.shape
        n_kpoints = shape_data_bands[0]
        n_bands = shape_data_bands[1]
        assert shape_data_bands[0] == shape_data_kpoints[0]
        assert shape_data_bands[1] <= max_N_bands
    
    # Create grid for interpolation
    x_min, x_max = np.min(kpoints[:, 0]), np.max(kpoints[:, 0])
    y_min, y_max = np.min(kpoints[:, 1]), np.max(kpoints[:, 1])  # Fixed: added y_max calculation
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    

    # Initialize Plotly figure
    fig = go.Figure()

    

    if bands_color is None or spins_color is None:
        # Generate colors using jet colormap
        colors = plt.cm.jet(np.linspace(0, 1, n_bands))
        
        if bands_color is None:
            # Convert matplotlib colors to plotly format with transparency
            bands_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{transparency})' 
                         for c in colors]
            
        if spins_color is None:
            # Inverse colors for spins (use reversed color array)
            spins_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},1)' 
                         for c in colors[::-1]]
    else:
        # Convert user-provided RGBA colors to plotly format
        bands_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{transparency})' 
                      for c in bands_color]
        spins_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},1)' 
                      for c in spins_color]

    # For each band's surface, create a single color scale
    band_colorscales = [[[0, color], [1, color]] for color in bands_color]
    spin_colorscales = [[[0, color], [1, color]] for color in spins_color]

    # Calculate spin vector positions if needed
    if want_plot_spin:
        step_plot_spin_dim_0 = np.linalg.norm(reci_cell[0])/vector_density
        step_plot_spin_dim_1 = np.linalg.norm(reci_cell[1])/vector_density
        grid_points_0 = np.arange(x_min, x_max + step_plot_spin_dim_0, step_plot_spin_dim_0)
        grid_points_1 = np.arange(y_min, y_max + step_plot_spin_dim_1, step_plot_spin_dim_1)
        
        index_plot_spin = []
        for x0 in grid_points_0:
            for x1 in grid_points_1:
                distances = np.sqrt((kpoints[:, 0] - x0)**2 + (kpoints[:, 1] - x1)**2)
                index_plot_spin.append(np.argmin(distances))
        index_plot_spin = np.array(index_plot_spin)


    ### zoom into bottom by flatten the energy above the bottom + (top-bottom) /factor_energy
    bands, new_max_energy_plot = zoom_in_bottom_global(bands, factor_energy=factor_energy)

    # Add z_min, z_max calculation before the layout update - handle NaN values
    z_min, z_max = np.nanmin(bands), np.nanmax(bands)
    z_center = (z_max + z_min) / 2
    z_range = (z_max - z_min) / 2

    # Plot bands and spins
    for i in range(n_bands):
        # Get band values

        band_values = bands if single_band else bands[:, i]
        grid_z = griddata(kpoints, band_values, (Xi, Yi), method='linear')

        # Add band surface
        fig.add_trace(go.Surface(
            x=Xi, y=Yi, z=grid_z,
            colorscale=band_colorscales[i],
            showscale=False,
            showlegend=True,
            name=f'Band {i+1}',
            hovertemplate=f"Band {i+1}<br>" +
                         "x: %{x:.3f}<br>" +
                         "y: %{y:.3f}<br>" +
                         "z: %{z:.3f}"
        ))

        # Add spin vectors if requested
        if want_plot_spin:
            # Handle single band vs multiple bands case for spins
            if single_band:
                spins = spin_vectors_each_band
            else:
                # For multiple bands, each band i should use spin vectors from index i
                spins = spin_vectors_each_band[:,i,:]  # Get spins for current band directly
            
            x_arrows = []
            y_arrows = []
            z_arrows = []
            u_arrows = []
            v_arrows = []
            w_arrows = []
            norm_spin_polar = []
            
            for j in index_plot_spin:
                x, y = kpoints[j]
                z = bands[j] if single_band else bands[j, i]
                if z is not np.nan :
                    spin = spins[j]
                    
                    if np.linalg.norm(spin) > 1e-5 and z < new_max_energy_plot:  # Only plot if spin is non-zero
                        # Normalize and scale the spin vector
                        # spin_normalized = scale_factor_spin * spin / np.linalg.norm(spin)
                        # spin_normalized =  spin / np.linalg.norm(spin)
                        spin_normalized =  spin
                        x_arrows.append(float(x))
                        y_arrows.append(float(y))
                        z_arrows.append(float(z))
                        u_arrows.append(float(spin_normalized[0]))
                        v_arrows.append(float(spin_normalized[1]))
                        w_arrows.append(float(spin_normalized[2]))
                        norm_spin_polar.append(np.linalg.norm(spin))
            
            if x_arrows:  # Only add trace if there are vectors to plot
                # data_scale = max(x_max - x_min, y_max - y_min, z_max - z_min)
                

                fig.add_trace(go.Cone(
                    x=x_arrows,
                    y=y_arrows,
                    z=z_arrows,
                    u=u_arrows,
                    v=v_arrows,
                    w=w_arrows,
                    
                    colorscale=spin_colorscales[i],
                    showscale=False,
                    showlegend=True,
                    # sizemode="absolute",
                    sizemode='scaled', 
                    # sizeref=base_size,
                    sizeref=scale_factor_spin,
                    anchor="tail",
                    name=f'Spin {i+1}',
                    hovertemplate=f"Spin {i+1}<br>" +
                             "x: %{x:.3f}<br>" +
                             "y: %{y:.3f}<br>" +
                             "z: %{z:.3f}<br>" +
                             "u: %{u:.3f}<br>" +
                             "v: %{v:.3f}<br>" +
                             "w: %{w:.3f}<br>" +
                             "norm : %{norm:.3f}<br>"
                    ))

    # Update layout with better camera view and larger figure size
    fig.update_layout(
                        scene=dict(
                            xaxis_title=xlabel,
                            yaxis_title=ylabel,
                            zaxis_title='Energy',
                            xaxis=dict(range=[x_min, x_max]),
                            yaxis=dict(range=[y_min, y_max]),
                            zaxis=dict(range=[z_min, z_max]),
                            aspectratio=dict(x=1, y=1, z=0.5)
                        ),
                        title='Energy Bands with Spin Polarization (Interactive)',
                        showlegend=True,
                        width=1200,  # Increase width
                        height=700,  # Increase height
                        legend=dict(
                            yanchor="top",
                            x=0.9,
                            y=0.9,
                            xanchor="right",
                            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background
                            bordercolor="rgba(0,0,0,0.5)",
                            borderwidth=2,
                            font = dict(size=12))
                            )
    
    fig.update_legends()
    return fig



def plot_bands_diff(band1, band2, kpoints):
    """
    Plot difference band1 - band2 at all k points using Plotly surface plot
    
    Parameters :
    -----------
        band1, band2 : np.ndarray (N,)
        Energy values at each k-point
        kpoints : np.ndarray (N, 2)
        k-point coordinates in 2D
        
    Returns :
    --------
    plotly.graph_objects.Figure
        Interactive surface plot showing band energy difference
    """
    # Calculate band difference
    band_diff = band1 - band2

    # Create interpolation grid
    x_min, x_max = np.min(kpoints[:, 0]), np.max(kpoints[:, 0])
    y_min, y_max = np.min(kpoints[:, 1])
    grid_size = 100
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate band difference onto regular grid
    grid_z = griddata(kpoints, band_diff, (Xi, Yi), method='cubic')

    # Create surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=Xi, y=Yi, z=grid_z,
            colorscale='RdBu',  # Red-White-Blue colorscale
            showscale=True,
            hovertemplate="ka: %{x:.3f}<br>" +
                         "kb: %{y:.3f}<br>" +
                         "ΔE: %{z:.3f} eV<br><extra></extra>",
            colorbar=dict(
                title='Energy Difference (eV)',
                titleside='right'
            )
        )
    ])

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='ka',
            yaxis_title='kb', 
            zaxis_title='ΔE (eV)',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        title='Band Energy Difference',
        width=800,
        height=800
    )

    return fig





def reshape_energy(band : np.ndarray, index_max_kpoint : int, fill_empty_value = np.nan, dtype = None):
    """
    
    You need a array with both info : index of kpoint and energy value. But sometime the index of kpoint is not the index of the array. 
    usually you want to aligne this energy array with the kpoint array. This function will do this for you.
    So my solution is just fill the empty energy by a j. but the return array have the same shape 
    
    """
    assert len(band.shape) == 1, "band should be 1D array"
    # assert type(band[0]) == np.void, "band should be array of tuple who contain index of kpoint and energy value"
    # dtype_index = band.dtype['index']
    # dtype_energy = band.dtype['energy']
    if dtype is None:
        dtype = band.dtype
    
    n_kpoint = index_max_kpoint + 1
    # dtype = np.dtype([('index', dtype_index), ('energy', np.complexfloating)]) #### '<f8' is a good sub class of this complexfloating
    # dtype['energy'] = np.complexfloating 
    fill_value = np.array([(-1, fill_empty_value)], dtype=dtype)
    structured_array = np.full(n_kpoint, fill_value, dtype=dtype)

    for item in band:
        index_k = item[0]
        structured_array[index_k] = item
    
    return structured_array



































###################
###################
###################
###################
###################
###################
###################
################### some plot of spin texture

def calculate_cross_products(A_vectors: np.ndarray, angle_xy: float, norm_b: float = 1.0) -> np.ndarray:
    """
    Chatgpt code, just want to do some demo with Rashba or dresselhaus image illustration :
    How to find the effectif magnetic field of a Rashba or Dresselhaus system
    
    Calculate cross products between a list of vectors A and a vector b that lies in xy plane
    with specified angle and norm
    
    Parameters:
    -----------
    A_vectors : np.ndarray 
        Shape (N, 3) array of N vectors with equal norms
    angle_xy : float
        Angle in radians in xy plane (0 = along x axis)
    norm_b : float, optional
        Length of vector b (default = 1.0)
        
    Returns:
    --------
    np.ndarray : Shape (N, 3) array of cross products
    b_vector : np.ndarray (3,)
    """
    # Create b_vector in xy plane with given angle and norm
    b_vector = np.array([
        norm_b * np.cos(angle_xy),  # x component
        norm_b * np.sin(angle_xy),  # y component
        0.0                         # z component (in xy plane)
    ])
    
    # Verify A vectors have equal norm
    norms = np.linalg.norm(A_vectors, axis=1)
    assert np.allclose(norms, norms[0]), "All A vectors must have equal norm"
    
    # Calculate cross products
    return np.cross(A_vectors, b_vector), b_vector








def plot_vectors_3D(A_vectors: np.ndarray, 
                    b_vector: np.ndarray, 
                    cross_products: np.ndarray,
                    scale: float = 1.0, 
                    color_A: str = 'blue', 
                    color_b: str = 'red', 
                    color_cross: str = 'green', 
                    color_sum: str = 'purple',
                    A_vectors_label = 'E',
                    b_vector_label = 'P',
                    cross_products_label = 'B',
                    sum_label = 'B_sum') -> go.Figure:
    """
    Plot 3D vectors using plotly with specific color scheme:
    - All A vectors in one color
    - b vector in another color
    - All cross products in a third color
    - Sum of cross products in a fourth color
    
    Parameters :
    -----------
    A_vectors : np.ndarray
        Shape (N, 3) array of input vectors
    b_vector : np.ndarray
        Shape (3,) vector 
    cross_products : np.ndarray
        Shape (N, 3) array of cross products
    scale : float, optional
        Scaling factor for vector visualization
    color_A, color_b, color_cross, color_sum : str
        Colors for different vector groups
        
    Returns :
    --------
    go.Figure : Plotly figure object with arrows and labels for all vectors
    """
    fig = go.Figure()
    
    # Helper function to add vector with arrow and label
    def add_vector_with_label(start, vec, color, name, sizeref=0.2):
        # Ensure vec is normalized correctly
        vec_scaled = vec * scale
        end = start + vec_scaled
        
        # Add arrow shaft
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False
        ))
        
        # Add arrow head using the original vector direction
        fig.add_trace(go.Cone(
            x=[end[0]], y=[end[1]], z=[end[2]],
            u=[vec[0]], v=[vec[1]], w=[vec[2]],
            sizemode='scaled',
            sizeref=sizeref,
            anchor="tip",
            showscale=False,
            colorscale=[[0, color], [1, color]],
            name=name
        ))
        
        # Add label
        mid = (start + end) / 2
        offset = np.array([0.1, 0.1, 0.1])
        fig.add_trace(go.Scatter3d(
            x=[mid[0] + offset[0]],
            y=[mid[1] + offset[1]],
            z=[mid[2] + offset[2]],
            mode='text',
            text=[name],
            textposition="middle right",
            showlegend=False
        ))

    # Plot all A vectors
    for i in range(len(A_vectors)):
        add_vector_with_label(
            np.zeros(3), 
            A_vectors[i], 
            color_A, 
            f'{A_vectors_label}{i+1}'
        )
    
    # Plot b vector (ensure it's a single vector)
    if b_vector.ndim == 1:
        add_vector_with_label(
            np.zeros(3), 
            b_vector,
            color_b,
            f'{b_vector_label}'
        )
    
    # Plot all cross products
    for i in range(len(cross_products)):
        add_vector_with_label(
            np.zeros(3), 
            cross_products[i],
            color_cross,
            f'{cross_products_label}{i+1}'
        )
    
    # Calculate and plot sum of cross products
    sum_cross = np.sum(cross_products, axis=0)
    if np.linalg.norm(sum_cross) > 1e-10:
        add_vector_with_label(
            np.zeros(3), 
            sum_cross,
            color_sum,
            f'{sum_label}',
            sizeref=0.3
        )

    # Add origin point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=4, color='black'),
        name='Origin'
    ))
    
    # Update layout with improved visibility
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            annotations=[
                dict(
                    showarrow=False,
                    x=0, y=0, z=0,
                    text="O",
                    xanchor="left",
                    yanchor="bottom"
                )
            ]
        ),
        title='3D Vector Visualization',
        showlegend=True,
        width=800,
        height=800
    )
    
    return fig

# Example usage:
# A_vectors = np.array([[1,0,0], [0,1,0]])
# b_vector = np.array([0,0,1])
# cross_products = calculate_cross_products(A_vectors, angle_xy=0)
# fig = plot_vectors_3D(A_vectors, b_vector, cross_products)
# fig.show()




def plot_spin_xy_projection(kpoints, 
                          spin_vectors_each_band,
                          bands: np.ndarray,  # Added bands parameter
                          target_energy: float,  # Added energy filter
                          energy_tolerance: float = 0.05,  # Added tolerance parameter 
                          vector_density=200,
                          spins_color=None,
                          bands_color = None,
                          scale_factor_spin=1.0,
                          transparency = 0.7,
                          xlabel = 'ka',
                          ylabel = 'kb'):
    """
    Plot 2D spin projection at a specific energy level
    
    Args:
        kpoints: Points k (N,2)
        spin_vectors_each_band: Vecteurs de spin (N,n_bands,3) 
        bands: Band energies (N,n_bands) array
        target_energy: Energy level to plot spins at
        energy_tolerance: Energy window around target_energy
        vector_density: Density of vectors to display
        spins_color: Colors for each band's spins
        scale_factor: Scale factor for cone sizes
    """
    fig = go.Figure()
    
    x_min, x_max = np.min(kpoints[:,0]), np.max(kpoints[:,0])
    y_min, y_max = np.min(kpoints[:,1]), np.max(kpoints[:,1])
    z_min, z_max =np.min(bands[:,1]), np.max(bands[:,1])
    

    n_bands = spin_vectors_each_band.shape[1] if len(spin_vectors_each_band.shape) > 2 else 1
    

    


    


    

    # Generate colors if not provided
    if bands_color is None or spins_color is None:
        # Generate colors using jet colormap
        colors = plt.cm.jet(np.linspace(0, 1, n_bands))
        
        if bands_color is None:
            # Convert matplotlib colors to plotly format with transparency
            bands_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{transparency})' 
                         for c in colors]
        else :
            bands_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{transparency})' 
                      for c in bands_color]
            
        if spins_color is None:
            # Inverse colors for spins (use reversed color array)
            spins_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},1)' 
                         for c in colors[::-1]]
        else :
            spins_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},1)' 
                      for c in spins_color]

    else:
        # Convert user-provided RGBA colors to plotly format
        bands_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{transparency})' 
                      for c in bands_color]
        spins_color = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},1)' 
                      for c in spins_color]
    # For each band's surface, create a single color scale
    band_colorscales = [[[0, color], [1, color]] for color in bands_color]
    spin_colorscales = [[[0, color], [1, color]] for color in spins_color]
    # Plot spins for each band

    for band_idx in range(n_bands):
        # Get points near target energy
        grid_points = []
        band = bands[:,band_idx]
        for idx, kpt in enumerate(kpoints):
            
            
            # Check if any band at this k-point is within energy window
            if not np.isnan(band[idx]):
                if np.abs(band[idx] - target_energy) <= energy_tolerance:
                    grid_points.append(idx)
        x_arrows = []
        y_arrows = []
        u_arrows = []
        v_arrows = []
        w_arrows = []
        z_arrows = np.ones((len(grid_points),)) * target_energy
        z_arrows = z_arrows.tolist()
        for idx in grid_points:
            spin = spin_vectors_each_band[idx, band_idx]
            kpoint = kpoints[idx]
            if np.linalg.norm(spin) > 1e-5:
                        x_arrows.append(float(kpoint[0]))
                        y_arrows.append(float(kpoint[1]))
                        u_arrows.append(float(spin[0]))
                        v_arrows.append(float(spin[1]))
                        w_arrows.append(float(spin[2]))


        
        fig.add_trace(go.Cone(
                x=x_arrows,
                y=y_arrows,
                z=z_arrows,
                u=u_arrows,
                v=v_arrows,
                w=w_arrows,
                
                colorscale=spin_colorscales[band_idx],
                showscale=False,
                showlegend=True,
                # sizemode="absolute",
                sizemode='scaled', 
                # sizeref=base_size,
                sizeref=scale_factor_spin,
                anchor="tail",
                name=f'Spin {band_idx+1}',
                hovertemplate=f"Spin {band_idx+1}<br>" +
                            "x: %{x:.3f}<br>" +
                            "y: %{y:.3f}<br>" +
                            "u: %{u:.3f}<br>" +
                            "v: %{v:.3f}<br>" +
                            "w: %{w:.3f}<br>" +
                            "norm : %{norm:.3f}<br>"
                ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Spin polarisation at E = {target_energy:.3f} eV",
            y=0.95,  # Position at 95% from bottom
            x=0.5,   # Centered
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
                            xaxis_title=xlabel,
                            yaxis_title=ylabel,
                            zaxis_title='Energy',
                            # zaxis_title='Energy',
                            xaxis=dict(range=[x_min, x_max]),
                            yaxis=dict(range=[y_min, y_max]),
                            zaxis=dict(range=[z_min, z_max]),
                            aspectratio=dict(x=1, y=1, z=0.5)
                        ),
                        
        showlegend=True,
        width=1200,  # Increase width
        height=700,  # Increase height
        legend=dict(
            yanchor="top",
            x=0.9,
            y=0.9,
            xanchor="right",
            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background
            bordercolor="rgba(0,0,0,0.5)",
            borderwidth=2,
            font = dict(size=12))
                            )
    

    return fig



######
######
###### finally, i want to plot bands (in high symmetry path) and spin in (kx, ky) plan.

def get_contour_points_from_hull(kpoints, tolerance=1e-6):
    """
    Extract all points lying on the convex hull boundary (including edges)
    
    Parameters:
    -----------
   
    kpoints : array of all k-points
    tolerance : numerical tolerance for point inclusion
    
    Returns:
    --------
    contour_points : array of points on the hull boundary
    contour_indices : indices of contour points in original array
    """

    # Get the Hull object
    from scipy.spatial import ConvexHull
    hull = ConvexHull(kpoints[:, :2])  # Only need 2D for convex hull and kpoints are in 2D plan.

    # Get the vertices (simplices)
    vertices = kpoints[hull.vertices]
    
    # For 2D case, we need to get points along the edges
    # First approach: Use hull equations to find points on boundaries
    contour_indices = set(hull.vertices)
    
    # Check each point to see if it lies on any hull facet
    for i, point in enumerate(kpoints):
        if i in hull.vertices:
            continue  # Already included as vertex
            
        # Check if point lies on any facet (within tolerance)
        for j, facet in enumerate(hull.simplices):
            # Get the two vertices defining this edge
            v1, v2 = kpoints[facet[0]], kpoints[facet[1]]
            
            # Check if point lies on the line segment between v1 and v2
            if point_on_line_segment(point, v1, v2, tolerance):
                contour_indices.add(i)
                break
    
    contour_indices = sorted(contour_indices)
    contour_points = kpoints[contour_indices]
    
    return contour_points, contour_indices, vertices






def point_on_line_segment(point, v1, v2, tolerance=1e-6):
    """
    Check if a point lies on the line segment between v1 and v2

    Return : Bool
    """
    # Vector from v1 to point and v1 to v2
    v1p = point - v1
    v12 = v2 - v1
    
    # Check if point is collinear with v1 and v2
    cross_product = np.linalg.norm(np.cross(v1p, v12))
    if cross_product > tolerance:
        return False
    
    # Check if point is between v1 and v2
    dot_product = np.dot(v1p, v12)
    if dot_product < -tolerance or dot_product > np.dot(v12, v12) + tolerance:
        return False
    
    return True



































#################
#################
#################
#################
################# Find some local minimum or maximum in the band structure. Then analyse the SOC type

##### TODO : finish this part of band analysis.


def sort_Ek_in_2D_plan(E_k):
    """
    sort the kpoints in a 2D plan so 3D array.

    Note:
    -----
        we can assume that the kx and ky form a grid. Where dx and dy are 2 constant.
        We don't need to assume that the x and y are orthogonal.

    Args :
    -------
        E_k (3D array): 3D array of energy values at each kpoint.
    Returns :
    -------
        table (3D array): 2 first dimension of sorted kpoint, 3rd dimension is [kx, ky, E]. 
                            Line with same kx, column with same ky.
    """
    kx = E_k[:, 0]
    ky = E_k[:, 1]
    E = E_k[:, 2]

    ind_sort_1st_kx_2nd_ky = np.lexsort((ky, kx)) # Sort by kx first, then ky : ((sequence kx constant),...)
    previous_kx = None
    table = []
    row = []
    previous_kx = kx[ind_sort_1st_kx_2nd_ky[0]]
    for index in range(len(ind_sort_1st_kx_2nd_ky)):
        i = ind_sort_1st_kx_2nd_ky[index]
        if previous_kx is None :
            row.append([kx[i],ky[i],E[i]])
            
        elif  previous_kx is not None and kx[ind_sort_1st_kx_2nd_ky[index]] == previous_kx:
            row.append([kx[i],ky[i],E[i]])
        else:
            table.append(row)     
            previous_kx = kx[i]  
            
            row = [] 
            row.append([kx[i],ky[i],E[i]])
        if index == len(ind_sort_1st_kx_2nd_ky) - 1:
            table.append(row)
    try:
        return np.array(table)
    except:
        print("Error in sorting E_k, probably due to non homogeneous kx or ky length.")
        return None



def calculate_gradient(E_k):
    """
    
    ### This function is deprecated.



    for a each kpoint in the result of funciton sort_Ek_in_2D_plan, find the 4 neighbours up, down, left, right.
    Then calculate the gradient of the energy in the kx, ky direction.

    Hypothesis :
    ------------
    Same as the hypothesis in sort_Ek_in_2D_plan.

    Methode:
    -------
    Like the hypothesis in sort_Ek_in_2D_plan, we can assume that the kx and ky form a grid. Where dx and dy are 2 constant.
    The gradient is calculated by dE/dkx * ex + dE/dky * ey. Represent by a vector of 2 components.
    
    dE/dkx(or y) = (dE/dkx (or y) (up, or left) + dE/dkx (down, right)) / 2, 
    if dE/dkx (up), dE/dkx (down) are close to each other. Tolerance absolute between them is 0.05.

    Here we reused the idea of existance of derivative. The limite from left and right is the same. 
    Else, if the point is at the edge of a zone np.nan, the gradient is np.nan. 
    If it is at the edge of the global zone, the gradient is not nan. but a direction (in x or y) where the definition of derivative existe
    #### !!!And at the direction (x, or  y) where the derivative does not exist, the gradient is the dE/dk(x or y) at a side where we have data * j. 
    This is to remind that at the edge, the derivative does not exist i a x or y direction but we calculate the value at one side.
    
        
    

    Args :
    ----
        E_k (3D array): output array of sort_Ek_in_2D_plan.

    Returns :
    -------
    """
    shape = E_k.shape
    index_max_x = shape[0]
    index_max_y = shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i != 0 and j != 0 and i != index_max_x - 1 and j != index_max_y - 1:
                # Calculate the gradient using the finite difference method
                # dE_dkx = (E_k[i+1, j, 2] - E_k[i-1, j, 2]) / (2 * (E_k[i+1, j, 0] - E_k[i-1, j, 0]))
                # dE_dky = (E_k[i, j+1, 2] - E_k[i, j-1, 2]) / (2 * (E_k[i, j+1, 1] - E_k[i, j-1, 1]))
                dE_dkx_left = (E_k[i, j, 2] - E_k[i -1, j, 2]) / (E_k[i, j, 0] - E_k[i - 1, j, 0])
                dE_dkx_right = (E_k[i, j, 2] - E_k[i+1, j, 2]) / (E_k[i, j, 0] - E_k[i+1, j, 0])
    return 



def calculate_gradient_vNumpy(E_k, edge_order=2):
    """
    from the output of sort_Ek_in_2D_plan, calculate the gradient of the energy with numpy.gradient function.

    Assumsion :
    -----------
        same as sort_Ek_in_2D_plan.
    Args :
    ----
        E_k (3D array): output array of sort_Ek_in_2D_plan.
    
    """
    energy = E_k[:, :, 2]
    dx = abs(E_k[0, 0, 0] - E_k[1, 0, 0])
    dy = abs(E_k[0, 0, 1] - E_k[0, 1, 1])
    gradient = np.gradient(energy, dx, dy, edge_order=edge_order)

    gradient_vecs = np.ones((gradient[0].shape[0], gradient[0].shape[1], 2))
    for i in range(gradient[0].shape[0]):
        for j in range(gradient[0].shape[1]):
            gradient_vecs[i, j, 0] = gradient[0][i, j]
            gradient_vecs[i, j, 1] = gradient[1][i, j]

    return gradient_vecs


def check_gradient_around_n_time_radiaus(E_k, n = 4):
    pass



def check_local_extremum(E_k):
    """

    Args:
    --------
        E_k : np.ndarray : shape (N,3) where N is the number of kpoints and 3 component are recpectively ka, kb, energy.
    Returns:
    --------
        Extremums : np.ndarray : shape (N,3) where N is the number of local extremum and 3 component are recpectively ka, kb, energy.
    Note : 
    -----------
        The E_k can have energy np.nan.
    """
    x = E_k[:, 0]
    y = E_k[:, 1]
    f = E_k[:, 2]

    df_dx = np.gradient(f, x, edge_order=2)
    
    

def spin_splitting_type(E_k):
    """
    To recognize how the energy split to each other

    Args :
    --------
        E_k : np.ndarray : shape (N,3) where N is the number of kpoints and 3 component are recpectively ka, kb, energy.

        
    Returns :
    --------

    Note: 
    -------
        The E_k can have energy nan.
        
    """
    
    






def type_SOC_from_bands(bands, target_energy, spins_texture):
    """
    Function to determine the type of SOC from the bands
    """
    pass































#####
#####
#####
##### read spin files from QE calculation. Like *.out files and spin.* files




def index_VB_assum_no_metal(lines : list[str],
                            index : int = 0,
                            ):
    """
    After nscf calculation, a nscf.out file contain the occupation of each band at each k point. We just take one to assume the T = 0k and non metal to count the number of bands occupied.

    Args:
    -----
        lines : str : all lines of nscf.out file
        index : int : index of line in lines currently reading in read_nscf_out_file function()

    Note :
    ------
        Here we search the occupation data in fact for the 1st k point (Normaly Gamma point) in the nscf output file.
        But in fact the order of occupation should be different in different k point. Because the band can cross each other.
    TODO : Follow the note, apply the correct logic to this function
    """
    line = lines[index]
    if line.find("occupation numbers") != -1:
        i = 1
        while lines[index + i].find('k') == -1 and index + i < len(lines) -1: ## occupations numbers are between 2 lines 'occupation numbers' and 'k = ...'
            i += 1
        
        occupations_data = []
        for j in range(1,i ):
            line_occup = lines[index + j]
            floats_in_line = re.findall(r'[-+]?\d*\.\d+|\d+', line_occup) ### rule of float
            occupations_data.extend(floats_in_line)
        
        occupations_data = np.array(occupations_data, dtype=float)
        nbnd_occupied = np.sum(occupations_data > 0.) ### count the number of bands occupied. occupation == 1 cause we use 'occupation == fixed' in QE
        jobs_done = True
        return nbnd_occupied, jobs_done
    else:
        return -1, False
    
# def sort_occupation_number():
    
###### some function to read the nscf.out file from nscf calculation.

def refind_path_from_k_point_weight_from_nscf_out(file_nscf_out : str):
    """
    
    
    This is to find the k point symmetry. And trace a band path with it.
    I dont like this idea. I just keep this idea if i will use it some day. (03/11/25)
    
    
    """
    


def read_G_vectors(lines, index):
    """
    in nscf.out file, read the reciprocal lattice vectors after line "reciprocal axes: (cart. coord. in units 2 pi/alat)"
    
    """
    G_vectors = [[]]
    if lines[index].find("reciprocal axes") != -1:
        for i in range(3):
            line = lines[index + i + 1]
            floats_in_line = re.findall(r'[-+]?\d*\.\d+', line) ### rule of float
            G_vectors.append(floats_in_line)
        G_vectors.pop(0)
        return np.array(G_vectors, dtype=float), True
    else:
        return None, False

    
def read_R_vectors(lines, index):
    """
    in nscf.out file, read the direct lattice vectors after line "crystal axes: (cart. coord. in units of alat)"
    """
    R_vectors = [[]]
    if lines[index].find("crystal axes") != -1:
        for i in range(3):
            line = lines[index + i + 1]
            floats_in_line = re.findall(r'[-+]?\d*\.\d+', line) ### rule of float
            R_vectors.append(floats_in_line)
        R_vectors.pop(0)
        return np.array(R_vectors, dtype=float), True
    else:
        return None, False

def read_lattice_para(lines, index):
    """
    in nscf.out file, read the lattice vector in line ' lattice parameter (alat)  =       **'
    
    usually, unit is a.u. which means bohr (bohr radius)

    Returns :
    --------
        number_unit : tuple : (number, unit), number is float, unit is str. if not found, number_unit is -1.0
        jobs_done : bool : True if the line is found, False otherwise.
    
    """
    if lines[index].find("lattice parameter (alat)") != -1:
        number_unit = re.findall(r'=\s*([-+]?\d*\.?\d+)\s+(\S+)', lines[index])
        lattice_vec_num = float(number_unit[0][0])
        number_unit = (lattice_vec_num, number_unit[0][1])
        return number_unit, True
    else:
        return -1.0, False

def read_nscf_out_file(path, _file_='aiida.out'):
    """
    After nscf calculation, a nscf.out file is given.

    This function is to containt function who reads information that you want to extract from nscf.out file. 
    Function should be add in the iteration of lines. 

    Args:
    -----
        path (str): Path to the directory containing scf.out file
    
    """
    with open(path+f'/{_file_}', 'r') as f:
            
            lines = f.readlines()
    ## Boolean to indicate if the job (function) is done (at least one time, but it depend if they are completed).
    job_done_index_VB = False
    # n_bands_occupied = -1
    job_read_G_vectors = False
    job_read_R_vectors = False
    job_read_lattice_para = False
    for index, line in enumerate(lines):

        


        # if not job_done_index_VB : ## just do it one time.
        #     n_bands_occupied, job_done_index_VB = index_VB_assum_no_metal(lines, index)
        
        if not job_read_G_vectors : ## just do it one time.
            G_vectors, job_read_G_vectors = read_G_vectors(lines, index)
        if not job_read_R_vectors : ## just do it one time.
            R_vectors, job_read_R_vectors = read_R_vectors(lines, index)
        if not job_read_lattice_para :
            lattice_vec, job_read_lattice_para = read_lattice_para(lines, index)

    return G_vectors, R_vectors, lattice_vec






def parse_spins(path, list_bands, _file_='spin'):
    """
    Parse spin components from QE calculation output files (spin.1, spin.2, spin.3).

    Args:
    -----
        path (str): Path to the directory containing spin.* files
        list_bands (list): List of band indices (0-based) to extract, e.g. [0,1,3] for first, second and fourth bands

    Returns:
    --------
        spins (np.ndarray): Array of shape (n_selected_bands, n_kpoints, 3) containing spin components

    Note:
    --------
        since spin and spin.* files have similar format. The structures of code in parse_spin and parse_bands are similar.

    """

    for isp in range(3):

        file = _file_ + '.' +str(isp+1)
        with open(path+'/'+file, 'r') as f:
            lines = f.readlines()

        #dirty fix for a dirty bug:
        if lines == []:
            blah = [x for x in os.listdir(path) if 'fort' in x]
            file = blah[0]
            with open(path+'/'+file, 'r') as f:
                lines = f.readlines()

        if isp == 0:
            nbnd, nks = [int(x) for x in re.findall(r'\d+', lines[0])]
            spins = np.zeros(( len(list_bands),nks,3))

        nlk = int((len(lines)-1)/nks) # number of lines for each k point: (lines in the file minus header)/number of kpoints
        nbl = len(lines[2].split())       # number of bnds per line


        for ib, ibnd in enumerate(list_bands):
            il, ip = math.ceil((ibnd+1)/nbl), (ibnd+1)%nbl-1  # line number and position in line (arrondi supérieur). ibnd+1 becasue bands in python indices
            for ik in range(nks):
                spins[ib,ik,isp] = float(lines[nlk*ik+il+1].split()[ip])
    return spins



def parse_high_symmetry_path(path, file='aiida.out'):
    """
    Parse high-symmetry points from a file.
    
    Expected line format:
    'high-symmetry point:  0.0000 0.0000 0.0000   x coordinate   0.0000'
    
    Parameters:
    -----------
    filename : str
        Path to the file containing high-symmetry points
    
    Returns:
    --------
    dict : {float: np.array}
        Dictionary mapping x-coordinates to high-symmetry points as numpy arrays
    """
    filename = path + '/' + file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to match the line
    # Captures: three k-point coordinates and x-coordinate
    pattern = pattern = r'high-symmetry point:\s+([-+]?\d+\.?\d*)(?:\s|(?=-))+([-+]?\d+\.?\d*)(?:\s|(?=-))+([-+]?\d+\.?\d*)\s+x coordinate\s+([-+]?\d+\.?\d*)'
    
    matches = re.findall(pattern, content)
    
    # Build dictionary
    result = {}
    for match in matches:
        kx, ky, kz, x_coord = match
        k_point = np.array([float(kx), float(ky), float(kz)])
        x_coord_val = float(x_coord)
        result[x_coord_val] = k_point
    
    return result




def parse_bands(path, file='spin'):
    """
    extract band energies from QE calculation output files (spin). Its name is spin but this file contain only bands information, just because it's the result of spin texture calculation from bands. 
    To extract the spins, you need to read spin.1, spin.2, spin.3 files with parse_spins function.

    Args:
    -----
        path (str): Path to the directory containing spin file
    """
    # file = 'spin'
    with open(path+'/'+file, 'r') as f:
        lines = f.readlines()

    #dirty fix for a dirty bug:
    if lines == []:
        blah = [x for x in os.listdir(path) if 'fort' in x]
        file = blah[0]
        with open(path+'/'+file, 'r') as f:
            lines = f.readlines()

    ### in spin file, the lines are like this :
    ###     kx, ky, kz
    ###  band1, band2, band3, band4
    ###  band5, band6, band7, band8
    ###  ...
    ###  same for spins' moments in files spin.1, spin.2, spin.3
    ###  So we need to know how many lines for each kpoint, and how many bands per line. 
    ### Code below are to read them.
    
    
    nbnd, nks = [int(x) for x in re.findall(r'\d+', lines[0])] ## the first line contain nbnd and nks with intergers. Other str are not intergers.

    nlk = int((len(lines)-1)/nks) # number of lines for each k point: (lines in the file minus header)/number of kpoints
    nbl = len(lines[2].split())       # number of bnds per line

    kpoints = []

    for line in lines:
        floats_in_line = re.findall(r'[-+]?\d+\.\d+|\d+', line) ### rule of float
        
        if len(floats_in_line) == 3: ### if there are just 3 floats in the line, we have a kpoint
            # This line contains k-point coordinates
            k = np.array(floats_in_line, dtype=float)
            # You can store or process the k-point coordinates as needed
            kpoints.append(k)
    kpoints = np.array(kpoints)

    if len(kpoints) != nks:
        print("Warning: number of kpoints read does not match header info.")
        print(f"Header says {nks}, but found {len(kpoints)} kpoints.")

    

    bands = np.zeros((nbnd, nks))
    

    # bands[ib, ik] : ib band index, ik kpoint index , energy.
    for ib, ibnd in enumerate(range(nbnd)):
        # il,ip = int((ibnd+1)/nbl)+1, (ibnd+1)%nbl-1  # line number and position in line (bug). ibnd+1 becasue bands in python indices
        il, ip = math.ceil((ibnd+1)/nbl), (ibnd+1)%nbl-1  # line number and position in line (arrondi supérieur). ibnd+1 becasue bands in python indices
        
        for ik in range(nks):
                bands[ib,ik] = float(lines[nlk*ik+il+1].split()[ip]) ## in fact, lines index should be nlk*ik (number of lines) + 2 (head line and one line kpoint) -1 (python index) + il (line number for band in a kpoint)


    # kpoints array
    
    
    return bands, kpoints


def kpoints_fix_v1(kpoints):
    """
    The QE generated kpoints have a probleme. A point (0, kimax) that i expect is at (0, -kimax). 
    So this (0, -kimax) is not at the irreducible Brillouin zone that other kpoints in. It's obvious if we plot the kpoints.

    In many of my case, the material haven't inversion symmetry. I can't just inverse the kpoint and use the same band energy (each band are not spin degenerate).
    
    Solution : So i will just remove this point out of the irrBZ before i get more information from the QE or my adviser.

    This function must be done before the get_contour_points_from_hull function.



    Args:
    -----
        kpoints (np.ndarray): shape (N,2) array of kpoints.

    Returns:
    --------
        kpoints_fixed (np.ndarray): shape (N-1,2) array of kpoints without the point (0, -kimax).
    """

    kymax = np.max(kpoints[:,1])
    kxmax = np.max(kpoints[:,0])
    kymin = np.min(kpoints[:,1])
    kxmin = np.min(kpoints[:,0])

    if kymax * kymin < 0:
        index_to_remove = np.where((kpoints[:,0] == 0) & (kpoints[:,1] == kymin))[0]
    elif kxmax * kxmin < 0:
        index_to_remove = np.where((kpoints[:,1] == 0) & (kpoints[:,0] == kxmin))[0]
    elif kxmax * kxmin < 0 and kymax * kymin < 0:
        index_to_remove = np.where((kpoints[:,0] == kxmin) & (kpoints[:,1] == kymin))[0]
    
    kpoints_fixed = np.delete(kpoints, index_to_remove, axis=0)
    return kpoints_fixed, index_to_remove


def path_segment_between_high_S_kpoint(start : np.ndarray, stop : np.ndarray, contour: np.ndarray, tolerance=1e-4):
    """
    for each point in contour obtained by get_contour_points_from_hull function, find the path (kpoints in contour) between 2 givened high symmetry kpoints.
    """

    path_segment = [[]]

    for k in contour:
        if point_on_line_segment(k, start, stop, tolerance=tolerance):
            path_segment.append(k.tolist())
    path_segment.pop(0)
    return path_segment



def plot_path_without_spin(energies_path : np.ndarray, 
                        x_coordinates : np.ndarray,
                        n_bands_occupied : int,
                        e_min_ev = -5, ## actually any unit you want. Cause this values is calculate by you
                        e_max_ev = 0,
                        special_points : dict = None, ## dict of special points in x_coordinates {x_coordinate : special point label}
                        k_min = None,
                        k_max = None,
                        index_band_min = 0,
                        index_band_max = None,
                        xlabel = 'x k-point along path',
                        ylabel = 'Energy (eV)',
                        title = 'Band structure',
                        path_save_fig = None
                        ):
    """
    energies_path : np.ndarray : shape (N, nbands) array of (bands) energies along a path of kpoints contain N point.
    x_coordinate : np.ndarray : shape (N,)
    """
    plt.figure()
    if index_band_max is None:
        index_band_max = energies_path.shape[1] - 1
    for i in range(index_band_min, index_band_max + 1):
        
        if i == n_bands_occupied -1:
            plt.plot(x_coordinates, energies_path[:,i], color='r', alpha=0.5, label='VB')
            
        elif i == n_bands_occupied:
            plt.plot(x_coordinates, energies_path[:,i], color='b', alpha=0.5, label='CB')
        else:
            plt.plot(x_coordinates, energies_path[:,i], color = 'gray', alpha=0.5)
    

    if special_points is not None:
        ticks = []
        label = []
        for x_val, kpoint_label in special_points.items():
            plt.axvline(x=x_val, color='k', linestyle='--', alpha=0.7)
            ticks.append(x_val)
            label.append(kpoint_label)
        plt.xticks(ticks= ticks, \
                labels= label)
            
    plt.ylim(e_min_ev, e_max_ev)
    plt.xlim(k_min, k_max)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if path_save_fig is not None:
        ## need to check if the save_path's directory exist, if not create it
        if not os.path.exists(os.path.dirname(path_save_fig)):
            os.makedirs(os.path.dirname(path_save_fig), exist_ok=True)
        plt.savefig(path_save_fig)
    plt.show()
    
    plt.close()



# def from_ksirrBZ_to_conventional_ksirrBZ(kpoints_irrBZ : np.ndarray, structure : StructureData):
#     """
#     Problem of QE : the generate kpoints in irrBZ is rarely what enveloppe by somme generated k path's irrBZ. like Y is [0.5, 0, 0] but the QE's could be rotated by 120.
    
#     A super stupid problem created by QE is that those "k point in irrBZ" are not exactly in the same irrBZ. So a contour of theses point is not the path.
    
#     So, this function is to refind the kpoint in a generated k path's contour (also a irrBZ). For exemple, generate by get_kpoints_path in aiida module.
    
#     With this correspondance, i can then find the path easily with the result of this function

#     Args:
#     ------
#         kpoint_irrBZ : list of (3,) np.array:  the kpoints extrated by bands.x calculation from the scf or nscf pw.x calculation of structure
#         structure :  the structure of type StructureData that you interste in.

#     """

#     ase_objet = structure.get_ase()
#     primitive_cell = ase_objet.cell ### it's not necessary standard cell. On of the concequence is that a and b both doesnt aligned to x and y.
#     convention_ksirrBZ_contour = get_explicit_kpoints_path(cell = primitive_cell, pbc= [True, True, False], cartesian=True)[3]
#     polygon_conv_irrBZ = Polygon(convention_ksirrBZ_contour)

#     print(polygon_conv_irrBZ)

#     spglib_symmetry = ase.spacegroup.symmetrize.check_symmetry(ase_objet)

#     scaled_rotations = spglib_symmetry.rotations

#     primitive_cell_rotations = [[]]

#     for rotation in scaled_rotations : ### just notice, there a
#         cart_rotation = change_basis(rotation, np.linalg.inv(primitive_cell.T)) ### calculate rotaiton matrix in primitive cell's basis give attentioin to tha transform matrix definition
#         cart_rotation = cart_rotation[:-1, :-1]
#         primitive_cell_rotations.append(cart_rotation)
    
#     primitive_cell_rotations.pop(0)
#     print(primitive_cell_rotations)

#     link_k_kroted = {}
#     for index , k in enumerate(kpoints_irrBZ):
#         if (k != np.array([0,0,0])).any(): ### what i suppose is that the polygon.contains methode dont work well with O point
#             inside = polygon_conv_irrBZ.contains(Point(k)) ### k point rotated stricly inside the irrBZ generated
#             at_the_edge = polygon_conv_irrBZ.covers(Point(k)) ### k point rotated stricly on the edge of irrBZ generated.
            
#             if not inside and not at_the_edge:
#                 print(f'point {k} in processing')
#                 k_rotated_in_conv_irrBZ = False
#                 # k_rotated = k
#                 while not k_rotated_in_conv_irrBZ:
#                     ### pick randomly a rotation to rotatd the k until it's in the conventionnal irrBZ
#                     index_random = np.random.randint(low=1, high=len(primitive_cell_rotations))
#                     rot = primitive_cell_rotations[index_random]
#                     k_rotated = rot @ k
#                     ### a boolean of k in the polygon formed by generated k path. It's always convex.
#                     inside = polygon_conv_irrBZ.contains(Point(k_rotated)) ### k point rotated stricly inside the irrBZ generated
#                     at_the_edge = polygon_conv_irrBZ.covers(Point(k_rotated)) ### k point rotated stricly on the edge of irrBZ generated.
#                     print(f'point {k} rotated to {k_rotated}, {inside}, {at_the_edge}')
#                     if  inside or at_the_edge:
#                         k_rotated_in_conv_irrBZ = True
#                         link_k_kroted[index] = [k, k_rotated]
#                         # print(f'point {k} rotated in generated irrBZ')
#             # for rot in primitive_cell_rotations :
#             #     k_rotated = rot @ k_rotated
#             #     if 
#         else :
#             pass

#     return link_k_kroted


def fullfill_1stBZ_from_randomirrBZ(ks_irrBZ, structure : StructureData): 
    """
    Since we have QE generated kpoints in irrBZ, we want to get the full 1st BZ kpoints from those kpoints in irrBZ.
    This will let we to easily find the k path and its energy with API like get_kpoints_path in aiida module. 
    Cause those API always generate k path from structure. But not from a given kpoints in irrBZ.
    So i want to fullfill the 1st BZ from those kpoints in irrBZ, and then get the k path with energies.


    Note :
    ------
        In fact, this function fullfill a half of the BZ because the time reversal symmetry tell me another can only be fullfill by a spin opposite band.
        If there is not time reversal symmetry, this funciton give a full 1st BZ.
        Since this function act on only kpoints, the functions deail with energy are bands_full_1st_BZ and fullfill_bands_with_time_reversal_symmetry.

    Args:
    -----
        ks_irrBZ (np.ndarray): shape (N,3) array of kpoints in irrBZ. Even in case that I work in materials 2D, the z component must be given and must be 0
        structure (StructureData): the structure of type StructureData that you interste in.

    Returns:
    -----
        k_full_1st_BZ : np.ndarray: k points of HALF of 1st BZ if there is a time reversal symmetry. If there isn't time reversal symmetry, it should be full 1st BZ.
        link_irr_1st_each_k : a dict of index in the axis 0 of k_full_1st_BZ: {<k in irrBZ in the entry> : [k1 equivalent, k2 equivalent....kn equivalent]}. With this link, you can move the energy of k point in irrBZ to other equivalent k.
    """
    ase_objet = structure.get_ase()
    primitive_cell = ase_objet.cell ### it's not necessary standard cell. On of the concequence is that a and b both doesnt aligned to x and y.
    

    spglib_symmetry = ase.spacegroup.symmetrize.check_symmetry(ase_objet)

    scaled_rotations = spglib_symmetry.rotations

    primitive_cell_rotations = [[]]

    for rotation in scaled_rotations : ### 
        cart_rotation = change_basis(rotation, np.linalg.inv(primitive_cell.T)) ### calculate rotaiton matrix in cartesian basis.
        cart_rotation = cart_rotation[:-1, :-1]
        primitive_cell_rotations.append(cart_rotation)
    
    primitive_cell_rotations.pop(0)


    k_full_1st_BZ = deepcopy(ks_irrBZ)

    link_irr_1st_each_k = {} ### a link of each k in irrBZ to its equivalent k in 1st BZ. the k in label in index
    for index_irr, k in enumerate(ks_irrBZ):
        for rot in primitive_cell_rotations :
            rot_angle_radian = find_rotation_axis(rot)[2]
            order = int(round(2 * np.pi / rot_angle_radian, 1)) ## i select the 1st digit cause it should be very close to and interger
            if order == 3 or order == 2: ### if rotation order is 2 or 3, the rotation will be + or - angle, that are all rotations.
                k_rotated = rot @ k      
                k_full_1st_BZ = np.vstack((k_full_1st_BZ, k_rotated))
                index_rotated = k_full_1st_BZ.shape[0] -1

                ## to concatenated the list in the key index_irr
                try:
                    link_irr_1st_each_k[index_irr].append(index_rotated)
                except :
                    link_irr_1st_each_k[index_irr] = [index_rotated]
            elif order == 4 : ## do twice rotation (4/2 where 2 means we have an rot + angle and rot - angle)
                k_rotated_1 = rot @ k
                k_full_1st_BZ = np.vstack((k_full_1st_BZ, k_rotated_1))
                index_rotated_1 = k_full_1st_BZ.shape[0] -1
                
                k_rotated_2 = rot @ rot @ k
                k_full_1st_BZ = np.vstack((k_full_1st_BZ, k_rotated_2))
                index_rotated_2 = k_full_1st_BZ.shape[0] -1

                try:
                    link_irr_1st_each_k[index_irr].append(index_rotated_1)
                    link_irr_1st_each_k[index_irr].append(index_rotated_2)
                except :
                    link_irr_1st_each_k[index_irr] = [index_rotated_1, index_rotated_2]
            elif order == 6 : ## similar to order == 4
                k_rotated_1 = rot @ k
                k_full_1st_BZ = np.vstack((k_full_1st_BZ, k_rotated_1))
                index_rotated_1 = k_full_1st_BZ.shape[0] -1

                k_rotated_2 = rot @ k
                k_full_1st_BZ = np.vstack((k_full_1st_BZ, k_rotated_2))
                index_rotated_2 = k_full_1st_BZ.shape[0] -1

                k_rotated_3 = rot @ k
                k_full_1st_BZ = np.vstack((k_full_1st_BZ, k_rotated_3))
                index_rotated_3 = k_full_1st_BZ.shape[0] -1

                try :
                    link_irr_1st_each_k[index_irr].append(index_rotated_1)
                    link_irr_1st_each_k[index_irr].append(index_rotated_2)
                    link_irr_1st_each_k[index_irr].append(index_rotated_3)
                except:
                    link_irr_1st_each_k[index_irr] = [index_rotated_1, index_rotated_2, index_rotated_3]

            elif order == 1:
                pass
            else :
                raise ValueError(f'The order of rotation of {rot} is {order}, is not in 1, 2, 3, 4, 6. It should be some value error in the rotation matrix above')
    
    return k_full_1st_BZ, link_irr_1st_each_k



            
def fullfill_bands_1st_BZ_without_time_reversal_symm(k_full_1st_BZ, link_irr_1st_each_k, energy_irrBZ : np.ndarray):

    """
    after found the link of index of k in irrBZ to the index in k of k_full_1st_BZ,
    one want to have energy array (1D where index are index of band) for each k point in k_full_1st_BZ 

    Note : 
    ------ No consideration of time reversal symmetry here. If there is time reversal symmetry, the other half of BZ should be fullfill by spin opposite band. using fullfill_bands_with_time_reversal_symmetry function.    
    """
    bands_full_1stBZ = np.zeros(shape=(np.shape(k_full_1st_BZ)[0], np.shape(energy_irrBZ)[1]))
    for index, k in enumerate(k_full_1st_BZ):
        if index in set(link_irr_1st_each_k.keys()):
            indexs_ks_equivalent = link_irr_1st_each_k[index]
            bands_full_1stBZ[index] = energy_irrBZ[index] ### first, put the energy of k in irrBZ to the k in full 1st BZ
            for index_eq in indexs_ks_equivalent:
                bands_full_1stBZ[index_eq] = energy_irrBZ[index] ### then put the energy of k in irrBZ to its equivalent k in full 1st BZ

    
    return bands_full_1stBZ

def mask_k_without_inversion_partener(kpoints : np.ndarray):

    """
    check which kpoint in kpoints dont have its inversion partener -k in the kpoints list to performe time reversal symmetry operation on the corresponding bands in the function fullfill_bands_with_time_reversal_symmetry.
    
    Args:
    -------
        kpoints : np.ndarray of shape (N kpoints, 3)
    
    """
    
    index_need_to_inverse = []
    for i, k in enumerate(kpoints):
        has_inversion_partener = False
        for j, k2 in enumerate(kpoints):
            if np.allclose(k, -k2, rtol= 1e-3) and i != j:
                has_inversion_partener = True
                break
        if not has_inversion_partener:
            index_need_to_inverse.append(i)

    return index_need_to_inverse


def fullfill_bands_with_time_reversal_symmetry(bands, k_points):
    """
    if you have time reversal symmetry (TR), some k' point's energy is obtain on the spin opposite band with k inversed. 
    Since DFT like QE just calculate the irrBZ, we sometimes need to fullfill the BZ. Here we do the operation of time reversal symmetry.
    Here we suppose the bands have on column band who distingush also spin's state. So bands number is even. Each paire of (n, n+1) with n even is a paire of band of spin opposite.
    

    
    Args :
    -------
        bands : np.ndarray of shape (N kpoints, M bands)
        k_points : np.ndarray of shape (N kpoints, 3)
    
        
    Returns:
    -------
        k_points_after_TR : np.ndarray of shape (2 N kpoints, 3)
        bands_after_TR : np.ndarray of shape (2 N kpoints, M bands)
    """

    

    ## here we need to avoid if some high symmetry k point do have itself inverse. Like Gamma point (0,0,0)
    ## so we select those k point who doesnt have themselves inverse in the k_points
    
    
        
    index_need_to_inverse = mask_k_without_inversion_partener(k_points)
    k_points_after_TR = np.vstack((k_points, -k_points[index_need_to_inverse]))

    bands_another_half = np.zeros(shape=(len(index_need_to_inverse),np.shape(bands)[1]))
    for i, index in enumerate(index_need_to_inverse):
        for j in range(np.shape(bands)[1]):
            if j % 2 == 1 :
                bands_another_half[i, j-1] = bands[index, j]
            elif j % 2 == 0:
                bands_another_half[i, j + 1] = bands[index , j]

    bands_after_TR = np.vstack((bands, bands_another_half))

    return k_points_after_TR, bands_after_TR


def generate_plotable_path_data(ks_irrBZ : np.ndarray,
                                energy_irrBZ : np.ndarray, 
                                R_vectors : np.ndarray,
                                G_vectors : np.ndarray,
                                structure : StructureData,
                                if_band_index_1fold_degeneracy : bool = True,
                                if_time_reversal_symmetry : bool = True,
                                ):
    """
    To generate the plotable data on the path between high symmetry points generated from the structure.
    We will do it between each high symmetry point given by ase.

    Args :
    ------
        ks_irrBZ : np.ndarray : shape (N,3) array of kpoints in irrBZ
        energy_irrBZ : np.ndarray : shape (nbands, N) array of band energies at each kpoint in irrBZ
        R_vectors : np.ndarray : shape (3,3) array of direct lattice vectors (whatever the unit you want)
        G_vectors : np.ndarray : shape (3,3) array of reciprocal lattice vectors (make sure it's coherent to the ks' unit in ks_irrBZ)
        structure : StructureData : the structure of type StructureData that you interste in.

        if_band_index_1fold_degeneracy : bool : True if the calculation make each index of band 1 fold degeneracy. Default is True. It didnt mean that there is no degeneracy at all, just that each index of band correspond to one 'state'.

        if_time_reversal_symmetry : bool : True if the calculation have time reversal symmetry. Default is True. If there is time reversal symmetry, the full 1st BZ is just half of the BZ. So the function fullfill_1stBZ_from_randomirrBZ will just fullfill half of the BZ. The other part should be given by the spin opposite band.
    
    Return :
    ------
        k_path : list of np.ndarray : list of kpoints on the full path generate by Atoms.cell.bandpath(pbc = (True, True, False)).
        bands_on_k_path : list of np.ndarray : list of band energies at each kpoint on the path between high symmetry points.
        x_coord_path : list of float : list of x coordinate of each kpoint on the path. The unit is simply the same as the unit of kpoints.
        special_kpoints_cartesian : dict : {'label of special k point' : np.array of k cartesian}
        special_kpoints_x_coord : dict : {x coordinate of special k point : 'label of special k point'} we do this cause x coord is unique but label could be repeated like 'Gamma' point in a closed (start at Gamma, end at Gamma).
        band_path_labels : str : a string like 'GMKG' each letter is a special point
    Note : the ks_irrBZ and energy_irrBZ, R_vectors should be all from the QE calculation of the structure. the R_vectors could be diff to the ase objet's cell from structure.
    
    TODO : the k path have repeated k point, so need to deal with the x coordinat of k point on k path. This calculation is not done yet. Done in
    """

    from ase.cell import Cell
    direct_cell = Cell(R_vectors)
    band_path = direct_cell.bandpath(pbc=(True, True, False))

    ## get special k points in cartesian coordinate
    special_kpoints = band_path.special_points
    special_kpoints_cartesian = {} ## fullfill it like {'label of special k point' : np.array of k cartesian}
    for label, kpoint_frac in special_kpoints.items():
        kpoint_cart = G_vectors.T @ kpoint_frac ## use the G_vector here, dont use other API to avoid if the cell is diff(in unit or others).
        special_kpoints_cartesian[label] = kpoint_cart
    full_BZ, link_irrBZ_full_BZ = fullfill_1stBZ_from_randomirrBZ(ks_irrBZ=ks_irrBZ, structure=structure)

    bands_full_BZ = fullfill_bands_1st_BZ_without_time_reversal_symm(full_BZ, link_irrBZ_full_BZ, energy_irrBZ) ## the 'full BZ's ' energy shape (kpoint index, band index)

    ## if both true, mean the 'bands_full_BZ' is not fullfill but a half
    ## so the extend full_BZ and bands_full_BZ will be stack on their time inversion part: another (N kpoint, 3) and (N kpoint, M bands) matrix. Both istance will have form (2 N kpoint, *) 
    ## then you get the true full BZ to match generate k path on it
    if if_band_index_1fold_degeneracy and if_time_reversal_symmetry: 
        full_BZ, bands_full_BZ = fullfill_bands_with_time_reversal_symmetry(bands=bands_full_BZ,
                                                                       k_points=full_BZ)
    # if if_band_index_1fold_degeneracy and if_time_reversal_symmetry: 
    #     full_BZ = np.vstack((full_BZ, -1 * full_BZ))
    #     bands_another_half = np.zeros(shape=(np.shape(bands_full_BZ)))
    #     for i in range(np.shape(bands_full_BZ)[0]):
    #         for j in range(np.shape(bands_full_BZ)[1]):
    #             if j % 2 == 1 :
    #                 bands_another_half[i, j-1] = bands_full_BZ[i, j]
    #             elif j % 2 == 0:
    #                 bands_another_half[i, j + 1] = bands_full_BZ[i , j]

    #     bands_full_BZ = np.vstack((bands_full_BZ, bands_another_half))

    band_path_labels = band_path.path ## a string like 'GMKG' each letter is a special point
    
    
    


    ### those matrice will all have the 1st dimension correspond to k point on the path
    k_path = []
    bands_on_k_path = []
    x_coord_path = [] 
    

    special_kpoints_x_coord = {} ## to store the x coordinate of each special k point on the path


    for index_k, k in enumerate(full_BZ):
        x_coor_stard = 0 ## Initialize the x coordinate of the start point of each segment 
        x_coor_end = 0 ## Initialize the x coordinate of the end point of each segment 
        ## This is to initialize the calculation of x coordinate.

        for index in range(len(band_path_labels) - 1):

            start_label = band_path_labels[index]
            end_label = band_path_labels[index + 1]
            start = special_kpoints_cartesian[start_label]
            end = special_kpoints_cartesian[end_label]
            
            ## x coordinate of start and end
            ## you can see here the unit of is simply the same as initial unit of all k
            x_coor_stard = np.linalg.norm(x_coor_end)
            x_coor_end = np.linalg.norm(end - start) + x_coor_stard

            special_kpoints_x_coord[x_coor_stard] = start_label
            special_kpoints_x_coord[x_coor_end] = end_label
            
            ## you can see for k in a segment (the if on_segment below),
            ##  the x coordinate of each end start are iterate and well calculate
            
            
            on_segment = point_on_line_segment(point=k, v1=start, v2=end, tolerance= 1e-4)
            if on_segment:
                k_path.append(k)
                bands_on_k_path.append(bands_full_BZ[index_k])

                ## calculate the x coord of k on k path 
                vector_k_start = k - start
                x_coord_k = np.linalg.norm(vector_k_start) + x_coor_stard
                x_coord_path.append(x_coord_k)


    k_path = np.array(k_path)
    bands_on_k_path = np.array(bands_on_k_path)
    x_coord_path = np.array(x_coord_path)

    ### order the path by x coordinate
    sorted_indices = np.argsort(x_coord_path)
    x_coord_path = x_coord_path[sorted_indices]
    k_path = k_path[sorted_indices]
    bands_on_k_path = bands_on_k_path[sorted_indices]
    return k_path, bands_on_k_path, x_coord_path, special_kpoints_cartesian, special_kpoints_x_coord, band_path_labels


    
    


# from analyse_symmetry import find_point_group_of_kpoint

def plot_spin_on_kmesh(spins :np.ndarray, 
                       kpoints:np.ndarray,
                       title : str,
                       save_path : str):
    """
    plot the spin texture on kpoints mesh. Those spins should from one single band.
    
    Z component is represent by color (red for positive, blue for negative). The value of components' are in [-0.5, 0.5]. In hbar unit.
    
    If you used the QE's bands from bands.x, be careful that the bands are not separated by band index n, but just by the order of energy. 
    This will make problem if there is band crossing.
    
    
    
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalisation centrée sur 0 avec vmin et vmax fixes
    norm = plt.Normalize(vmin=-0.5, vmax=0.5)
    import matplotlib.colors as mcolors
    cdict = {
    'red':   [(0.0, 0.0, 0.0),
              (0.5, 0.0, 0.0),
              (1.0, 1.0, 1.0)],
    'green': [(0.0, 0.0, 0.0),
              (0.5, 0.0, 0.0),
              (1.0, 0.0, 0.0)],
    'blue':  [(0.0, 1.0, 1.0),
              (0.5, 0.0, 0.0),
              (1.0, 0.0, 0.0)]
    }
    custom_cmap = mcolors.LinearSegmentedColormap('BlueBlackRed', cdict)

    norm = plt.Normalize(vmin=-0.5, vmax=0.5)
    colors = custom_cmap(norm(spins[:, 2]))

    # On peut séparer positive et negative comme tu veux, mais ici on prend tout ensemble :
    q = ax.quiver(kpoints[:, 0],
                kpoints[:, 1],
                spins[:, 0],
                spins[:, 1],
                color=colors,
                scale=20,
                width=0.005)

    # Barre de couleur pour Sz
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Spin $S_z$ component (hbar)')


    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title(f'Spin texture {title} (red: $S_z>0$, blue: $S_z<0$)')
    plt.axis('equal')
    plt.grid(True)
    ## need to check if the save_path's directory exist, if not create it
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()













    





    













































# %%
if __name__ == '__main__':
    pass



# %%
    from aiida import load_profile, get_profile
    if get_profile() == None:
        load_profile()

    import ase
    from aiida.orm import load_node
    from aiida.orm import load_code, load_computer

# %% figure storage path
    folder_fig = '/home/jyin/Images/Doc plot/bands'

# %% ## test parse_bands and parse_spins functions
    # path_mate = '/home/jyin/workspace/q-e-develop/qelocal/scratch_jy/MoS2Tempelate'

    struc_node = load_node('f2c23dd6-7ef1-4c6b-9bc9-973e63905a0d') # GeTe P3m1
    path_mate = load_node(519507).get_remote_path()
    path_nscf = '/home/jyin/workspace/scratch/ISDM_results/copied_from_26cc33fe-5994-4e72-bd1f-26b3691b4644/33fe-5994-4e72-bd1f-26b3691b4644' # GeTe P3m1
    
    
    # struc_node = load_node('de5d09ce-bf2d-4ba9-87d7-f342b2a2a636') # As2O3 C6V
    # path_mate = load_node(519676).get_remote_path() ## As2O3 C6v
    # path_nscf = load_node(519670).get_remote_path() # As2O3 C6v nscf.out

    # struc_node = load_node('9be800e5-41ac-4875-8746-e10b6995cb8b') # AsSb C3v
    # path_mate = load_node(519677).get_remote_path() ## AsSb C3v bands output folder
    # path_nscf = load_node(519671).get_remote_path() # AsSb C3v nscf.out folder
    bands_mate_1k_out_of_irrBZ, kpoints_mate_1k_out_of_irrBZ = parse_bands(path_mate, file='bands.dat')
    n_bands = bands_mate_1k_out_of_irrBZ.shape[0]
    spins_mate_1k_out_of_irrBZ = parse_spins(path_mate, list_bands=range(n_bands), _file_='bands.dat')

# %% create bandsdata node
    # from aiida.plugins import DataFactory
    # BandsData = DataFactory('core.array.bands')
    from aiida.orm import BandsData
    bands_data = BandsData()
    bands_data.set_kpoints(kpoints_mate_1k_out_of_irrBZ) # shape (n_kpoints, 3)
    bands_data.set_bands(bands_mate_1k_out_of_irrBZ.T)  # transpose to shape (n_kpoints, n_bands)


# %% in BandsData get KpointData objet
    kpoints_aiida = KpointsData()
    kpoints_aiida.set_kpoints(kpoints_mate_1k_out_of_irrBZ)
   

# %%
    kpoints_mate = bands_data.get_kpoints() 
    G_vectors_mate, R_vectors_mate, lattic_vec = read_nscf_out_file(path_nscf, _file_='aiida.out')
# %% parse high symmetry path in aiida.out in the out put RemoteData's path of bands.x calculation

    high_S_kpoints_dict = parse_high_symmetry_path(path_mate, file='aiida.out')
# %% set cell for kpoints_aiida
    R_vectors_mate_ang = R_vectors_mate * lattic_vec[0] * 0.529177 ## convert to angstrom
    kpoints_aiida.set_cell(cell=R_vectors_mate_ang,pbc=[True, True, False])
# # %% set cell for bands_data
#     bands_data.set_cell(cell=R_vectors_mate_ang,pbc=[True, True, False])
# # %% get bands plot data 
#     mpl_data = bands_data._get_bandplot_data(cartesian=True)
#     bands_segments = bands_data._get_band_segments(cartesian=True)
# %% plot bands_data by segments : (like Gamma to Y, Gamma to X, X to Y......)

    # plot_bands_high_symmetry_path(energies_path=mpl_data['y'],
    #                        n_bands_occupied=n_bands_occupied,
    #                        e_min_ev=-5,)
   
        

# %% find path with ase function
    # from ase.dft import kpoints
    # path_ase = kpoints.labels_from_kpts(kpts=kpoints_aiida.get_kpoints() ,
    #                                   cell=R_vectors_mate,
    #                                   eps=1e-3)

# %%
    spin_VB_mate = spins_mate_1k_out_of_irrBZ[n_bands_occupied-1]  # spin of the highest occupied band
    VB_mate = bands_mate_1k_out_of_irrBZ[n_bands_occupied-1]  # energy of the highest occupied band





# %% plot band VB mate with spin texture
    # plot_mate = plot_bands_2D_with_spin_interac(bands=VB_mate, 
    #                                             kpoints=kpoints_mate_cart[:,:2], 
    #                                             spin_vectors_each_band=spin_VB_mate,
    #                                             want_plot_spin=True,
    #                                             scale_factor_spin=0.5,
    #                                             vector_density=500)
# %% fix kpoints
    # kpoints_mate, index_k_to_remove = kpoints_fix_v1(kpoints_mate_1k_out_of_irrBZ)
    # spins_mate = np.delete(spins_mate_1k_out_of_irrBZ, index_k_to_remove, axis=1)
    # bands_mate = np.delete(bands_mate_1k_out_of_irrBZ, index_k_to_remove, axis=1)
    spins_mate = spins_mate_1k_out_of_irrBZ
    bands_mate = bands_mate_1k_out_of_irrBZ
# %% converHull's contour 
    # contour_mate,_ , vertex_mate = get_contour_points_from_hull(kpoints_mate, tolerance=1e-3)
# %% plot kpoints mesh
    plt.figure()
    plt.scatter(kpoints_mate[:,0], kpoints_mate[:,1], s=1)
    plt.axis('equal')
    plt.show()

# %% 
    # plt.figure()
    # plt.scatter(contour_mate[:,0], contour_mate[:,1], s=1)
    # plt.axis('equal')
    # plt.show()


# %% find path with an ase function
    # path_mate = ase.dft.kpoints.labels_from_kpts(kpts = contour_mate,
    #                                              cell = R_vectors_mate,
    #                                              eps = 1e-3)
    ## this don't work well.
# %% find path between given high symmetry kpoints
    # point_G = np.array([0.0, 0.0,0.0])
    # # point2 = np.array([0.      , 0.529238, 0.      ])
    # point2 = np.array([0.333333, 0.57735 , 0.      ])
    # segment = path_segment_between_high_S_kpoint(point_G, point2, contour_mate)
# %% find segment with energy
    # energies_segment = []
    # spins_segment = []
    # for k in segment:
    #     index_k = np.where((kpoints_mate == k).all(axis=1))[0][0]
    #     energies_segment.append(bands_mate[:,index_k])
    #     spins_segment.append(spins_mate[:,index_k])
    # energies_segment = np.array(energies_segment)
    # band_VBM = energies_segment[:, n_bands_occupied -1]
    # band_CBM = energies_segment[:, n_bands_occupied ]
    # spins_segment = np.array(spins_segment)
# %% spin of VB and CB or more
    

    spin_VB = spins_mate[n_bands_occupied -1, : ]
    spin_CB = spins_mate[n_bands_occupied , : ]
    spin_CB_pair = spins_mate[n_bands_occupied + 1, :]
    spin_VB_pair = spins_mate[n_bands_occupied - 2, :]

# %% plot spin texture of other bands 
    band_index_diff_to_CB = 4
    band_index = n_bands_occupied + band_index_diff_to_CB
    spin_other_band = spins_mate[band_index, :]

# %% plot energies_segment
    # plot_path_without_spin(energies_segment, 
                         
    #                     n_bands_occupied, 
    #                     e_max_ev=-1,
    #                     e_min_ev=-6,
    #                     # index_band_min=n_bands_occupied + 1,
    #                     # index_band_max=n_bands_occupied + 1
    #                     )
# %% get some info to use in plots' save
    chemical_symbols = struc_node.get_ase().get_chemical_formula()
    space_group = Atoms_symmetry_group_direct_space()
    space_group.get_site_point_group(struc_node.get_ase())
    space_group_symbol = space_group._space_group_HM 
# %% plot spin texture of all irrBZ's kpoint mesh.
    # plot_spin_on_kmesh(spin_CB, kpoints_mate[:,:2])
    for i in range(0,6):
        band_index_diff_to_CB = i * (1)
        band_index = n_bands_occupied + band_index_diff_to_CB
        spin_other_band = spins_mate[band_index, :]
        title = f'n CB + ({band_index_diff_to_CB})'
        plot_spin_on_kmesh(spin_other_band, 
                           kpoints_mate[:,:2],
                           title= title,
                           save_path=f'{folder_fig}/{chemical_symbols}_{space_group_symbol}/spin_{title}.png')
    
    



    

# %%
    ase_objet = struc_node.get_ase()
# %%
    asgds = Atoms_symmetry_group_direct_space()
    asgds.get_site_point_group(ase_objet)
    spglib_symmetry = asgds.get_spglib_symmetry_data()
    rotations = spglib_symmetry['rotations']
    # translations = spglib_symmetry['translations'] ## nonsymmorphic operations is not metter in reciprocal space. It will go outside of the BZ.






# %%
    k_full_BZ, links_index_irr_half_BZ = fullfill_1stBZ_from_randomirrBZ(kpoints_mate, structure=struc_node)
# %% plot k_full_BZ and k in irrBZ generated by BZ
    # plt.figure()
    # plt.scatter(k_full_BZ[:,0], k_full_BZ[:,1], s=1)
    # plt.axis('equal')
    # plt.show()





# %%
    bands_full_BZ = fullfill_bands_1st_BZ_without_time_reversal_symm(k_full_1st_BZ=k_full_BZ, 
                                     link_irr_1st_each_k=links_index_irr_half_BZ, 
                                     energy_irrBZ=bands_mate.T)
# %%
    k_path, bands_path, x_coordinates_path, special_points_cart, special_point_x_coor, band_path_label = generate_plotable_path_data(ks_irrBZ=kpoints_mate, 
                                                                    energy_irrBZ=bands_mate.T,
                                                                    R_vectors= R_vectors_mate,
                                                                    G_vectors=G_vectors_mate,
                                                                    structure=struc_node,
                                                                    if_time_reversal_symmetry=True,)


# %%
    plot_path_without_spin(energies_path=np.array(bands_path), 
                           x_coordinates=x_coordinates_path,
                           n_bands_occupied=n_bands_occupied,
                           special_points=special_point_x_coor,
                           e_min_ev=-7,
                           title=f'{chemical_symbols} {space_group_symbol} band structure with SOC',
                           path_save_fig=f'{folder_fig}/{chemical_symbols}_{space_group_symbol}/{chemical_symbols}_{space_group_symbol}_band_structure_{band_path_label}_with_SOC.png',
                        #    k_max= 0.2
                           )

# %%
    # plt.figure()
    # plt.scatter(k_path[:,0], k_path[:,1], s=1)
    # plt.axis('equal')
    # plt.show()
# %%
    E_CB = bands_mate[n_bands_occupied:n_bands_occupied + 2, :]
# %%
