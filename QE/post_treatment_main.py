import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from spin.spin_texture import parse_bands, parse_spins, plot_bands_2D_with_spin_interac,plot_spin_on_kmesh


import ase
import spin.spin_texture as stt
import os
import argparse
from pathlib import Path

from aiida.orm import QueryBuilder, Node
from aiida.engine import run
from aiida.orm.groups import Group
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, load_node, load_computer

from analyse_symmetry import Atoms_symmetry_group_direct_space

import numpy as np

import fit_model_DFTdata as fit

def parse_args():
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument('--input_folder_nscf', type=str, help="Folder of nscf calculation's output as absolut path")
    parser.add_argument('--input_folder_bandx', type=str, help="Folder of band.x calculation's output as absolut path")
    parser.add_argument('--structure_data_uuid', type=str, help="structure data's uuid")
    parser.add_argument('--config', type=str, help="JSON config file with all input arguments")
    # parser.add_argument('--number', type=int, default=10, help="Number parameter")
    # parser.add_argument('--flag', action='store_true', help="An example boolean flag")
    return parser.parse_args()

def check_arg1_is_folder_contain_bandsx_output(input_path, prefix = 'bands.dat'):
    if os.path.exists(input_path):
        if os.path.isfile(input_path):
            raise ValueError("please enter an folder, here we want a folder with band.x output")
        elif os.path.isdir(input_path):
            path_objet = Path(input_path)
            for file in path_objet.iterdir():
                if file.is_file() and file.name.startswith(prefix):
                    input_folder_spin = input_path
                    return input_folder_spin
            raise ValueError("it's a folder but not bandx calculation output folder")
        else:
            raise ValueError("It's neither a file nor a folder path.")

def check_arg1_is_folder_contain_nscf_output(input_path, prefix = 'aiida.in'):
    if os.path.exists(input_path):
        if os.path.isfile(input_path):
            raise ValueError("please enter an folder, here we want a folder with nscf output")
        elif os.path.isdir(input_path):
            path_objet = Path(input_path)
            for file in path_objet.iterdir():
                if file.is_file() and file.name == 'aiida.in':
                    with open(file=file) as f :
                        lines = f.readlines()

                    if len(lines) < 2:
                        raise ValueError("it's a folder but not nscf calculation output folder")

                    # Check if first line is exactly '&CONTROL'
                    elif '&CONTROL' not in lines[0] :
                        raise ValueError("it's a folder but not nscf calculation output folder")
                        

                    # Check if second line contains "calculation = 'nscf'"
                    elif "calculation = 'nscf'" not in lines[1]:
                        raise ValueError("it's a folder but not nscf calculation output folder")
                        
                    else :
                        return input_path
            raise ValueError("it's a folder but not nscf calculation output folder")
        else:
            raise ValueError("It's neither a file nor a folder path.")




def generate_spin_plot_kmesh_with_band_indexs(spins_mate : np.ndarray, 
                                              kpoints_mate : np.ndarray, 
                                              bands_indexs : list, 
                                              save_path : str):


    for band_index in bands_indexs:
            # band_index_diff_to_CB = i * (1)
            # band_index = n_bands_occupied + band_index_diff_to_CB
            spin_other_band = spins_mate[band_index, :]
            title = f' spin at n = ({band_index})'
            plot_spin_on_kmesh(spin_other_band, 
                            kpoints_mate[:,:2],
                            title= title,
                            save_path=save_path)

















if __name__ == '__main__':
    

    n_bands_occupied = 30 ### manually found, TODO : rewrite the function to find it. 
    i_band_lowest_inoccupied = 31

    current_folder = os.getcwd()
    # args = parse_args()
    with open(f'{current_folder}/QE/input_files/GeTe_P3m1.json') as f_json :
        params = json.load(f_json)
        input_folder_nscf = params['input_folder_nscf']
        input_folder_spin = params['input_folder_bandx']
        uuid = params['structure_data_uuid']

    input_folder_spin = check_arg1_is_folder_contain_bandsx_output(input_path=input_folder_spin)
    input_folder_nscf = check_arg1_is_folder_contain_nscf_output(input_path=input_folder_nscf)
    # uuid = args.structure_data_uuid


    ## just some exemple for debug :
    # input_folder_nscf = '/home/jyin/workspace/scratch/ISDM_results/copied_from_26cc33fe-5994-4e72-bd1f-26b3691b4644/33fe-5994-4e72-bd1f-26b3691b4644'
    # input_folder_spin = '/home/jyin/workspace/scratch/69/9e/7e3f-3b07-4921-928e-fe0d8ca4398e'
    # uuid = 'f2c23dd6-7ef1-4c6b-9bc9-973e63905a0d'

    path_storage_general = '/home/jyin/workspace/scratch/post_processing_QE'

    ## materials' basic infomation :
    try:
        struc_data = load_node(uuid=uuid)
        ase_objet = struc_data.get_ase()
        
    except :
        raise ValueError("the uuid provide don't point to a aiida StructureData")
    
    chemical_formula = ase_objet.get_chemical_formula()
    dat_syms = Atoms_symmetry_group_direct_space()
    dat_syms.get_site_point_group(ase_objet)
    space_group = dat_syms._space_group_HM
    point_group = dat_syms._point_group_Sch
    Hsoc_kp = fit.kp_Hsoc_point_group[point_group][0]
    Hsoc_kp_spin_scalar = fit.kp_Hsoc_point_group[point_group][1]

    ## use material's information to name the storage folder of outputs
    name_folder_analysis = path_storage_general + '/' + f'{chemical_formula}' + '_' + f'{space_group}' + '_' + f'{uuid}'

    ### read the bands and k points and spin datas
    bands_irrBZ, kpoints_irrBZ = parse_bands(input_folder_spin, file='bands.dat')
    n_bands = bands_irrBZ.shape[0]
    spins_irrBZ = parse_spins(input_folder_spin, list_bands=range(n_bands), _file_='bands.dat')
    G_vectors_mate, R_vectors_mate, lattic_para = stt.read_nscf_out_file(input_folder_nscf, _file_='aiida.out')

    ### find pairs of bands with spin opposite
    spin_VB = spins_irrBZ[n_bands_occupied, :]
    spin_VB_pair = None
    band_VB = bands_irrBZ[n_bands_occupied, :]
    band_VB_pair = None
    spin_CB = spins_irrBZ[i_band_lowest_inoccupied, :]
    spin_CB_pair = None
    band_CB = bands_irrBZ[i_band_lowest_inoccupied, :]
    band_CB_pair = None
    if n_bands_occupied % 2 == 0:
        spin_VB_pair = spins_irrBZ[n_bands_occupied + 1, :]
        band_VB_pair = bands_irrBZ[n_bands_occupied + 1, :]
    elif n_bands_occupied % 2 == 1:
        spin_VB_pair = spins_irrBZ[n_bands_occupied - 1, :]
        band_VB_pair = bands_irrBZ[n_bands_occupied - 1, :]
    
    else :
        raise TypeError('n_bands_occupied must be an integer')
    
    if i_band_lowest_inoccupied % 2 == 0:
        spin_CB_pair = spins_irrBZ[i_band_lowest_inoccupied + 1, :]
        band_CB_pair = bands_irrBZ[i_band_lowest_inoccupied + 1, :]
    elif i_band_lowest_inoccupied % 2 == 1:
        spin_CB_pair = spins_irrBZ[i_band_lowest_inoccupied - 1, :]
        band_CB_pair = bands_irrBZ[i_band_lowest_inoccupied - 1, :]
    else :
        raise TypeError('i_band_lowest_inoccupied must be an integer')
    

    


    #### fit rashba hamiltonian 
    energys_CB_CB_pair = np.vstack((band_CB, band_CB_pair)).T

    ## CB pair's spin splitting
    model, terms, eigvals, eigvecs, alphas, score_diag = fit.process_diag_until_linearmodel(H=Hsoc_kp,
                                                              energys_2_bands=energys_CB_CB_pair,
                                                              kxy=kpoints_irrBZ,
                                                              radius_to_fit=0.15)
    
    # print(model.coef_)
    # print(terms)
    # print(alphas)

    sols = fit.solve_alphas(alphas=alphas,
                        terms=terms,
                        models_line=model)
    
    convert_factor_k_GeTe_QE_to_1on_angs = fit.convert_factor_k_alat_on_bohr_angs(latt_len_bohr=lattic_para[0])
    
    spin_CB_CB_pair = np.zeros(shape=(2, spin_CB.shape[0], 3))
    spin_CB_CB_pair[0] = spin_CB
    spin_CB_CB_pair[1] = spin_CB_pair
    
    print("Model fitting using directly diagonalization")
    print(sols)
    for index, sol in enumerate(sols):
        print(f"solution {index} below:")
        for i, (para, val) in enumerate(sol.items()):
            print(f"{para} : {val / convert_factor_k_GeTe_QE_to_1on_angs}", end=" ")
        print(f"unit : ev/angstrom or ev/(angstrom ^ 3)")
        print('\\')

    print(f"score of diagonalization method : {score_diag}")

    model_with_DFT_spin, score_with_DFT_spin = fit.process_using_DFT_spin_until_linearmodel(H=Hsoc_kp_spin_scalar,
                                                                                                energys_2_bands=energys_CB_CB_pair,
                                                                                                kxy=kpoints_irrBZ,
                                                                                                spin_texture=spin_CB_CB_pair,
                                                                                                raidus_to_fit=0.15)
    # print("Model fitting using directly diagonalization")
    # print(sols)
    
    print("below Rashba coef with DFT spin inserted")
    print(np.array(model_with_DFT_spin.coef_) / convert_factor_k_GeTe_QE_to_1on_angs)
    print(f"unit : ev/angstrom or ev/(angstrom ^ 3)")
    print(f"score : {score_with_DFT_spin}")

    

    

    



    