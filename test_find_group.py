# In[]
import find_groups
import importlib
importlib.reload(find_groups)
from find_groups import *
import time

# In[]
from aiida import load_profile, get_profile
if get_profile() == None:
    load_profile()


from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, load_node
from aiida.orm.nodes.process.calculation.calcfunction import CalcFunctionNode, CalcFunctionNodeLinks
from aiida.tools.visualization import Graph
# In[]
#### les fonction de test
def test_read_spacegroup_pointgroup():
    ensemble_path_cif = ask_file_or_directory_cif()

    for file_path in ensemble_path_cif:
        # atoms = read_cif_ase(file_path)
        # atoms = read_cif_pyxtal(file_path)
        # space_group, point_group = get_spacegroup_pointgroupe_pyxtal(atoms)
        # print(space_group)
        # print(point_group)


        # atoms = read_cif_pyxtal(file_path)
        # space_group_pyxtal, point_group_pyxtal = get_spacegroup_pointgroupe_pyxtal(atoms)
        # print(space_group_pyxtal)
        # print(point_group_pyxtal)

        atoms = read_cif_ase(file_path)
        space_group_spglib, point_group_spglib, symmetry_data = get_space_point_group_spglib(atoms, symprec)
        print(space_group_spglib)
        print(point_group_spglib)


# def test_site_sys():
#     ensemble_path_cif = ask_file_or_directory_cif()
#     for file_path in ensemble_path_cif:
#         atoms = read_cif_ase(file_path)
        
#         site_symm, ITA_HM, ITA_Number = get_site_point_group_pyxtal(atoms)
#         print(ITA_HM)
#         print(ITA_Number)
#         for site in site_symm:
#             print(site[:-1])
#         # site_point_group_all_atoms_pyxtal(c_pyl)


def test_site_symmetry_database_spglib():
    ensemble_path_cif = ask_file_or_directory_cif()
    for file_path in ensemble_path_cif:
        atoms = read_cif_ase(file_path)
        space_group_spglib, point_group_spglib, atoms_wyckoff_site_symmetry = get_space_point_group_spglib(atoms, symprec)
        print(space_group_spglib)
        print(point_group_spglib)
        print(atoms_wyckoff_site_symmetry)


def test_get_site_symm_ops_from_wp():
    site_symmetry_objet = get_symm_ops_of_site_symmetry_from_wp(187, 'm')[0]
    opas = site_symmetry_objet.opas
    for opa in opas :
        print(opa)
    







def test_site_point_group_from_a_material():
    ensemble_path_cif = ask_file_or_directory_cif()
    for file_path in ensemble_path_cif:
        atoms = read_cif_ase(file_path)
        all_sites_info = site_point_group_from_a_material(atoms)
        print(all_sites_info)


def test_bi_check_if_string_disorded():
    Bool = bi_check_if_string_disorded('-42m', '-4m2')
    print(Bool)

def test_if_site_point_group_equal_site_symetry ():
    """
    fuck, site point group cant be found symbolily by site symmetry symbol
    
    """
    ensemble_path_cif = ask_file_or_directory_cif()
    Bool_tot = True
    for file_path in ensemble_path_cif:
        atoms = read_cif_ase(file_path)
        all_sites_info = site_point_group_from_a_material(atoms)[4]
        for site in all_sites_info:
            site_point_group_symbol = site[1]
            site_symmetry_symbol = site[3]
            Bool = bi_check_if_string_disorded(site_point_group_symbol, site_symmetry_symbol)
            Bool_tot = Bool_tot and Bool
            if not Bool :
                print(file_path)
                print('---------')
                print(f"{site}")

    print(f" if all the site symmetry in ITA are equal to site point group calculated : {Bool_tot}") 
    return Bool_tot
            


            


if __name__ == '__main__':
    

    ### test utiliser le lecture avec pymatgen : induit les erreur pendant la lecture des fichier.
    # parser = CifParser(filepath)
    # structure = parser.get_structures()[0]

    # lattice = structure.lattice.matrix
    # positions = structure.frac_coords
    # atomic_numbers = [site.specie.number for site in structure]
    # cell = (lattice, positions, atomic_numbers)

    # spacegroup = spg.get_spacegroup(cell)
    # print("Space group:", spacegroup)
    
    
    # test_read_spacegroup_pointgroup()
    # test_site_sys()
    # test_get_symmetry()
    # test_site_symmetry_database_spglib()
    # test_get_site_symm_ops_from_wp()
    # test_is_a_pointgroup_site_point_group()

    # test_list_subgroup_of_pointgroup()

    # test_site_point_group_from_a_material()
    # test_bi_check_if_string_disorded()

# In[]
    test_if_site_point_group_equal_site_symetry()

# In[]
### test if all_remain_same_after_some_op() for 2 exemple, one should return true and the other false.
    
    ### exemple spglib return a wrong space group :
    node_Hg4Mo2O8 = load_node('daaee44e-78b0-4002-85c9-25f55fcf3b31')
    ase_obj_Hg4Mo2O8 = node_Hg4Mo2O8.get_ase()
    spg_number = get_space_point_group_spglib(ase_obj_Hg4Mo2O8)[0][1]
    spg_objet = Group(spg_number)
    spg_opas = spg_objet.get_spg_symmetry_object().opas
    is_spg_correct = all_remain_same_after_some_op(ase_obj_Hg4Mo2O8, spg_opas)
    print(is_spg_correct)



# In[]
    ase_obj_Hg4Mo2O8 = node_Hg4Mo2O8.get_ase()
    Atoms().cell.cellpar()
    angles = ase_obj_Hg4Mo2O8.cell.cellpar()
    cell_ = ase_obj_Hg4Mo2O8.get_cell()
    print(angles)
    print(cell_)
# In[]
    ase_obj_Hg4Mo2O8.edit()
# In[]
    vec1 = cell_[0]
    vec2 = cell_[1]
    vec3 = cell_[2]
    angle_alpha = angle_2_vec(vec1, vec2)
    angle_beta = angle_2_vec(vec2, vec3)
    angle_gamma = angle_2_vec(vec1, vec3)
    print(vec1)
    print(vec2)
    print(vec3)
    print(cell_)
    print(angle_alpha)
    print(angle_beta)
    print(angle_gamma)
# In[]

    rot_mats = get_space_point_group_spglib(ase_obj_Hg4Mo2O8)
    print(rot_mats)
# In[]

    cell_to_spglib = (ase_obj_Hg4Mo2O8.get_cell(), ase_obj_Hg4Mo2O8.get_scaled_positions(), ase_obj_Hg4Mo2O8.get_atomic_numbers())
    datasset = spg.get_symmetry_dataset(cell_to_spglib)
    print(datasset)
# In[]
    ### exemple spglib return a wrong space group :
    node_MoS2_6 = load_node('d25962ea-3c39-4e86-9333-2baf8f31a1a8')
    ase_obj_MoS2_6 = node_MoS2_6.get_ase()
    spg_number = get_space_point_group_spglib(ase_obj_MoS2_6)[0][1]
    # spg_objet = Group(spg_number)
    # spg_opas = spg_objet.get_spg_symmetry_object().opas
    # is_spg_correct = all_remain_same_after_some_op(ase_obj_MoS2_6, spg_opas)
    # print(is_spg_correct)

# In[]
    asgds_MoS2_6 = Atoms
# %%
    cell_eg = ase_obj_MoS2_6.get_cell()
    cell_eg = np.transpose(cell_eg)
    itself_inv = np.linalg.inv(cell_eg) @ cell_eg 
    condition = np.linalg.cond(cell_eg)
    print(cell_eg)
    print(condition)
    print(itself_inv)

# In[]
    component_translation_possible = check_what_are_possible_scaled_translate_component()
    print(component_translation_possible)
# %%
###  mirror in plan of in a crystal having C3 or C6 axis. (basis vector a and b form angle 120 degree) 
# Test if the mirror in the plan can be correctly convert from coordinate cartesien to coordinate of bravais lattice.
    def set_a_hex_cell(a_length, c_length):
        
        a_vec = [a_length, 0, 0]
        b_vec = [a_length*np.cos(2*np.pi/3), a_length*np.sin(2*np.pi/3),0]
        c_vec = [0,0,c_length]
        cell_hex_eg = np.array([a_vec, b_vec, c_vec])
        cell_hex_eg = np.transpose(cell_hex_eg)
        return cell_hex_eg
# In[]
    def set_a_rotation_mat(angle_degree):
        angle_rad = np.deg2rad(angle_degree)
        mat = np.array([[np.cos(angle_rad), -1* np.sin(angle_rad), 0],
                        [np.sin(angle_rad), np.cos(angle_rad) , 0],
                        [0, 0, 1]])
        return mat
# In[]
    a_length = 8.901673417
    c_length = 23.8718379

    cell_hex_eg = set_a_hex_cell(a_length=a_length, c_length=c_length)
    
    #### a mat cartesian of mirror in the plan (xz)
    mirror_mat_carte = np.array([[1,0,0],
                           [0,-1,0],
                           [0,0,1]])
    mirror_mat_frac = change_matrix_basis_from_eucli_to_frac(matrix=mirror_mat_carte, abc_vector=cell_hex_eg)
    print(mirror_mat_frac)

# In[]
    def deviate_cell(cell : np.ndarray, deviation_angle_degree : float):
        cell_T = np.transpose(cell)
        rotation_mat = set_a_rotation_mat(deviation_angle_degree)
        lignes = []
        for i in range(np.shape(cell_T)[0]):
            ligne = rotation_mat @ cell_T[i]
            lignes.append(ligne)
        new_cell = np.array(lignes)
        new_cell = np.transpose(new_cell)
        return new_cell
# In[]
    cell_hex_eg_deviated = deviate_cell(cell_hex_eg, deviation_angle_degree=10.)
    mirror_mat_frac_cal_deviated_cell = change_matrix_basis_from_eucli_to_frac(matrix=mirror_mat_carte,
                                                                               abc_vector=cell_hex_eg_deviated)
    print(mirror_mat_frac_cal_deviated_cell)

# In[]

    
    import pandas as pd
    path_all_pks_struc2D = '/home/jyin/workspace/gene_data/pks_struc_2D_diff_cate/pks_chem_all.csv'
    pks_struc2D = pd.read_csv(path_all_pks_struc2D)
    pks_struc2D = pks_struc2D.iloc[:,0].to_list()

# In[]
    import statistique
    importlib.reload(statistique)
    from statistique import test_collinear_basis_in_plan__xy

# In[]
    result = test_collinear_basis_in_plan__xy(pks_struc2D)

# In[]
    

    