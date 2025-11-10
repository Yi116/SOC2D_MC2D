from analyse_symmetry import *
from band_analyse import *


import json 
import time


#### aiida importation
import aiida


# from IPython import get_ipython
# ipython = get_ipython()
# ipython.magic("load_ext aiida")
# %load_ext aiida
# %aiida

from aiida import load_profile, get_profile
if get_profile() == None:
    load_profile()


from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, load_node
from aiida.orm.nodes.process.calculation.calcfunction import CalcFunctionNode, CalcFunctionNodeLinks
from aiida.tools.visualization import Graph

### assigne some type to types alias
# Using typing module
from typing import TypeAlias

CalcFunctionNode : TypeAlias = aiida.orm.nodes.process.calculation.calcfunction.CalcFunctionNode
BandsData : TypeAlias = aiida.orm.nodes.BandsData
JobCalculation : TypeAlias = aiida.orm.nodes.process.calculation.calcjob.CalcJobNode
Node : TypeAlias = aiida.orm.nodes.Node
Dict_aiida = aiida.orm.nodes.data.dict.Dict

# from aiida import load_profile, get_profile
# if get_profile() == None:
#     load_profile()

### a small test if the bravais lattice in plan basis vector a, b have one collinear to cartesian vector x or y.
def test_collinear_basis_in_plan__xy(pks):
    """
    check if the bravais lattice in plan basis vector a, b have one collinear to cartesian vector x or y.
    for reason, see find_groups.operate_to_unit_cell()

    ## Indeed, with error of a component less than 1e-5, it's true
    """
    all_colinear = True
    for pk in pks:
        node = load_node(pk) 
        ase_objet = node.get_ase()
        cell = ase_objet.get_cell()
        any_colinear = False
        for i in range(np.shape(cell)[0]):
            basis_vec = cell[i, :]
            if round(basis_vec[0], 5) == 0. or round(basis_vec[1], 5) == 0.:
                any_colinear = True
        
        all_colinear = all_colinear and any_colinear
        if not all_colinear:
            return False, pk
    
    return all_colinear


def stat_R2_D2(data : list[Atoms_symmetry_group_direct_space]):
    #### function checked
    N_crys_analysed = len(data) ### number of materiaux totaly analysed.
    N_hid_R_D = 0 ### number of materiaux who have hidden Rashba-Dressehauls or Dressehauls effet.
    N_no_hid_R_D = 0 ### number of materiaux who have non hidden (conventional) Rashba-Dressehauls effet or Dressehauls effet.

    N_hid_RD = 0 ### number of materiaux who have hidden Rashba-Dressehauls.
    N_hid_D = 0 ### number of materiaux who have pure hidden Dressehauls effet.
    N_D1_NP_sites = 0 ### number of materiaux who have pure Dressehauls effet in case all site are Non polar
    N_R1_and_D1 = 0 ### number of materiaux who have R-1 and D-1 effet because they have polar space group
    # N_unknow_case = 0 ### number of materiaux who have non centrosymmetry space group and at least one site point group is polar. There are 2 case and 2 consequences, indeterminate for now
    N_D1_P_sites = 0 ### number of materiaux who have pure Dressehauls effet in case at least one polar site (dipole add up to zero)
    N_no_R_D = 0 ### number of materiaux who don't have  Rashba or Dressehauls effet

    N_unexpecte = 0 ### number of materiaux who have unexpecte case (case not yet studied)
    N_error_case = 0 ###
    for crystal in data :
        SOC_type = find_Rashba_Dresshauls_effet_of_1_crystal(crystal)
        if SOC_type == "R-2 and D-2":
            N_hid_RD += 1
            N_hid_R_D += 1
        elif SOC_type == "D-2":
            N_hid_D += 1
            N_hid_R_D += 1
        elif SOC_type == "D-1 no any polar site":
            N_D1_NP_sites += 1
            N_no_hid_R_D += 1
        elif SOC_type == "R-1 and D-1":
            N_R1_and_D1 += 1
            N_no_hid_R_D += 1
        elif SOC_type == "D-1 polar site exist":
            N_D1_P_sites += 1
            N_no_hid_R_D += 1
        elif SOC_type == "no R or D" :
            N_no_R_D += 1
        elif SOC_type == "unexpected case" :
            N_unexpecte += 1
            
        else :
            print(crystal._formula)
            print(f"this error have SOC type:{SOC_type} ")
            N_error_case += 1

    dic = {"no R or D" : N_no_R_D,
           "R-2 or D-2" : N_hid_R_D,
           "R-1 or D-1" : N_no_hid_R_D, ### until here main categorie
           
           "R-2 and D-2" : N_hid_RD,
           "D-2" : N_hid_D,
           "D-1 no any polar site" : N_D1_NP_sites,
           "R-1 and D-1" : N_R1_and_D1,
           "D-1 polar site exist" : N_D1_P_sites,

           "unexpected case" : N_unexpecte,
           "error case" : N_error_case,
           "N total" : N_crys_analysed}
    return dic



def stat_R2_D2_from_json(path):
    #### funciton checked
    df = get_dataframe_ASGDS_json(path)
    # print(df.columns)
    col = df['Material and Direct space symmetrys']
    col = col.to_list()
    stat = stat_R2_D2(col)
    return stat

def stat_max_Z_at_no_centro_site(materiaux : list[Atoms_symmetry_group_direct_space]):
    """
    list of max Z number has a non centrosymmetrc site in a materiaux.
    another list of chemical formula of each index of the first list.
    """
    list_max_Z = []
    list_formula = []
    for mat in materiaux :
        max_Z = find_max_Z_at_no_centrosymmetric_sites(mat)
        formula = mat._formula
        list_max_Z.append(max_Z)
        list_formula.append(formula)
    return list_max_Z, list_formula

def stat_max_Z_from_json(path):
    df = get_dataframe_ASGDS_json(path)
    # print(df.columns)
    col = df['Material and Direct space symmetrys']
    col = col.to_list()
    stat = stat_max_Z_at_no_centro_site(col)
    return stat

def stat_C1_mirror_point_group( data : list[Atoms_symmetry_group_direct_space]):
    N_C1 = 0
    N_Cs = 0
    for d in data:
        if d._point_group_Sch == 'C1':
            N_C1 += 1
        if d._point_group_Sch == 'Cs':
            N_Cs += 1
    return N_C1, N_Cs


def stat_C1_mirror_from_json(path):
    df = get_dataframe_ASGDS_json(path)
    # print(df.columns)
    col = df['Material and Direct space symmetrys']
    col = col.to_list()
    stat = stat_C1_mirror_point_group(col)
    return stat


def stat_site_C2h(data : list[Atoms_symmetry_group_direct_space]):
    N_C2h = 0
    N_site_total = 0
    for d in data:
        sites_info = d._sites_point_group
        for site in sites_info:
            N_site_total += 1
            if site._site_point_group_Sch == 'C2h':
                N_C2h += 1 

    return N_C2h, N_site_total

def stat_func_from_json(path, func):
    df = get_dataframe_ASGDS_json(path)
    # print(df.columns)
    col = df['Material and Direct space symmetrys']
    col = col.to_list()
    stat = func(col)
    return stat


    


# def pks_structure_non_metal():
#     """
#     return the pks in aiida group structure 2D who is not metal if the band gap calculation was done, and the pks where the band gap calculation wasn't done.
#     """
    
#     ### initial dicts or list of non metal (what we want for now), and the list of non metal but aiida do not calculate the band gap
#     list_pk_non_metal = []
#     list_node_havent_bandsgap = []
#     N_total_analysed = 0
#     for node_struc2D in aiida_group_struc2D.nodes:
#         pk = node_struc2D.pk
#         N_total_analysed += 1
#         band_simples_propretys = band_propretys()
#         band_simples_propretys.from_pk_struc2D(pk)
#         gap_value = band_simples_propretys.band_gap
#         if isinstance(gap_value, float):
#              list_pk_non_metal.append(pk)
#         elif  gap_value == 2j:
#              list_node_havent_bandsgap.append(pk)
#         # t1 = time.time()
        
#     return list_pk_non_metal, list_node_havent_bandsgap



# def compare_CBM_VBM_symmetrie_points_no_metals(pks_non_metal : list[int]):
#     """
#     From pks of structure 2D non metal, return a list concernant 
#     """
    
#     # bands_gap = load_group("band_gap")

    

#     list_bands = {}

#     for pk in pks_non_metal:
#         node_struc2D = load_node(pk)
    
        

        
        
            
        
#             # print("no band gap__________________________________")
#         # print(f"a global boucle : {time.time() - t1}############################################")
#     return list_bands


#### 
def generate_datafram_ASGDS_from_aiida_group(group : Group):
    list_ase_structures = []
    list_pk = []
    for node in group.nodes:
        pk = node.pk
        try:
            ase_struc = node.get_ase()
            list_ase_structures.append(ase_struc)
            list_pk.append(pk)
        except :
            # print(f"node : {node.pk} have no ase structure")
            raise ValueError(f"group containe node at pk: {node.pk} with no ase structure")
        
    dataframe = generate_database_symmetry_directspace_from_ase_Atoms(list_ase_structures, uids = list_pk)
    return dataframe

def generate_datafram_ASGDS_from_pks(pks : list[int]):
    list_ase_structures = []
    list_pk = []
    for pk in pks:
        node = load_node(pk)
        try:
            ase_struc = node.get_ase()
            list_ase_structures.append(ase_struc)
            list_pk.append(pk)
        except :
            # print(f"node : {node.pk} have no ase structure")
            raise ValueError(f"group containe node at pk: {node.pk} with no ase structure")
        
    dataframe = generate_database_symmetry_directspace_from_ase_Atoms(list_ase_structures, uids = list_pk)
    return dataframe



def ASGDS_objet_from_pk_struc2D(pk : int):
    """ 
    from a pk in MC2D aiida group structure 2D, get the Atoms_symmetry_group_direct_space() objet
    """
    pk_in_structure2D = find_nodes_in_group([pk], pks_group_struc2D)
    assert len(pk_in_structure2D) != 0 , 'you are not entering a pk of a node in the group structure 2D'
    node_struc2D = load_node(pk)
    ase_structure = node_struc2D.get_ase()
    direct_symm_objet = Atoms_symmetry_group_direct_space()
    direct_symm_objet.get_site_point_group(ase_structure)
    return direct_symm_objet
    
def score_max_z_site_point_group_IA(objet : Atoms_symmetry_group_direct_space):
    max_z_site_IA = find_max_Z_at_no_centrosymmetric_sites(objet)

def socre_band_gap_from_pk_struc2D(pk : int):
    pk_in_structure2D = find_nodes_in_group([pk], pks_group_struc2D)
    assert len(pk_in_structure2D) != 0 , 'you are not entering a pk of a node in the group structure 2D'
    band_simple_proprety = band_propretys()
    band_simple_proprety.from_pk_struc2D(pk)



def analyse_loop_pks_struc_2D(list_pks_non_metal : list[int] = None):
    """
    analyse who should return the scores of a materiaux have a shape as you cans see in Return:

    Scores:
            1. band gap
            2. max z site point group IA
            3. Number of atomes at unit cell.
            ### as you want to add

    Para :
    list_pks_non_metal :  list[int] , pks of aiida group structure 2D as you specify. Just make sure the score are meaningfal in your pks. Like in pks metals. Band gap have non sense for metals., band gap value will be return None.
                            Ex : You can specify this list as pks correspond to structure. are non metals. usually calculated by pks_structure_non_metal()


                            If list_pks_non_metal is None, means function loop all pks in MC2D aiida group structure 2D.
                                And add two return :
                                                list_pk_non_metal : calculation of bandgap down and the band gap value existe as float.
                                                list_pk_havent_bandgap : pks for whom MC2D didnt do the calculation of bandgap 
    Return :
            tuple of list of scores, 
            tuple of label of each scores
    

    for now, the score are added manually in the source code.
    """
    ###
    pk_list = []
    bands_gap_list = []
    max_Z_site_IA_list = []
    N_atoms_list = []
    list_pks_to_loop = list_pks_non_metal
    if list_pks_non_metal == None :
        list_pks_to_loop = pks_group_struc2D
        list_pk_non_metal_searching = []
        list_node_havent_bandsgap = []

    for pk in list_pks_to_loop :
        ###### creat objet of classes necessary
        ### creat the objet Atoms_symmetry_group_direct_space()
        direct_symm_objet =  ASGDS_objet_from_pk_struc2D(pk)

        #### creat the objet bands_propretys:
        bands_simple_propretys = band_propretys()
        bands_simple_propretys.from_pk_struc2D(pk)
        
        
        


        #### score1 : band gap
        band_gap_value = bands_simple_propretys.band_gap
        if isinstance(band_gap_value, float): ### the first step is select all no metals.
            if list_pks_non_metal == None :
                list_pk_non_metal_searching.append(pk) 
            score_1 = 'band gap (ev)'
            bands_gap_list.append(band_gap_value)
            #### the pk is add if the band gap existe as a float
            pk_list.append(pk)
            
            #### score2 : max Z number who occupie a non centrosymmetric sites
            score_2 = 'max Z at IA site'
            max_Z_site_IA = find_max_Z_at_no_centrosymmetric_sites(direct_symm_objet)
            max_Z_site_IA_list.append(max_Z_site_IA)

            #### score 3 : number atoms in unit cell
            score_3 = 'atomes numbers'
            N_atoms = len(direct_symm_objet._sites_point_group)
            N_atoms_list.append(N_atoms)

            #### add as you want
        elif band_gap_value == 2j and list_pks_non_metal == None :
            list_node_havent_bandsgap.append(pk)
    if list_pks_non_metal == None:
        return (pk_list, bands_gap_list, max_Z_site_IA_list, N_atoms_list), (score_1,score_2,score_3), list_pk_non_metal_searching, list_node_havent_bandsgap
    else :
        return (pk_list, bands_gap_list, max_Z_site_IA_list, N_atoms_list), (score_1,score_2,score_3)


def scores_to_dataframe( scores_lists : tuple[list], scores_label : tuple[str]):
    """
    convert the output from analyse_loop_pks_struc_2D to a panda dataframe as format following:
    Para : 
        score_liste : tuple[list], (pks_list, score_1_list, ... ,score_N_list)
        score_label : tuple[str], (score_1_label, score_2_label, ... , score_N_label)

    pandas Dataframe:
            "chemical formula" <score 1> ... <score N>
    <index>
    pk0
    pk1
    ....

    where index and column index are all strings.
    """

    index = scores_lists[0] ### the pks are index of lines
    dic_data = {}
    for i in range(len(scores_label)):
        score_list_i = scores_lists[i+1]
        score_label = scores_label[i]
        score_serie_i = pd.Series(score_list_i, index=index)
        dic_data[score_label] = score_serie_i
    
    dataframe = pd.DataFrame(dic_data)

    return dataframe


     








if __name__ == '__main__':
    
    # # file_path = '/home/jyin/workspace/test_garbage/test.json'
    # # with open(file_path, "w") as f :
    # #         json.dump(scores_dataframe.to_dict(), f, cls=NpEncoder) ### int64 not JSON serializable. we use the class NpEncoder to convert type compatible to json
    # #         f.close()
    # t1 = time.time()
    # list1, list2 , pks_struc_non_metal, pks_havent_band_gaps = analyse_loop_pks_struc_2D()
    # N_struc2D = aiida_group_struc2D.count()


    # print(f"{N_struc2D} structure 2D analysed")
    # print(f"{len(pks_havent_band_gaps)} structure 2D don't have band gaps")
    # t2 = time.time()
    # delta_t = t2-t1
    # print(f"{delta_t} seconde to find do analyse_loop_pks_struc_2D")

    # scores_dataframe = scores_to_dataframe(list1, list2)

    # file_path = '/home/jyin/workspace/test_garbage/scores.json'
    # scores_dataframe.to_json(file_path)

    # file_path_1 = '/home/jyin/workspace/test_garbage/pks_non_metal.json'
    # with open('data.json', 'w') as f:
    #     json.dump({"non metal" :pks_struc_non_metal, "no calculation band gap":pks_havent_band_gaps}, f)

    pass


# %% 
    import pandas as pd
    path_nonmag_nonmetal = '/home/jyin/workspace/gene_data/pks_struc_2D_diff_cate/pks_non_metal_non_mag.csv'
    pk_nonmag_nonmetal = pd.read_csv(path_nonmag_nonmetal)
# %%
    