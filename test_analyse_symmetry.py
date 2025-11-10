# In[0]
import importlib 
import statistique
import analyse_symmetry
import find_groups
import visualisation_stat
# In[]
importlib.reload(analyse_symmetry)
from analyse_symmetry import *
# In[]
importlib.reload(statistique)
from statistique import *
# In[]
importlib.reload(find_groups)
from find_groups import *

# In[]
importlib.reload(visualisation_stat)
from visualisation_stat import plot_brillouin_zone
# In[0]
import pandas as pd
import json 
import pickle as pkl
# In[0]

# In[0]
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

def test_list_all_space_group():
    all_space_group = list_all_space_group()
    print(all_space_group)
    for i in range(all_space_group.shape[0]):
        print(all_space_group.iloc[i, :])

def test_is_group_polar_centrosymmetric_chiral():
    polar, centrosymmetric, chiral = is_group_polar_centrosymmetric_chiral('432')
    print(f"{polar}, {centrosymmetric}, {chiral}")



def test_generate_database_symmetry_directspace_from_cifs_of_diff_materials():
    # path_json = '/home/jyin/workspace/gene_data/direct_space_data.json'
    path_json = '/home/jyin/workspace/test_garbage/test.json'
    database = generate_database_symmetry_directspace_from_cifs_of_diff_materials()

    save_dataframe_ASGDS_json(database,path_json)
    



def test_get_dataframe_ASGDS_json():
    # path_json = '/home/jyin/workspace/test_garbage/test.json'
    path_json = '/home/jyin/workspace/gene_data/direct_space_data.json'
    dd = get_dataframe_ASGDS_json(path_json)
    print(dd)
    

# In[]

    
    



if __name__ == "__main__":
    pass

    # test_list_all_space_group()
    # test_is_group_polar_centrosymmetric_chiral()
    # test_generate_database_symmetry_directspace_from_cifs_of_diff_materials()
    
    # test_get_dataframe_ASGDS_json()

    #### test if a pandas.Dataframe can take a class that i 've created. and test if json can save pandas dataframe or class.
    # aa = Atoms_symmetry_group_direct_space()
    # aa._formula = 'H2O'
    # aa._space_group_HM = 123
    # datas = pd.DataFrame({"uid" : [445],
    #                     "Material and Direct space symmetrys" : [aa]
    #                                             })

    # print(datas)
    # print(aa)

    # path = '/home/jyin/workspace/test_garbage/test.json'

    # with open(path,'w') as file:
    #     json.dump(aa, file)


    #### test if we can read the class saved in the csv.
    # path_csv = '/home/jyin/workspace/test_garbage/test_save_data.csv'
    # data = pd.read_csv(path_csv)
    # print(data)
    # symmetry_1_mat = data.iloc[0]
    # symmetry_1_mat = symmetry_1_mat.loc['Material and Direct space symmetrys']
    # space_group = symmetry_1_mat._space_group_HM
    # print(space_group)

    ### 
    # g = Atoms_symmetry_group_direct_space()
    # g._formula = 100

    # f = Atoms_symmetry_group_direct_space(formula=89)
    # generate_datafram_ASGDS_from_aiida_group(aiida_group_struc2D)
# In[1]:
    importlib.reload(analyse_symmetry)
    from analyse_symmetry import *
    node_LuIO = load_node('0a8f8fd6-b4f5-4a52-8b26-681c3afb1cf9')
    ase_structure = node_LuIO.get_ase()
    asgde_objet_LuIO = Atoms_symmetry_group_direct_space()
    asgde_objet_LuIO.get_site_point_group(ase_structure)
    # t1 = time.time()
    Z_wp_non_null, pairs_null, Z_wp_sector_pair = get_dipole_sum_for_Cn_bigger_2(asgde_objet_LuIO)
    # print(time.time() - t1)
    print(Z_wp_non_null)
    print(pairs_null)
    print(Z_wp_sector_pair)

# In[2]:
    importlib.reload(analyse_symmetry)
    from analyse_symmetry import *
    tuples_list = [
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
        (np.array([7, 8, 9]), np.array([10, 11, 12])),
        (np.array([4, 5, 6]), np.array([1, 2, 3])),  # Duplicate based on z components
        (np.array([13, 14, 15]), np.array([16, 17, 18])),
        (np.array([10, 11, 12]), np.array([7, 8, 9]))  # Duplicate based on z components
    ]
    pri = find_pairs(tuples_list)
    print(pri)
# In[]
    node_MoS2_6 = load_node('d25962ea-3c39-4e86-9333-2baf8f31a1a8')
    ase_obj_MoS2_6 = node_MoS2_6.get_ase()

    bra_latt_MoS2_6 = ase_obj_MoS2_6.cell.get_bravais_lattice()
    MoS2_6_special_kpoints = bra_latt_MoS2_6.get_special_points_array()

# In[]
    cell_MoS2_6 = ase_obj_MoS2_6.get_cell()
    X_kpoint = MoS2_6_special_kpoints[2] ### it should invariant under rotation 120 degres

# In[]
    rot_120 = set_a_rotation_mat(120)
    reci_abc_MoS2_6 = get_reciprocal_basis_vectors(cell_MoS2_6)
    rot_120_scaled_recip = change_matrix_basis_from_eucli_to_frac(rot_120, reci_abc_MoS2_6)

# In[]
    affin_rot_120 = expand_rot_mat_to_affine_mat(rot_120)
    op_120_objet = SymmOp(affine_transformation_matrix=affin_rot_120)
    is_X_inva_120 = is_kpoint_invariant_under_op(op=op_120_objet, kpoint=X_kpoint, reciprocal_basis_vectors=reci_abc_MoS2_6)
# %%
    spg_num_MoS2_6 = get_space_point_group_spglib(ase_obj_MoS2_6)[0][1]
    pg_MoS2_6 = PointGroup(Group(spg_num_MoS2_6).point_group)
    X_kpoint_pg = find_point_group_of_kpoint(X_kpoint, 
                               reciprocal_basis_vectors=reci_abc_MoS2_6, 
                               primitive_cell=np.transpose(cell_MoS2_6), 
                               pg=pg_MoS2_6)
    
# In[]
    plot_brillouin_zone(reci_abc_MoS2_6[0], reci_abc_MoS2_6[1], reci_abc_MoS2_6[2], view="oblique")


# In[] #### exemple of LuIO
    node_LuIO = load_node('0a8f8fd6-b4f5-4a52-8b26-681c3afb1cf9')
    ase_obj_LuIO = node_LuIO.get_ase()
    cell_LuIO = ase_obj_LuIO.get_cell()
    reci_abc_LuIO = get_reciprocal_basis_vectors(cell_LuIO)
    plot_brillouin_zone(reci_abc_LuIO[0], reci_abc_LuIO[1], reci_abc_LuIO[2])
# In[]
    spg_num_LuIO = get_space_point_group_spglib(ase_obj_LuIO)[0][1]
    pg_LuIO = PointGroup(Group(spg_num_LuIO).point_group)
# In[]
    bra_latt_LuIO = ase_obj_LuIO.cell.get_bravais_lattice()
    LuIO_special_kpoints = bra_latt_LuIO.get_special_points()
# In[]
    find_point_group_of_kpoint(LuIO_special_kpoints['R'], reci_abc_LuIO,np.transpose(cell_LuIO),pg_LuIO)



# In[]
    """
    ## just test why the fucking monoclinc 2D material is not working.
    ## Answer : cause the basis is in standard cell but not in primitive cell.
    """

# In[]
    node_Hg4Mo2O8 = load_node('daaee44e-78b0-4002-85c9-25f55fcf3b31')
    ase_obj_Hg4Mo2O8 = node_Hg4Mo2O8.get_ase()
    cell_Hg4Mo2O8 = ase_obj_Hg4Mo2O8.get_cell()
    reci_abc_Hg4Mo2O8 = get_reciprocal_basis_vectors(cell_Hg4Mo2O8)
    plot_brillouin_zone(reci_abc_Hg4Mo2O8[0], reci_abc_Hg4Mo2O8[1], reci_abc_Hg4Mo2O8[2])
# %%
    asgds_obj_Hg4Mo2O8 = Atoms_symmetry_group_direct_space()
    asgds_obj_Hg4Mo2O8.get_site_point_group(ase_obj_Hg4Mo2O8)
# In[]
    asgds_obj_MoS2_6 = Atoms_symmetry_group_direct_space()
    asgds_obj_MoS2_6.get_site_point_group(ase_obj_MoS2_6)
    symm_data_MoS2_6 = asgds_obj_MoS2_6.get_spglib_symmetry_data()
# In[]
    # S1 = asgds_obj_MoS2_6.get_scaled_positions()[2]
    rots = symm_data_MoS2_6.rotations
    trans = symm_data_MoS2_6.translations
    affines = affine_from_rot_and_trans(rots, trans)
    # operate_to_unit_cell(S1,rot_6)
    all_remain_same_after_some_op(asgds_obj_MoS2_6.get_ase_objet(), affines)
# In[]
    symm_data_Hg4Mo2O8 = asgds_obj_Hg4Mo2O8.get_spglib_symmetry_data()
    rots = symm_data_Hg4Mo2O8.rotations
    trans = symm_data_Hg4Mo2O8.translations
    affines = affine_from_rot_and_trans(rots, trans)
    all_remain_same_after_some_op(asgds_obj_Hg4Mo2O8.get_ase_objet(), affines)

# In[]
    all_asgds_list = df_asgds_pks['Material and Direct space symmetrys'].to_list()
# In[]
    index_wrong_spg = []
    for i, asgds in enumerate(all_asgds_list):
        symm_data = asgds.get_spglib_symmetry_data()
        rots = symm_data.rotations
        trans = symm_data.translations
        affines = affine_from_rot_and_trans(rots, trans)
        is_spg_correct = all_remain_same_after_some_op(asgds.get_ase_objet(), affines)
        if not is_spg_correct:
            pk = df_asgds_pks.index[i]
            index_wrong_spg.append(pk)

# In[]
    ### correct the principal axis's coordinate basis (from std to primitive)
    """
    it's already corrected and storaged in the new_asgds.json
    """
    
    
# In[]
    df_new_asgds = generate_datafram_ASGDS_from_aiida_group(load_group("structure_2D"))
    
# In[]
    
# In[]
    path_new_asgds = '/home/jyin/workspace/gene_data/new_asgds.json'
    save_dataframe_ASGDS_json(df_new_asgds, path_new_asgds)
# In[]
    # new_all_asgds_list = df_new_asgds.iloc[:,0].to_list()
# In[]
    ### check if the site symmetry is correctly find.
    asgds_site_wrong = []
    for asgds in all_asgds_list:
        all_correctly_find = True
        for site in asgds._sites_point_group:
            site_symm_ita = site._site_symmetry_ITA
            site_symm_hm = site._site_point_group_HM
            correctely_find = bi_check_if_string_disorded(site_symm_ita, site_symm_hm)
            all_correctly_find = correctely_find and all_correctly_find
        if not all_correctly_find:
            asgds_site_wrong.append(asgds)
                
# In[]
    pairs_wrong = []
    for asgds in asgds_site_wrong:
        for site in asgds._sites_point_group:
            site_symm_ita = site._site_symmetry_ITA
            site_symm_hm = site._site_point_group_HM
            correctely_find = bi_check_if_string_disorded(site_symm_ita, site_symm_hm)
            if not correctely_find :
                pair = (site_symm_ita, site_symm_hm)
                pairs_wrong.append(pair)
# In[]
    set_pairs_wrong = set(pairs_wrong)
    
# In[]
    pos_except = np.array([0.7493387458730502, 2.1671378965303335e-05, -1.0506858766708962e-07])
    I = np.identity(4)
    I_ = expand_rot_mat_to_affine_mat(-1 * np.identity(3))
    symmetry_related(pos_except, pos_except, I)
# In[] save corrected asgds data. Already Done
    
    # save_dataframe_ASGDS_json(df_asgds_pks, asgds_json_path)

# In[]
    ### look at into some example of site symmetry and see why the site symmetry is not correctly found.
    asgds_eg = asgds_site_wrong[0]
# In[]
    #### if the site symmetry symbol is same to site point group symbol.
    spg_HM = asgds_eg._space_group_HM
    spglib_data = asgds_eg.get_spglib_symmetry_data()
    sites_info = asgds_eg._sites_point_group
    print(f"N atoms {len(sites_info)}")
    for site in sites_info:
        # if not bi_check_if_string_disorded(site._site_symmetry_ITA, site._site_point_group_HM):
        if site._site_symmetry_ITA != site._site_point_group_HM:
            print(f"{site._Z_number} : pos scaled : {site._scaled_position} , site_symm_ita : {site._site_symmetry_ITA} , site_symm_HM : {site._site_point_group_HM}")
            print(f'principal axis : {site._principal_axis}')
            print(f'order of principal axis: {site._order_max}')
            print(f'type of principal axis: {site._type_principal_axis}')

            print('\n')

# In[]
    rotations = spglib_data.rotations
    for rot in rotations:
        axis, det, angle = find_rotation_axis(rot)
        if det == -1 and angle == np.pi:
            print(rot)
            print(f"proprety : {axis} , {det} , {angle}")

# In[]
    
    all_remain_same_after_some_op(asgds_eg.get_ase_objet(), 
                                  affine_from_rot_and_trans(rotations, spglib_data.translations))

# In[]
#### find some C2 principal axis of crystal in plane.
    # asgds_C2_in_plane = []
    # for asgds in all_asgds_list:
    #     # p_axis = asgds._principal_axis
    #     # axis = p_axis[0]
        
    #     # order = p_axis[1]
    #     # type_axis = p_axis[2]
    #     # if type_axis == 'C2.0 proper':
    #     #     if axis[2] == 0:
    #     #         asgds_C2_in_plane.append(asgds)
    #     _ = if_C2_proper_inplan__centro_ornot(asgds)


# In[]
    #### List some exemple if the site symmetry is non symmorphic.
    asgds_wear_site_symm = {}
    for i, asgds in enumerate(df_new_asgds.iloc[:,0]):
        pk = df_new_asgds.index[i]
        if not asgds._site_symm_all_symmorphic :
            asgds_wear_site_symm[pk] = asgds
            
# In[]
    for pk in asgds_wear_site_symm.keys():
        asgds = asgds_wear_site_symm[pk]
        number_atoms = len(asgds._sites_point_group)
        if number_atoms <=5:
            print(f"pk : {pk}")
            print(f"formula : {asgds._formula}")
            print(f"space group : {asgds._space_group_HM}")
            print(f"site symm : {asgds._site_symm_all_symmorphic}")
            print('\n')
    
# In[]
    for site in df_new_asgds.iloc[1290,0]._sites_point_group:
        print(site._site_symmetry_ITA)
        print(site._site_point_group_HM)
        print('\n')


# In[] ### test if site symmetry have non symmorphic operations, were they transform the site out of the unit cell?
    sites_non_symmorphic_out = {}
    for pk in df_new_asgds.index:
        asgds = df_new_asgds.loc[pk,'Material and Direct space symmetrys']
        sites = asgds._sites_point_group
        sites_in_question = []
        for site in sites:
            if site._have_non_symmorphic_op and site._out_unit_cell_by_non_symmorphic_op: 
                sites_in_question.append(site)
                sites_non_symmorphic_out[pk] = sites_in_question
                

# In[]
    asgds_AgTe = df_new_asgds.loc[188704,'Material and Direct space symmetrys']
    asgds_O16Tl2V6 = df_new_asgds.loc[190758,'Material and Direct space symmetrys']
# In[]

def print_some_site_ops(asgds) :
    spg_df = asgds.get_spglib_symmetry_data()
    affines = affine_from_rot_and_trans(spg_df.rotations, spg_df.translations)
    for site in asgds._sites_point_group:
        if site._out_unit_cell_by_non_symmorphic_op:
            print(site._Z_number)
            print(site._site_point_group_HM)
            print(site._scaled_position)
            for affine in affines :
                
                print(symmetry_related(site._scaled_position, site._scaled_position, affine))
                print(affine)
                print('\n')

# In[]
# print_some_site_ops(asgds_AgTe)
    print_some_site_ops(asgds_O16Tl2V6)

# In[]
    Ag_pos = np.array([0.5, 0, 0.5])
    Ag_pos_tar = operate_to_unit_cell(Ag_pos, affine_matrix=affines_AgTe[0])

# In[]
    spg_df_O16Tl2V6 = asgds_O16Tl2V6.get_spglib_symmetry_data()

# In[]
    dict_non_symmetry_out_plan = {}
    dict_dipole_in_plan = {}
    for pk in df_new_asgds.index:
        asgds = df_new_asgds.loc[pk,'Material and Direct space symmetrys']
        _ = if_C2_proper_inplan__centro_ornot(asgds)
        if _ is not None:
            axis_in_plan, centro, polar = _
            if axis_in_plan and not centro and not polar:
                dict_non_symmetry_out_plan[pk] = asgds
            
            if axis_in_plan and not centro and polar:
                dict_dipole_in_plan[pk] = asgds


# In[]
#### This is a thesis who say we can have transition from 3D zinc blende structure to 2D D2d layer and allow the Dresselhaus term in 2D. Just find some
    dict_D2d_mat = {}
    for pk in df_new_asgds.index:
        asgds = df_new_asgds.loc[pk,'Material and Direct space symmetrys']
        pg  = asgds._point_group_Sch
        if pg == 'D2d':
            dict_D2d_mat[pk] = asgds

# In[]
    dict_Rashba_inplan = {}
    for pk in df_new_asgds.index:
        asgds = df_new_asgds.loc[pk,'Material and Direct space symmetrys']
        pg = asgds._point_group_HM
        polar ,_ , __ = is_group_polar_centrosymmetric_chiral(pg)
        if polar:
            principal_axis = asgds._principal_axis
            if principal_axis[0] != [None]:
                x, y , z = principal_axis[0]
                
                if x == 0 and y == 0 and z != 0:
                    dict_Rashba_inplan[pk] = asgds

# %%
    
    asgds_json_path = '/home/jyin/workspace/gene_data/new_asgds.json'
    df_new_asgds = get_dataframe_ASGDS_json(asgds_json_path)
# %%
    dict_Z_max_polaire = {}
    for pk in df_new_asgds.index:
        asgds = df_new_asgds.loc[pk,'Material and Direct space symmetrys']
        Z_max_polaire = find_max_Z_at_polaire_principale_axis(asgds)
        if Z_max_polaire is not None:
            dict_Z_max_polaire[pk] = Z_max_polaire



















# In[] read file non mag semicond
    file_path = '/home/jyin/workspace/gene_data/pks_struc_2D_diff_cate/pks_non_metal_non_mag.csv'
    liste_pk_non_mag_semi = pd.read_csv(file_path).iloc[:,0].tolist()
# In[]
    # df_asgds_non_mag_semi = generate_datafram_ASGDS_from_pks(liste_pk_non_mag_semi)
# In[]
    path_save = '/home/jyin/workspace/gene_data/asgds_nonmag_semi.json'
# In[]
    save_dataframe_ASGDS_json(df_asgds_non_mag_semi, path_save )
# In[]
    df_asgds_non_mag_semi = get_dataframe_ASGDS_json(path_save)

# In[] dipôle in plan
    def find_dipole_in_plan(df_asgds):
        dict_ = {}
        for pk in df_asgds.index:
            asgds = df_asgds.loc[pk,'Material and Direct space symmetrys']
            polar = asgds._pg_polar
            if polar :
                principal_axis, order, tt = asgds._principal_axis
                type_axis = 'improper' in tt
                if principal_axis[0] is not None and not type_axis:
                    if principal_axis[2] == 0 and (principal_axis[1] != 0 or principal_axis[0] != 0):
                        
                        max_z_IA = find_max_Z_at_no_centrosymmetric_sites(asgds)
                        dict_[pk] = (asgds, max_z_IA)
        return dict_
# In[] dipôle in plan
    def find_dipole_in_plan(df_asgds):
        dict_ = {}
        for pk in df_asgds.index:
            asgds = df_asgds.loc[pk,'Material and Direct space symmetrys']
            polar = asgds._pg_polar
            if polar :
                principal_axis = asgds._principal_axis[0]
                if principal_axis[0] is not None:
                    if principal_axis[2] == 0 and (principal_axis[1] != 0 or principal_axis[0] != 0):
                        
                        max_z_IA = find_max_Z_at_no_centrosymmetric_sites(asgds)
                        dict_[pk] = (asgds, max_z_IA)
        return dict_
# In[] D2d 
    def find_pg(df_asgds, pg_sch : str):
        dict_pg = {}
        for pk in df_asgds.index:
            asgds = df_asgds.loc[pk,'Material and Direct space symmetrys']
            pg = asgds._point_group_Sch
            if pg == pg_sch:
                max_z_IA = find_max_Z_at_no_centrosymmetric_sites(asgds)
                dict_pg[pk] = (asgds, max_z_IA)
        return dict_pg
# In[] ## Max Z at polaire
    def find_max_Z_at_polaire_principale_axis(df_asgds):
        dict_Z_max_polaire = {}
        for pk in df_asgds.index:
            asgds = df_asgds.loc[pk,'Material and Direct space symmetrys']
            Z_max_polaire = find_max_Z_at_polaire_principale_axis(asgds)
            if Z_max_polaire is not None:
                dict_Z_max_polaire[pk] = Z_max_polaire
        return dict_Z_max_polaire

# In[] ## dipole per plan
    def find_dipole_per_plan(df_asgds):
        dict_ = {}
        for pk in df_asgds.index:
            asgds = df_asgds.loc[pk,'Material and Direct space symmetrys']
            polar = asgds._pg_polar
            if polar :
                principal_axis = asgds._principal_axis[0]
                if principal_axis[0] is not None:
                    if principal_axis[2] != 0 and (principal_axis[1] == 0 and principal_axis[0] == 0):
                        
                        max_z_IA = find_max_Z_at_no_centrosymmetric_sites(asgds)
                        dict_[pk] = (asgds, max_z_IA)
        return dict_
# In[] dipole in plan exe
    pk_dipole_in_plan_nonmag_semi = find_dipole_in_plan(df_asgds_non_mag_semi)

# In[] dipole perpendicular to plan exe
    pk_dipole_per_plan_nonmag_semi =  find_dipole_per_plan(df_asgds_non_mag_semi)

# In[] D3h exe
    pk_D3h_nonmag_semi = find_pg(df_asgds_non_mag_semi, 'D3h')
# In[] D2d exe
    pk_D2d_nonmag_semi = find_pg(df_asgds_non_mag_semi, 'D2d')

# In[] find Cs in plan.
    pk_asgds_Cs_in_plan = {}
    for pk in df_asgds_non_mag_semi.index :
        asgds = df_asgds_non_mag_semi.loc[pk,'Material and Direct space symmetrys']
        pg = asgds._point_group_Sch
        if pg == 'Cs':
            principal_axis = asgds._principal_axis[0]
            if principal_axis[0] is not None:
                if principal_axis[2] != 0 and principal_axis[1] == 0 and principal_axis[0] == 0:
                    # print(f"pk : {pk}")
                    # print(f"formula : {asgds._formula}")
                    # print(f"space group : {asgds._space_group_HM}")
                    # print(f"site symm : {asgds._site_symm_all_symmorphic}")
                    # print('\n')

                    pk_asgds_Cs_in_plan[pk] = asgds