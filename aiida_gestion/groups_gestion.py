# %%
# %reload_ext autoreload
# %autoreload 2
# %%
from aiida.orm import QueryBuilder
import aiida.orm as orm
from aiida.engine import run
import aiida.orm
from aiida.orm.groups import Group
from aiida.orm import List, Dict, KpointsData, StructureData, load_code, load_group, load_node, load_computer
from aiida import engine

# from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

# from aiida_quantumespresso.workflows.pw.base import PseudoDojoFamily

from aiida.engine import submit


from aiida.engine import calcfunction, workfunction

from aiida import load_profile, get_profile
if get_profile() == None:
    load_profile()



import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyse_symmetry import Atoms_symmetry_group_direct_space
from band_analyse import *


# %%
def creat_aiida_group(group_name : str):
    command = f"verdi group create {group_name}"
    os.system(command)



def find_non_magnetic_no_metal_struc_into_groups():
    """
    creat a group with name 'non_magnetic_struc' and a group 'non_metal_struc' if this aiida group does not exist yet
    then find all non magnetic structures in structures_2D group and add them into 'non_magnetic_struc' group

    """

    struc_2D = load_group('structure_2D')
    try:
        load_group('non_magnetic_structures')
    except:
        creat_aiida_group('non_magnetic_structures')
    non_magnetic_struc = load_group('non_magnetic_structures')

    try:
        load_group('non_metal_structures')
    except:
        creat_aiida_group('non_metal_structures')
    non_metal_struc = load_group('non_metal_structures')
    i = 0
    for node in struc_2D.nodes:
        print(f'the compound {i} up to 2759')
        uuid = node.uuid
        band_analyser = band_propretys()
        if type(node) == orm.StructureData:
            band_analyser.from_pk_struc2D(node.pk)
            is_magnetic = band_analyser.is_magnetic
            is_metal = band_analyser.band_gap is None


            if not is_magnetic and is_magnetic is not None:

                non_magnetic_struc.add_nodes(node)
                print(f"add structure {node.pk} {uuid}into non_magnetic_struc group")
            if not is_metal:
                non_metal_struc.add_nodes(node)
                print(f"add structure {node.pk} {uuid}into non_metal_struc group")




def groups_R_D_from_aiidagroup(group_name : str = 'structure_2D'):
    """
    select polar, non_centrosymetric_non_Polar and centrosymetric structures from an aiida group and add them into 3 different sub aiida groups
    """
    struc_2D = load_group(group_name)
    try:
        load_group(f'{group_name}/Polar_structures')
    except:
        creat_aiida_group(f'{group_name}/Polar_structures')
    Rashba_group = load_group(f'{group_name}/Polar_structures')

    try:
        load_group(f'{group_name}/non_centrosymetric_non_Polar_structures')
    except:
        creat_aiida_group(f'{group_name}/non_centrosymetric_non_Polar_structures')
    Dresselhauss_group = load_group(f'{group_name}/non_centrosymetric_non_Polar_structures')

    try :
        load_group(f'{group_name}/Centrosymetric_structures')
    except:
        creat_aiida_group(f'{group_name}/Centrosymetric_structures')
    Centrosymetric_group = load_group(f'{group_name}/Centrosymetric_structures')

    for node in struc_2D.nodes:
        uuid = node.uuid
        if type(node) == orm.StructureData:
            asgds = Atoms_symmetry_group_direct_space()
            asgds.get_site_point_group(node.get_ase())
            asgds._uuid = uuid
            if asgds._pg_polar:
                Rashba_group.add_nodes(node)
                print(f"add structure {node.pk} {uuid} into Polar_structures group")
            elif (not asgds._pg_centrosymmetric) and (not asgds._pg_polar):
                Dresselhauss_group.add_nodes(node)
                print(f"add structure {node.pk} {uuid} into non_centrosymetric_non_Polar_structures group")
            elif asgds._pg_centrosymmetric:
                Centrosymetric_group.add_nodes(node)
                print(f"add structure {node.pk} {uuid} into Centrosymetric_structures group")



def find_common_node_2_aiidagroups(group1_name: str, group2_name: str):
    """
    find common nodes in 2 aiida groups using QueryBuilder
    """
    group1 = load_group(group1_name)
    group2 = load_group(group2_name)

    common_nodes = {}
    for node1 in group1.nodes:
        uuid = node1.uuid
        qb = QueryBuilder()
        from aiida.orm import Group
        qb.append(Group, filters={'label': group2_name}, tag='group')
        qb.append(type(node1), filters={'id': node1.pk}, with_group='group')
        if qb.count() > 0:
            common_nodes[uuid] = node1
    return common_nodes



def query_one_node_in_group(node_id : str, group_name : str):
    """
    query one node in an aiida group using QueryBuilder
    """
    node = load_node(node_id)
    # group = load_group(group_name)
    qb = QueryBuilder()
    from aiida.orm import Group
    qb.append(Group, filters={'label': group_name}, tag='group')
    qb.append(type(node), filters={'id': node.pk}, with_group='group')
    if qb.count() > 0:
        return True
    else:
        return False

def query_list_nodes_in_group(node_ids : Iterable, group_name : str):
    """
    query a list of nodes in an aiida group using QueryBuilder

    Different to find_common_node_2_aiidagroups function, this function is from a list of node ids, not from an aiida group.
    """
    nodes_in_group = {}
    for node_id in node_ids:
        if isinstance(node_id, str or int):
            node = load_node(node_id)
            uuid = node.uuid
            if query_one_node_in_group(uuid, group_name):
                nodes_in_group[uuid] = node
    return nodes_in_group
###### Find specific structures propreties function, like janus, AB

def query_node_with_label(label_name : str):
    """
    query all nodes with a specific label using QueryBuilder
    """
    qb = QueryBuilder()
    qb.append(orm.Node, filters={'label': label_name}, tag='node')
    nodes = {}
    for node in qb.all(flat=True):
        nodes[node.uuid] = node
    return nodes




def find_Janus_materiaux(data :StructureData):
    """
    Janus materiaux have a polar group.

    Conditions :
    - polar point groupe (except C1 and Cs)
    - at least 2 different elements
    - the polar axis along z direction
    - the top and bottom atomic layers have each other one chemical element and they are different between them
    """
    if not isinstance(data, StructureData):
        raise TypeError("data must be an instance of StructureData")
    
    elements = set(data.get_ase().get_atomic_numbers())

    if len(elements) < 2:
        return False
    
    elif len(elements) >= 2:
        asgds = Atoms_symmetry_group_direct_space()
        ase_objet = data.get_ase()
        asgds.get_site_point_group(ase_objet)
        if asgds._pg_polar and (asgds._point_group_Sch not in ['C1', 'Cs']):
            if asgds._principal_axis[0] @ np.array([0,0,1]).T > 1e-3: # to ensure that the polar axis is along z direction
                # atom_layers_z = ase_objet.get_positions()[:,2]
                num_atom_layer = len(set(ase_objet.get_positions()[:,2].round(2)))
                if num_atom_layer >= 3: ## i am not sure to how many atomic layer should a Janus materiaux have.
                    Z_position = np.hstack((ase_objet.get_atomic_numbers().reshape(-1,1), ase_objet.get_positions()))
                    Z_position = Z_position.round(2)
                    z_values = Z_position[:, 3]
                    min_z = np.min(z_values)
                    max_z = np.max(z_values)

                    # Indices with minimum z value
                    min_z_indices = np.where(z_values == min_z)[0]
                    min_z_values = Z_position[min_z_indices, 0]

                    # Indices with maximum z value
                    max_z_indices = np.where(z_values == max_z)[0]
                    max_z_values = Z_position[max_z_indices, 0]

                    
                    Z_at_bottom = set(min_z_values) ## all chemical elements at the bottom layer
                    Z_at_upon = set(max_z_values)    ## all chemical elements at the top layer
                    if len(Z_at_bottom) == 1 and len(Z_at_upon) == 1 and Z_at_bottom != Z_at_upon:
                        return True
                    else:
                        return False
                else:
                    return False

                # else:
                #     return False
            else :
                return False
        else:
            return False


def find_polar_AB_materiaux(data : StructureData):
    """
    AB materiaux have less than two elements and polar point groupe (except C1 and Cs). 

    Conditions :
    - less than 2 elements
    - polar point groupe (except C1 and Cs)
    - 2 atomic layers (to ensure that the structure is really AB)
    - the polar axis along z direction
    """
    if not isinstance(data, StructureData):
        raise TypeError("data must be an instance of StructureData")
    # asgds = Atoms_symmetry_group_direct_space()
    # asgds.get_site_point_group(data.get_ase())
    
    elements = set(data.get_ase().get_atomic_numbers())
    if len(elements) > 2:
        return False
    elif len(elements) <= 2:
        asgds = Atoms_symmetry_group_direct_space()
        ase_objet = data.get_ase()
        asgds.get_site_point_group(ase_objet)
        if asgds._pg_polar and (asgds._point_group_Sch not in ['C1', 'Cs']):
            if asgds._principal_axis[0] @ np.array([0,0,1]).T > 1e-3: # to ensure that the polar axis is along z direction
                if len(set(ase_objet.get_positions()[:,2].round(2))) == 2: # to ensure that Having 2 atomic layers
                    return True
                else:
                    return False
            else :
                return False
        else:
            return False
    

# %%
if __name__ == "__main__":
    pass
# %% 
#     find_non_magnetic_no_metal_struc_into_groups()


# # %%
    Dict_non_mag_non_metal = find_common_node_2_aiidagroups('non_magnetic_structures', 'non_metal_structures')
# %%
    polar_non_mag_non_metal = query_list_nodes_in_group(Dict_non_mag_non_metal.keys(), 'structure_2D/Polar_structures')
# %%
    # dic = List(list(polar_non_mag_non_metal.keys()))
    # dic.label = 'non_magnetic_non_metal_polar_structures'
    # dic.store()

# %%
    nodes = query_node_with_label('non_magnetic_non_metal_polar_structures')
    if nodes.__len__() == 1:
        polar_non_mag_non_metal = nodes[list(nodes.keys())[0]].get_list()
    else:
        raise ValueError("There should be only one node with the label 'non_magnetic_non_metal_polar_structures'")

# %% AB materiaux
    AB_mats = []
    for uuid in polar_non_mag_non_metal:
        node = load_node(uuid)
        is_AB = find_polar_AB_materiaux(node)
        if is_AB:
            AB_mats.append(uuid)

# %%  Janus materials
    Janus_mats = []
    for uuid in polar_non_mag_non_metal:
        node = load_node(uuid)
        is_Janus = find_Janus_materiaux(node)
        if is_Janus:
            Janus_mats.append(uuid)
# %% change label existing janus node if exists (than you replace it by another)
    label = 'non_magnetic_non_metal_polar_Janus_structures'
    nodes = query_node_with_label(label)
    if nodes.__len__() == 1:
        existing_janus_node = nodes[list(nodes.keys())[0]]
        existing_janus_node.label = 'deprecated_' + label
        
# %%
    dic_Janus = List(list(Janus_mats))
    dic_Janus.label = 'non_magnetic_non_metal_polar_Janus_structures'
    dic_Janus.store()
# %%
    

    # for uuid in AB_mats:
    #     ase_obj = load_node(uuid).get_ase()
    #     print(ase_obj.get_chemical_formula())
    for uuid in Janus_mats:
        struc_data = load_node(uuid)
        ase_objet = struc_data.get_ase()
        asgds = Atoms_symmetry_group_direct_space()
        asgds.get_site_point_group(ase_objet)
        space_group_HM = asgds._space_group_HM
        chemical_symbols = ase_objet.get_chemical_formula()

        print(f'{chemical_symbols} : {space_group_HM}; {uuid}')



# %%
