from find_groups import *
import numpy as np


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



from aiida.engine import run
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




### Iterable class for list, tuple, dict, set......
from collections.abc import Iterable


#### find the vertex for a convex area
from scipy.spatial import ConvexHull, convex_hull_plot_2d




aiida_group_bands = load_group("bands_2D") ### dans le group band_2D, les BZ sont irreducdible
aiida_group_bands_gap = load_group("band_gap")

aiida_group_struc2D = load_group("structure_2D")

aiida_group_struc2D = load_group("structure_2D")


#### get pks of 2 group to the enter of find_nodes_in_group()
pks_group_struc2D = []
for node in aiida_group_struc2D.nodes:
    pks_group_struc2D.append(node.pk)
    
pks_group_bands_gap = []
for node in aiida_group_bands_gap.nodes:
    pks_group_bands_gap.append(node.pk)

pks_group_bands = []
for node in aiida_group_bands.nodes:
    pks_group_bands.append(node.pk)





"""
analyse bands propretys for a structure 2D given its pk in group structure 2D
"""





 

class band_propretys:
    def __init__(self, 
                 
                 vBM : dict = None,
                 cBM : dict = None,
                 band_gap : float = 1j, #### here the initiation is 1j, means band.band_gap is not written. If it's none, it's a metal. 
                 fermi_energy : dict = 1j, ### same as above
                 is_magnetic : bool = None):
        self.vBM = vBM
        self.cBM = cBM
        self.band_gap = band_gap
        self.is_magnetic = is_magnetic
        self.fermi_energy = fermi_energy
    
    def from_pk_struc2D(self, pk : int):
        """
        from pk in group struc 2D find the band gap and band structures' proprety need in band_propretys.

        Note, if after running this function the attribut band_gap is 2j, that means the band gap is not calculated for the structure 2D with pk in MC2D
        """
        pk_in_structure2D = find_nodes_in_group([pk], pks_group_struc2D)
        
        assert len(pk_in_structure2D) == 1 , f"you are not entering a pk : {pk} of a node in the group structure 2D"
        node_struc2D = load_node(pk)
       
        
        # print(f"struc -> special point : {time.time() - t1}")
        g = Graph()
        g.add_node(node_struc2D)
        g.recurse_descendants(node_struc2D)
        # print(g.nodes)
        

        gap_data_pks = find_nodes_in_group(g.nodes, pks_group_bands_gap)
        
        # print(f"struc -> band gap node : {time.time() - t1}")
        assert len(gap_data_pks) <= 1 , f"for a materiaux, it should just have a pk in bands_gap, get pks of bands_gap : {gap_data_pks} in structure {pk}"
        if len(gap_data_pks) == 1:
            node_gap = load_node(gap_data_pks[0])
            band_gap_data = get_gap_data(node_gap)
            is_metal = band_gap_node_metal_or_not(band_gap_data)

            if is_metal != True:
                

                ase_struc2D = node_struc2D.get_ase()
                Z_numbers = ase_struc2D.get_atomic_numbers()
                N_atoms_unit_cell = np.shape(Z_numbers)[0]

                ####
                # direct_space_symm = Atoms_symmetry_group_direct_space()
                # direct_space_symm.get_site_point_group(ase_struc2D)
                ### get the special points with their labels
                bravais_lattice = ase_struc2D.cell.get_bravais_lattice()
                special_points = bravais_lattice.get_special_points() ### it's a dict, keys are labels
                    
                
                        

                ### start analysing bands data if the crystal is not a metal
                bands_data_pks = find_nodes_in_group(g.nodes, pks_group_bands)
                assert len(bands_data_pks) == 1 , f"for a materiaux, it should just have a set of bands, get pks of bandsdata : {bands_data_pks} in structure {pk}" #### normally there is just a objet of band data for a materiaux. so bands_data_pks should be a list has just 1 value
                # if len(band_gap_data) == 1:
                pk_bands = bands_data_pks[0]
                bands_data_node = load_node(pk_bands)
                # t_b = time.time()
                is_magnetic, fermi_energy, vBM_energy, cBM_energy, vBM_k, cBM_k = find_propretys_bands_data(bands_data_node)
                # print(f"find propretys bands : {time.time() - t_b} s")
                vBM = {'special point' : None, 'k' : vBM_k, 'vBM energy' : vBM_energy}
                cBM = {'special point' : None, 'k' : cBM_k, 'cBM energy' : cBM_energy}
                for sp in special_points:
                    ### the special_points is a dic with keys of label of special point (e.g. 'X')
                    sp_value = special_points[sp]
                    if np.allclose(vBM_k, sp_value):
                        vBM['special point'] = sp
                    
                    if np.allclose(cBM_k, sp_value):
                        cBM['special point'] = sp

                ##### write the attributes        
                self.band_gap = is_metal[0]
                self.is_magnetic = is_magnetic
                self.fermi_energy = fermi_energy
                self.vBM = vBM
                self.cBM = cBM
            else :
                self.band_gap = None #### that mean the pk structure 2D is a metal

        else :
            self.band_gap = 2j ### that mean the pk structure 2D haven't the band gap in MC2D (not calculated)
    

    def to_dict(self):
        dic = {}
        dic['band gap'] = self.band_gap
        dic['vBM'] = self.vBM
        dic['cBM'] = self.cBM
        dic['is magnetic'] = self.is_magnetic
        dic['fermi energy'] = self.fermi_energy
        return dic

    def from_dic(self, dic: dict):
        try:
            self.band_gap = dic['band gap']
            self.vBM = dic['vBM']
            self.cBM = dic['cBM']
            self.is_magnetic = dic['is magnetic']
            self.fermi_energy = dic['fermi energy']
        except:
            print("Error : the dictionary has not correct keys or keys are not complet")
        


        

def get_gap_data(band_gap : CalcFunctionNode ):
    """
    for band_gap objet in group band_gap, all node is type CalcFunctionNode. But the result is store in a data node of type Dict. And i'm not sure which group are these data Node in. Maybe no group contain these node.
    So just find the output who contain a dict with keys like {'band_gap' : ,......}. We just confirme the first key is 'band gap'
    """
    g = Graph()
    g.add_node(band_gap.uuid)
    # g.add_outgoing(band_gap.uuid)
    g.add_outgoing(band_gap.uuid)
    nodes_pk = g.nodes
    # assert len(nodes_pk) > 2 , "why this band gap CalculFuncitonNode have more than one output"
    # pk_band_gap_func = band_gap.pk 
    node_gap_data = []
    for pk in nodes_pk:
        node = g._load_node(pk)
        if type(node) == Dict_aiida:
            dic = node.get_dict()
            first_key = list(dic.keys())[0]
            if first_key == 'band_gap' :
                node_gap_data.append(node)
    
    assert len(node_gap_data) <= 1 , f"why band_gap {band_gap.uuid} (type {type(band_gap)}) have previous output of gaps "
    if len(node_gap_data) == 0:
        return None
    else :
        return node_gap_data[0]

    


def find_nodes_in_group(nodes_to_filtre: Iterable[int] | Iterable[Node], pks_g : list[int]):

    """
    In a list of node. Find the node who belong to a given aiida group g

    we often need to find the node in a set of nodes given by somthing like a Graphe type objet. The type of node is not sufficent to understand what type of calcul or data that this node is. 
    So we should find it from a aiida group define by the greator who give a physical meaning to a set (group) of node.  

    Parameter : nodes_to_filtre : a list of node to compare.
                pk_g : pks : list[int] of a aiida group

    Return :

            pks_in_group : list[int] | [] . aiida pk. if it existe in group, pks_in_group is list of pk, else, pk_in_group is [].
    """


    pks_in_group = []
    # pks_g = []
    # for node in group.nodes:

    #     pks_g.append(node.pk)
    
    pks_g = set(pks_g)
    
    for node in nodes_to_filtre:
        pk_n = node
        # try :
        #     if type(node) == int:
        #         pk_n = node 
        #     else :
        #         pk_n = node.pk
        # except :
        #     print("node_to_filtre should be a iterable of int or aiida Node object")
        if pk_n in pks_g:
            pks_in_group.append(pk_n)
        
    return pks_in_group


def band_gap_node_metal_or_not(bandgap_node : CalcFunctionNode):
    """
    if material is metal, return True. else, return (the value of band gap in ev, if it's insulator)
    """
    
    data = get_gap_data(bandgap_node)
    data = data.get_dict()
    band_gap_value = data['band_gap']
    if band_gap_value != None:
        return band_gap_value, data['is_insulator']
    else :
        return band_gap_value, True
    

def find_propretys_bands_data(bands_data_node : BandsData):
    """
    This funciton is just use on the non metal crystal : SC or insulator.
    here in funciton names, the propretys mean :   
                                                if magnetique?
                                                fermi energy
                                                VBM, 
                                                CBM,
                                                VBM's k points, 
                                                CBM's k points

    Para :
        bands_data_node : BandsData objet
    
    
    Return : description (in order) above
                                                
    """
    
    creator_caljob = bands_data_node.creator
    creator_caljob_output_para = creator_caljob.outputs.output_parameters.get_dict()

    # Create a sub-dictionary with keys containing 'fermi_energy', what we can find are fermi_energy, fermi_energy up, fermi_energy_down
    fermi_energy = {key: value for key, value in creator_caljob_output_para.items() if 'fermi_energy' in key and isinstance(value, float)}
    # fermi_energy = creator_caljob_output_para['fermi_energy']
    is_magnetic = creator_caljob_output_para['number_of_spin_components'] != 1 ### si la valeur est 1, c'est non magnetique.

    bands = bands_data_node.get_bands()
    k_points = bands_data_node.get_kpoints()
    ##### find CBM, VBM and their symmetry
    electron_unit_cell_effective = creator_caljob_output_para['number_of_electrons']
    bands_occupied = electron_unit_cell_effective/2
    
    index_VB = int(bands_occupied) - 1
    vB = bands[:,index_VB]
    cB = bands[:, index_VB + 1]

    vBM = np.max(vB)
    cBM = np.min(cB)

    vBM_k_index = np.argmax(vB)
    cBM_k_index = np.argmin(cB)

    VBM_k = k_points[vBM_k_index]
    CBM_k = k_points[cBM_k_index]


    return is_magnetic, fermi_energy, vBM, cBM, VBM_k, CBM_k 




def find_irreducible_BZ_vertex_arris(k_points : np.ndarray):
    """
    in our database MC2D, or in generaly all the DFT of band structure. the calculation is do in a irreducible Brilliouin zone. 
    In this irreducible brilliouin zone. We can identify the special points on finding its vertex. (for now, not identification of the symmetry element who pass these vertex. But at least point (0,0,0) have all symmetrys of crystal point group)
                                        We can say all k points on the arris except the vertex have just a mirror pass the arris , and by materiaux 2D, perpendicular to the plan xy.
                                        We can say all k point do not belong to the edge have no symmetry element 

    Above, when we say a point have a symmetry element, that means the point is in this symmetry element. 

    we can do this by sorting the k point on x and on y.

    When we look for the arris, we suppose the arris are the max or min ky point for every kx. at the min kx, or max kx, all points are arris.
                                That means we suppose that the edges of the area don't have a rough error numericly. It should be well done when generating MC2D

    Parameter :
            k_points : a ndarray of k point, shape (n, 3)
    Return :
            Arris : ndarray, shape (n,3) : points on arris from left to right(kx), for each kx, from down to up
            vertexs : ndarray, shape (n,3) : points on vertex from left to right(kx), for each kx, from down to up
    """
    
    k_points_sort_by_kx = k_points[k_points[:,0].argsort()]
    shape = np.shape(k_points_sort_by_kx)
    dim_k = shape[1]
    unique_values = np.unique(k_points_sort_by_kx[:,0])
    min_kx = np.min(unique_values)
    max_kx = np.max(unique_values)
    arris_left = None
    arris_right = None
    # For each unique value in kx, sort corresponding rows by ky
    
    
    arris = np.array([]).reshape(0,dim_k)
    for val in unique_values:
        
            
        mask = k_points_sort_by_kx[:,0] == val
        subset = k_points_sort_by_kx[mask]
        sorted_subset = subset[subset[:,1].argsort()]
        
        
        
        

        ### append the point of arris from left to right, from down to up
        if val == min_kx: ### at the left side :  min kx,can have more the 2 points on arris
            arris_left = sorted_subset
            
            for k in arris_left:
                arris = np.vstack((arris, k))
        elif val == max_kx: #### at the right side : max kx, can have more the 2 points on arris
            arris_right = sorted_subset
            
            for k in arris_right:
                arris = np.vstack((arris, k))
        else:
            ### arris at up (max_subset) or down (min_subset) of ky if kx is not min or max.
            arris_up = sorted_subset[-1] 
            arris_down = sorted_subset[0]
            arris =  np.vstack((arris, arris_down))
            arris =  np.vstack((arris, arris_up))
            
        

    
    arris = np.array(arris)


    ### find the vertex directly from the 
    hull_objet = ConvexHull(k_points)
    vertexs = k_points[hull_objet.vertices]

    return vertexs, arris


    




def find_symmetry_of_k_point(special_points):

    pass

