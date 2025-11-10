# %%
from find_groups import *
import numpy as np
import matplotlib.pyplot as mpl
from pyxtal.symmetry import Group, get_point_group
import pyxtal as pyl
import pandas as pd
import json 
import csv 
import spglib

from pymatgen.symmetry.groups import SpaceGroup
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.operations import OperationAnalyzer


from aiida.orm.groups import Group as AiiDAGroup
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, load_node, load_computer



from aiida import load_profile, get_profile
if get_profile() == None:
    load_profile()


# %%
"""
# Module : analyse_symmetry
#### This is a module to analyse the symmetry of a crystal, in direct space and reciprocal space. The main mission is to associate a point group to a atom site or a k point in reciprocal space.
#### The second mission is to determine the asymmetry direction if group is non centrosymmetric and so the polar direction if the group is polar.
####### speically , if the space group is centrosymmetric and existe some site non centrosymmetric. Find the pairs of site with opposite asymmetry direction.
## To do : Juste pairs of site dipole along z is find. Need to find all other case of pairs of site if site are non centrosymmetric.

Note:
--------
    When using the transformation between cartesian and fractional coordinates (in direct or reciprocal space): 
    - The "equivalence" relation between 2 atoms position or k point is not really infect by the precision of the basis vector. But juste the precison of fractional coordinates.
    -- cause the "equivalence" calculation is juste remove the interger part of the fractional coordinates. So the coordinate is in the unit cell.
"""

### find pairs in a list
from itertools import combinations
def find_pairs(lst):
    return list(combinations(lst, 2))

###### analysis in direct space

defaut_path_json = '/home/jyin/workspace/test_garbage/data_direct_space_symmetry.json'
defaut_path_csv = '/home/jyin/workspace/test_garbage/data_direct_space_symmetry.csv'

class Site_point_group : ### short note SPG in all comment or name of variable.
    """
    ._Z_number : atomic number of the atom on the site.
    ._wyckoff_letter : wyckoff letter of the site.
    ._site_symmetry_ITA : site symmetry symbol in ITA.
    ._site_point_group_HM : the point group correspond to site symmetry with notation HM.
    ._site_point_group_Sch : same with notation Sch.
    ._order_max : order of principal axis
    ._principal_axis : list (3,3) of principal axis. initalized by []. if order max is 1, this attribue is None.
    ._polar : is the point group polar ?
    ._centrosymmetric : is the point group centrosymmetric ?
    ._chiral : is the point group chiral ?

    """
    
    def __init__(self, 
                 Z_number : int = None, ### use Optional to define the enter type and make the default value to None
                 scaled_position : list = None, 
                 wyckoff_letter : str = None, 
                 site_symmetry_ITA : str = None, 
                 site_point_group_HM : str = None , 
                 site_point_group_Sch : str = None,
                 order_max : int = None,
                 principal_axis : list = None,
                 type_principal_axis : str = None,
                 polar : bool = None, 
                 centrosymmetric : bool = None,
                 chiral : bool = None ,
                 have_non_symmorphic_op : bool = None,
                 was_out_of_unit_cell_with_non_symmorphic_op : bool = None):
        
        self._Z_number = Z_number
        self._scaled_position = scaled_position
        self._wyckoff_letter = wyckoff_letter
        self._site_symmetry_ITA = site_symmetry_ITA
        self._site_point_group_HM = site_point_group_HM
        self._site_point_group_Sch = site_point_group_Sch
        self._polar = polar
        self._centrosymmetric = centrosymmetric
        self._chiral = chiral
        self._order_max = order_max
        self._principal_axis = principal_axis
        self._type_principal_axis = type_principal_axis
        self._have_non_symmorphic_op = have_non_symmorphic_op
        self._out_unit_cell_by_non_symmorphic_op = was_out_of_unit_cell_with_non_symmorphic_op
        ## site symmetry notation in international table for crystallography

    ### convertion to dictionnary simplifit the storage of data into some file format like csv and json.
    def to_dict(self):
        dict = {"Z number" : self._Z_number,
                "scaled position" : self._scaled_position,
                "wyckoff letter" : self._wyckoff_letter,
                "site symmetry ITA notation" : self._site_symmetry_ITA,
                "site point group HM notation" : self._site_point_group_HM,
                "site point group Sch notation" : self._site_point_group_Sch,
                "order of principal axis" : self._order_max,
                "principal axis" : self._principal_axis,
                "type principal axis" : self._type_principal_axis,
                "polar" : self._polar,
                "centrosymmetric" : self._centrosymmetric,
                "chiral" : self._chiral,
                "have non symmorphic op" : self._have_non_symmorphic_op,
                "was out of unit cell with non symmorphic op" : self._out_unit_cell_by_non_symmorphic_op}
        return dict

    def from_dic(self, dict):
        try :
            self._Z_number = dict['Z number']
            self._scaled_position = dict['scaled position']
            self._wyckoff_letter = dict['wyckoff letter']
            self._site_symmetry_ITA = dict['site symmetry ITA notation']
            self._site_point_group_HM = dict['site point group HM notation']
            self._site_point_group_Sch = dict['site point group Sch notation']
            self._order_max = dict['order of principal axis']
            self._principal_axis = dict['principal axis']
            self._type_principal_axis = dict['type principal axis']
            self._polar = dict['polar']
            self._centrosymmetric = dict['centrosymmetric']
            self._chiral = dict['chiral']
            self._have_non_symmorphic_op = dict['have non symmorphic op']
            self._out_unit_cell_by_non_symmorphic_op = dict['was out of unit cell with non symmorphic op']
        except :
            print("it's not the correct format of dictionnary to create the class")

    def is_polar_centro_chiral(self):
        polar , centro, chiral = is_group_polar_centrosymmetric_chiral(self._site_point_group_HM)
        return polar, centro, chiral






class Atoms_symmetry_group_direct_space: ### short note ASGDS in all comments or name of variable.
    """
    ._formula : chemical formula of the crystal.
    ._uuid : uuid of the crystal in database MC2D.
    ._cell_basis_vectors : crystal basis vector. array([a,b,c]) where a, b , c are basis vector
    ._space_group_HM : space group of crystal with notation HM.
    ._point_group_HM : point group of crystal with notation HM.
    ._site_point_group : list of class Site_point_group 
    ._principal_axis : [principal axis, order, type] principal axis in _cell_basis_vectors's basis, the axis of the polar vector.  
                                                But notice that the polar axis have no signe +-.
    ._prim_std_trans : the transformation matrice from init ase's primitive cell (attribue ._cell_basis_vectors) to the standard cell defined in spglib. :https://spglib.readthedocs.io/en/stable/definition.html#rotation-introduced-by-idealization 
    ._pg_polar : is the point group polar ?
    ._pg_centrosymmetric : is the point group centrosymmetric ?
    ._pg_chiral : is the point group chiral ?
    ._site_symm_all_symmorphic : is all site in the crystal have symmorphic site symmetry ?

    Note : the cell basis vector is in the form of array([a,b,c]) where a, b, c are the basis vector of the crystal.
        
    ## 
    """
    
    def __init__(self, 
                 
                 formula : str = None, ## chemical formula
                 cell_basis_vectors : np.ndarray = None, 
                 uuid : str = None,
                 space_group_HM : str = None,
                 point_group_HM : str = None,
                 point_group_Sch : str = None,
                 sites_point_group : list[Site_point_group] = None,
                 prim_std_trans : np.ndarray = None,
                 site_symm_all_symmorphic : bool = None,):
        
        self._formula = formula
        self._cell_basis_vectors = cell_basis_vectors
        self._space_group_HM = space_group_HM
        self._point_group_HM = point_group_HM
        self._point_group_Sch = point_group_Sch
        self._sites_point_group = sites_point_group
        self._prim_std_trans = prim_std_trans
        self._site_symm_all_symmorphic = site_symm_all_symmorphic
        self._uuid = uuid
        
        
    
        

    # def get_space_point_group(self, atoms : Atoms):
    #     """
    #     fill ._formula, ._space_group_HM, ._point_group_HM, ._point_group_Sch, ._cell_basis_vectors attribute from a ase.atoms.Atoms object.
    #     """
    #     self._formula = atoms.symbols.get_chemical_formula()
    #     space_group, point_group, site_wyckoff_symmetry, abc_vector = get_space_point_group_spglib(atoms)
    #     self._space_group_HM = space_group[0]
    #     self._point_group_HM = point_group
    #     self._point_group_Sch = point_group_hm_sch[point_group]
    #     self._cell_basis_vectors = abc_vector
    
    def get_site_point_group(self,atoms : Atoms):
        
        """
        get information of space group, point group, crystal vector of crystal
        get information of site symmetry (also site point group) and classifie them into polar, centrosymmetric and chiral. 
        Then fill the attribule of the self.

        
        """
        space_group, point_group, site_wyckoff_symmetry, abc_vector, all_sites_info, spg_ops_scaled, prim_std_trans, have_non_symmorphic_site_op = site_point_group_from_a_material(atoms)
        # space_group, point_group, site_wyckoff_symmetry, abc_vector, all_sites_info, pg_rotation_matrixs_scaled, error_convert_siteop_eucli_scaled, N_op_total, N_site_point_group_wrong = site_point_group_from_a_material(atoms, test=True)
        
        # if is_group_polar_centrosymmetric_chiral(self._point_group_Sch)[0] :
        
        abc_vector_tran_mat = np.transpose(abc_vector) ### in abc_vector get from spglib, a, b, c vector are placed on each ligne. but in the transformation matrix. a, b, c should be at the column.
        self._formula = atoms.symbols.get_chemical_formula()
        self._prim_std_trans = prim_std_trans
        self._space_group_HM = space_group[0]
        self._space_group_number = space_group[1]
        self._point_group_HM = point_group
        self._point_group_Sch = point_group_hm_sch[point_group]
        self._cell_basis_vectors = abc_vector

        pg_polar, pg_centrosymmetric, pg_chiral = is_group_polar_centrosymmetric_chiral(self._point_group_Sch)
        self._pg_polar = pg_polar
        self._pg_centrosymmetric = pg_centrosymmetric
        self._pg_chiral = pg_chiral
        self._site_symm_all_symmorphic = not have_non_symmorphic_site_op

        # self._polar_axis = []

        #### if the space group is polar, find the polar axis and store it to self._polar_axis. if not, self._polar_axis = []
        
        principal_axis, order, type_principal_axis = find_principal_axis_and_order_from_scaled_rots(np.array(spg_ops_scaled)[:, :3, :3])
        self._principal_axis = [principal_axis, order, type_principal_axis]
        # if self._pg_polar :
        #     order_max = 0
        #     for rot_scaled_std in pg_rotation_matrixs_scaled : ### iteration in rotation matrix given by spglib.
        #         rot_scaled_prim = change_basis(rot_scaled_std, prim_std_trans) ### from std cell basis to primitive cell basis
        #         rot_scaled_prim = rot_scaled_prim[:3,:3] ### we need only the rotation matrix, not the translation part.
        #         abc_vector_inv = np.linalg.inv(abc_vector_tran_mat)
                
        #         # rot = change_matrix_basis_from_eucli_to_frac(rot_scaled, abc_vector_inv ) ### in fact here we change the rotation matrix from primitive cell basis to the cartesien basis
        #         rot_affine_cart = change_basis(rot_scaled_prim, abc_vector_inv) ### in fact here we change the rotation matrix from primitive cell basis to the cartesien basis
        #         rot = rot_affine_cart[:3,:3] ### we need only the rotation matrix, not the translation part.
        #         symmop_objet = SymmOp.from_rotation_and_translation(rotation_matrix=rot)
                
        #         analyser = OperationAnalyzer(symmop_objet)
        #         order = analyser.order
        #         axis = analyser.axis

        #         if order == None : ### this case occur only if OperationAnalyzer find that the rotation matrix is not orthogonal.
        #             # print(np.transpose(rot) @ rot) ### value item to np.dot() 

        #             print(rot)
        #             rot_affine_cart = change_basis(rot_scaled_prim, abc_vector_inv)
        #             m1 = np.dot(np.transpose(rot), rot)
        #             m2 = np.dot(rot, np.transpose(rot))
        #             print(m1)
        #             print(m2)
        #             print("\nDeviation of M @ M.T from identity:")
        #             print(m1 - np.eye(3))
        #             print("\nDeviation of M.T @ M from identity:")
        #             print(m2 - np.eye(3))
        #             print("\nTesting M @ M.T against identity:")
        #             print(np.allclose(m1, np.eye(3), rtol=1e-5, atol=1e-8))

        #             print("\nTesting M.T @ M against identity:")
        #             print(np.allclose(m2, np.eye(3), rtol=1e-5, atol=1e-8))

        #             # print(np.allclose(m2, np.identity(3), rtol=0.001))

        #         else :
        #             if order > order_max: 
        #                 order_max = order
        #                 # axis = axis.tolist()
        #                 # if axis == None :
        #                 #     principal_axis = [] ### show that if principal_axis == None, the group is C1
        #                 # else :
        #                 #     principal_axis = axis.tolist()
        #                 try :
        #                     principal_axis = axis
                
        #                     principal_axis = np.linalg.inv(prim_std_trans) @ np.array(principal_axis)
        #                     principal_axis = principal_axis.tolist()
        #                 except :
        #                     if axis is None : ## if order is not None, but axis is None, it means that the principal axis is the point group is C1
        #                         principal_axis = [] ### show that if principal_axis == None, the group is C1
        #                 type_axis = analyser.type
        #     self._polar_axis = [type_axis, principal_axis, order_max]
        # else :
        #     self._polar_axis = [] ### to tell that for non polar point group, it doesnt have polar axis.


        ### classification of site point group by polar, centrosymmetric and chiral of all site in the crystal entred 
        self._sites_point_group = []
        for i in range(len(all_sites_info)) :
            site = all_sites_info[i]
            site_point_group = Site_point_group()
            site_point_group._Z_number = site[0]
            site_point_group._scaled_position = site[5]
            site_point_group_HM = site[1]
            site_point_group._site_point_group_HM = site_point_group_HM
            site_point_group._site_point_group_Sch = site[2]
            site_point_group._wyckoff_letter = site[4]
            site_point_group._site_symmetry_ITA = site[3]
            site_point_group._order_max = site[6]
            site_point_group._principal_axis = site[7]
            site_point_group._type_principal_axis = site[8]
            site_point_group._have_non_symmorphic_op = site[9]
            site_point_group._out_unit_cell_by_non_symmorphic_op = site[10]
            # Z_number = site[0]
            
            is_polar, is_centrosymmetric,is_chiral =  is_group_polar_centrosymmetric_chiral(site_point_group_HM)
            site_point_group._polar = is_polar
            site_point_group._centrosymmetric = is_centrosymmetric
            site_point_group._chiral = is_chiral
            
            # all_sites_info[i].append(site_polar)
            # all_sites_info[i].append(site_centrosymmetric)
            # all_sites_info[i].append(site_chiral)

            self._sites_point_group.append(site_point_group)
        
        
        # return error_convert_siteop_eucli_scaled, N_op_total, N_site_point_group_wrong
    
    # def get_dipole_direction(self) -> np.ndarray:
    #     return self.

    ### some function to get the information of the atoms sites in order of the list of class Site_point_group
    def get_scaled_positions(self):
        list_scaled_position = []
        for site in self._sites_point_group:
            list_scaled_position.append(site._scaled_position)
        return list_scaled_position
    
    def get_wyckoff_letters(self):
        list_wyckoff_letter = []
        for site in self._sites_point_group:
            list_wyckoff_letter.append(site._wyckoff_letter)
        return list_wyckoff_letter
    
    def get_sites_symmetry_ITA(self):
        list_site_symmetry_ITA = []
        for site in self._sites_point_group:
            list_site_symmetry_ITA.append(site._site_symmetry_ITA)
        return list_site_symmetry_ITA
    
    def get_sites_point_group_HM(self):
        list_site_point_group_HM = []
        for site in self._sites_point_group:
            list_site_point_group_HM.append(site._site_point_group_HM)
        return list_site_point_group_HM
    
    def get_sites_point_group_Sch(self):
        list_site_point_group_Sch = []
        for site in self._sites_point_group:
            list_site_point_group_Sch.append(site._site_point_group_Sch)
        return list_site_point_group_Sch
    
    def get_Z_numbers(self):
        list_Z_numbers = []
        for site in self._sites_point_group:
            list_Z_numbers.append(site._Z_number)
        return list_Z_numbers
    
    
    def get_ase_objet(self):
        """
        get the ase atoms objet from this class

        """
        
        cell = self._cell_basis_vectors
        sites_info = self._sites_point_group
        scaled_positions = []
        Z_numbers = []
        for site in sites_info :
            Z_number = site._Z_number
            Z_numbers.append(Z_number)
            scaled_position = site._scaled_position
            scaled_position = np.array(scaled_position)
            scaled_positions.append(scaled_position)
        ase_object = Atoms(cell=cell, scaled_positions=scaled_positions, numbers=Z_numbers, pbc=True)

        return ase_object
    
    def is_polar_centro_chiral(self):
        polar , centro, chiral = is_group_polar_centrosymmetric_chiral(self._point_group_HM)
        return polar, centro, chiral

    def get_spglib_symmetry_data(self):
        
        cell_spglib =(self._cell_basis_vectors, self.get_scaled_positions(), self.get_Z_numbers())
        symm_data_spglib = spglib.get_symmetry_dataset(cell_spglib, symprec=1e-5)
        if self._prim_std_trans is None :
            self._prim_std_trans = symm_data_spglib.transformation_matrix
        return symm_data_spglib

    def to_dict(self):
        """
        TODO : i forgot all _pg_polar centrosymmetric or chiral !!! Done
        """
        list_sites_point_group = []
        for site in self._sites_point_group:
            dict_site = site.to_dict()
            list_sites_point_group.append(dict_site)
        dict = {"Type" : "dict from class Atoms_symmetry_group_direct_space",
                "formula" : self._formula,
                "cell basis vectors" : self._cell_basis_vectors.tolist() ,
                "space group notation HM" : self._space_group_HM,
                "point group notation HM" : self._point_group_HM,
                "point group notation Sch" : self._point_group_Sch,
                "principal axis" : self._principal_axis,
                "list site point group" : list_sites_point_group,
                "have non symmorphic site op" : self._site_symm_all_symmorphic,
                "primitive to standard transformation" : self._prim_std_trans.tolist(),
                'polar' : self._pg_polar,
                'chiral' : self._pg_chiral,
                'centrosymmetric' : self._pg_centrosymmetric,
                'uuid' : self._uuid}
        return dict
    
    def from_dict(self, dict):
        """
        TODO : i forgot all _pg_polar centrosymmetric or chiral !!! Done
        """
        try :
            self._formula = dict["formula"]
            self._cell_basis_vectors = dict["cell basis vectors"]
            self._space_group_HM = dict["space group notation HM"]
            self._point_group_HM = dict["point group notation HM"]
            self._point_group_Sch = dict["point group notation Sch"]
            self._principal_axis = dict["principal axis"]
            list_site_point_group = dict["list site point group"]
            self._site_symm_all_symmorphic = dict["have non symmorphic site op"]
            self._pg_polar = dict['polar']
            self._pg_chiral = dict['chiral']
            self._pg_centrosymmetric = dict['centrosymmetric']
            self._prim_std_trans = np.array(dict["primitive to standard transformation"])
            self._uuid = dict['uuid']
            list_objet_Site_point_group = []
            for site in list_site_point_group:
                ss = Site_point_group()
                ss.from_dic(site)
                list_objet_Site_point_group.append(ss)
            self._sites_point_group = list_objet_Site_point_group

        except:
            print("it's not the correct format of dictionnary to create the class")


def list_all_space_group():
    """
    get all the space group. 

    Return :
    ---------
            panda datafram with index who is the space group number international. And the line with the following : 
            [space group number , space group symbol HM, point group number, point group symbol HM, point group symbol Sch, polar , centrosymmetric, chiral ]
    among them, the polar, centrosymmetric, chiral are boolean who tell you if the point group (so the space group) are polar, centrosymmetric, or chiral.
    """
    ### init the dataframe
    space_group_list = pd.DataFrame({'spg number' : [], 
                                     'spg symbol HM' : [], 
                                     'pg number' : [], 
                                     'pg symbol HM' : [], 
                                     'pg symbol Sch' : [],
                                     'polar' : [],
                                     'centrosymmetric' : [],
                                     'chiral' : []})
    for i in range(1,231) :
        g = Group(i, style="spglib") ### style = spglib. cause we get group symbols from ase.atoms objet with spglib. 
        
        point_group_symbol_HM, point_group_number   = get_point_group(i)[:2]
        point_group_number = int(point_group_number)
        point_group_symbol_Sch = point_group_hm_sch[point_group_symbol_HM]
        space_group_symbol_HM = g.symbol
        space_group_number = int(g.number)
        is_polar = g.polar
        is_centrosymmetric = g.inversion
        is_chiral = g.chiral
        

        #### get these information as dataframe and concatenate to the space_group_list
        line_i = pd.DataFrame({'spg number' : [space_group_number], 
                                'spg symbol HM' : [space_group_symbol_HM], 
                                'pg number' : [point_group_number], 
                                'pg symbol HM' : [point_group_symbol_HM], 
                                'pg symbol Sch' : [point_group_symbol_Sch],
                                'polar' : [is_polar],
                                'centrosymmetric' : [is_centrosymmetric],
                                'chiral' : [is_chiral]}, index= [space_group_number])

        space_group_list = pd.concat([space_group_list, line_i])


    return space_group_list


#### Get the dataframe of space group, its point group, and their polar, centrosymmetric, chiral.
all_space_group = list_all_space_group()





def is_group_polar_centrosymmetric_chiral(group_symbol : str):

    """
    from the dataframe all_space_group generated by list_all_space_group()
    Check if group (space group or point group) is polar centrosymmetric chiral.

    Parametre : symbol HM of space group or point group

    Return : is_polar, is_centrosymmetric, is_chiral are Boolean, tell us if the point group is polar, centrosymmetric and chiral.

    """

    

    
    
    ### convert the point group symbol HM into point group symbol Sch. 
        ## Because i found a confusion in point group symbol HM, but not in Sch notation. Beside The space group HM notation shouldn't have confusion.
        ## so is easier for me to control the error by adding some line in the dictionnary point_group_hm_sch below used

    try :
        group_symbol = point_group_hm_sch[group_symbol] 
    
    except : 
        pass

    # lines_in_all_space_group = all_space_group[all_space_group.loc[:,'spg symbol HM'] == group_symbol 
    #                                                + all_space_group.loc[:,'pg symbol HM'] == group_symbol 
    #                                                + all_space_group.loc[:,'pg symbol Sch'] == group_symbol]
    
    lines1 = all_space_group[all_space_group["spg symbol HM"] == group_symbol]
    lines2 = all_space_group[all_space_group["pg symbol HM"] == group_symbol]
    lines3 = all_space_group[all_space_group["pg symbol Sch"] == group_symbol]
    # lines_in_all_space_group = all_space_group[all_space_group['spg number'].isin([group_symbol]) or all_space_group['pg number'].isin([group_symbol])]
    lines_in_all_space_group = pd.concat([lines1, lines2, lines3]) ### on of the linei above can be filled.
    
    # print(lines_in_all_space_group)

    is_polar = lines_in_all_space_group['polar'].iloc[0]
    is_centrosymmetric = lines_in_all_space_group['centrosymmetric'].iloc[0]
    is_chiral = lines_in_all_space_group['chiral'].iloc[0]

    return is_polar, is_centrosymmetric, is_chiral

###################
################### Some function generate the database and save them

def generate_database_symmetry_directspace_from_ase_Atoms(crystals : list[Atoms], uids = []):
    """
    
    for each material of an ase.atoms.Atoms object (who represent a crystal), find its symmetry information in directspace needed by the class Atoms_symmetry_group_direct_space. 
    
    Parameter : crystals : a list of ase.atoms.Atoms object. 
                uids : a list of uid whos will be set to the label of each lines. Default value is []. 

    Return database_groups_directspace : pandas.DataFrame: with column (N0.) "Material and Direct space symmetrys" (might add some more column if we want), 
                                                            n lines where n equal to the number of Atoms of the list entered.
                                                            each element is a Atoms_symmetry_group_direct_space.
    """
    #### just test : how many there are in all cif the site : sum error_convert_siteop_eucli_scaled of every crystal (Atoms)
    # N_error_convert_siteop_eucli_scaled = 0
    # N_site_op = 0
    # N_site_pg_wrong = 0
    database_groups_directspace = pd.DataFrame(dtype=object)
    if uids == []:
        uids = range(len(crystals))
    
    for i in range(len(crystals)):
        atoms = crystals[i]
        direct_space_data = Atoms_symmetry_group_direct_space()
        
        
        direct_space_data.get_site_point_group(atoms)
        # error_convert_siteop_eucli_scaled, N_op_total, n_site_poing_group_wrong = direct_space_data.get_site_point_group(atoms)
        # N_error_convert_siteop_eucli_scaled += error_convert_siteop_eucli_scaled
        # N_site_op += N_op_total
        # N_site_pg_wrong += n_site_poing_group_wrong
        direct_space_data._uuid = uids[i]
        R_D_type = find_Rashba_Dresshauls_effet_of_1_crystal(direct_space_data)
        
        ## direct_space_data.get_space_point_group(atoms)
        line_database = pd.DataFrame({
                                      "Material and Direct space symmetrys" : [direct_space_data],
                                      "Rashba or Dressehauls type" : [R_D_type],
                                      }, 
                                      index = [uids[i]]
                                      )
        
        database_groups_directspace = pd.concat([database_groups_directspace, line_database])
        
        
        ### write some simple object first (like list) than some objet complexe (like classes that i've define)
        # database_groups_directspace.at[uid, 'Rashba or Dressehauls type'] = R_D_type ## this column is to determin the type of SOC of Rashba or Dressehauls.
        # print(database_groups_directspace)
        # print("____________________")
        # database_groups_directspace.at[uid, 'Material and Direct space symmetrys'] = direct_space_data
        # print(database_groups_directspace)
        # database_groups_directspace = database_groups_directspace[['Material and Direct space symmetrys', 'Rashba or Dressehauls type']]
        
        # print("###################")
        # print("###################")
    # print(f"{N_error_convert_siteop_eucli_scaled} site ops has bad conversion from euclidean space in {N_site_op} matrixs")
    # print(f"{N_error_convert_siteop_eucli_scaled} site ops after initial methode different to correspond site op in wyckoff_position objet in {N_site_op} matrixs")
    # print(f"{N_site_pg_wrong} site get wrong point group with initial methode")
    return database_groups_directspace





def generate_database_symmetry_directspace_from_aiida_group_matuuid(structures_2D : AiiDAGroup):
    atoms = []
    uids = []
    for node in structures_2D.nodes:
        try :
            ase_objet = node.get_ase()
        except :
            Warning(f'node {node.uuid} cannot be converted to ase object. not a structureData')
            ase_objet = None
            pass
        uuid = node.uuid

    for node in structures_2D.nodes:
        try :
            ase_objet = node.get_ase()
        except :
            Warning(f'node {node.uuid} cannot be converted to ase object. not a structureData')
            ase_objet = None
            pass

        if ase_objet is not None :
            direct_space_data = Atoms_symmetry_group_direct_space()
            direct_space_data.get_site_point_group(ase_objet)
            direct_space_data._uuid = node.uuid
            uuid = node.uuid
            R_D_type = find_Rashba_Dresshauls_effet_of_1_crystal(direct_space_data)
            line_database = pd.DataFrame({
                                        "Material and Direct space symmetrys" : [direct_space_data],
                                        "Rashba or Dressehauls type" : [R_D_type]}, 
                                        index = [uuid]
                                        )
            
            try :
                database_groups_directspace = pd.concat([database_groups_directspace, line_database])
            except :
                database_groups_directspace = line_database    

        
    return database_groups_directspace


def generate_database_symmetry_directspace_from_cifs_of_diff_materials():
    """
    This funciton open a file or selection window. You need to select a cif file or a directory contain file cifs. 
    
    use generate_database_symmetry_directspace_ase_Atoms to generate a database who save symmetry information in directspace needed by the class Atoms_symmetry_group_direct_space for each crystal.
    
    Parameter : None

    Return database : pandas.DataFrame: with just 1 column "Material and Direct space symmetrys" , 
                      n lines where n equal to the number of cif file that you selected. ligne label is uid. Here uid is the file name
                      each element is a Atoms_symmetry_group_direct_space.
    """
    ensemble_path_cif = ask_file_or_directory_cif()
    list_Atoms = []
    uids = []

    for file_path in ensemble_path_cif:
        atoms = read_cif_ase(file_path)
        list_Atoms.append(atoms)
        uid = file_path.split('/')[-1]
        uid = uid.split('.')[0]
        uids.append(uid)
    
    database = generate_database_symmetry_directspace_from_ase_Atoms(list_Atoms, uids=uids)
    return database




class NpEncoder(json.JSONEncoder):
    """convert non-serializable types of JSON to their serializable type
    To expand if you have more non-serializable type.
        
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_dataframe_ASGDS_json(data : pd.DataFrame, file_path = defaut_path_json):
    if data.columns[0] == "Material and Direct space symmetrys" :
        ### first convert the ASGDS objet in the dataframe into the dictionnary
        dict_tot = data.to_dict()
        for uid in data.index.tolist():
            crystal = data.loc[uid , 'Material and Direct space symmetrys']
            
            dict = crystal.to_dict()
            dict_tot['Material and Direct space symmetrys'][uid] = dict
        
        with open(file_path, "w") as f :
            json.dump(dict_tot, f, cls=NpEncoder) ### int64 not JSON serializable. we use the class NpEncoder to convert type compatible to json
            f.close()

def get_dataframe_ASGDS_json(path : str):
    # data_dict = json.load(path)
    d = pd.read_json(path)
    index = d.index.tolist()
    for uid in index :
        dict_ASGDS = d.loc[uid, 'Material and Direct space symmetrys']
        ASGDS = Atoms_symmetry_group_direct_space()
        ASGDS.from_dict(dict_ASGDS)
        d.loc[uid, 'Material and Direct space symmetrys'] = ASGDS
    return d







#############
############# check if the space group given by spglib is correct.
def check_pks_space_group_ops(pks : list[int]):
    for pk in pks:
        ase_objet = load_node(pk).get_ase()
        asgds_obj = Atoms_symmetry_group_direct_space()
        asgds_obj.get_site_point_group(ase_objet)
        prim_std_trans = asgds_obj._prim_std_trans
        spg_num = asgds_obj._space_group_number
        spg = Group(spg_num)
        spg = spg.get_spg_symmetry_object()
        spg_opas_carte = spg.opas







#############################################
#############################################
###### if site point group or cyrstal point group is polar, from the symmetry, determine the direction of the dipole (excepte the signe of the dipole).
########################################################### Note, the dipole direction of crystal is already registed in the class Atoms_symmetry_group_direct_space







def get_pair_sector_for_C2_or_C2v(ASGDS : Atoms_symmetry_group_direct_space):
    """
    same idea as get_dipole_sum_for_Cn_bigger_2. It's just C2 or C2v can have polar axis in plan (so dipole in plan). This function traited just case site dipole in plan. Case perpendicular to plan is in get_dipole_sum_for_Cn_bigger_2

    Para, Return : same as get_dipole_sum_for_Cn_bigger_2, but the sector is a line in plane.
    """




def get_plan_glideplan_opas_from_sg(ASGDS : Atoms_symmetry_group_direct_space):
    """
    From a objet Atoms_symmetry_group_direct_space, return the glide plane operations (in the direct space) of the plan glide plane of the site point
    This operations are pymagten OperationAnalyser type

    ### Done!!! TODO, replace the par of pyxtal Group objet by the axis refound in the spglib database regenerated by Atoms_symmetry_group_direct_space objet
    
    Input : 
    ---------
    ASGDS : Atoms_symmetry_group_direct_space objet

    Return :
    --------- 
        
        opas_m_ez : 
        opas_glide_m_ez
        opas_m_inplan
        opas_glide_m_inplan 

        they are list of affine matrix. ez means normal axis is follow z axis. inplan means the normal axis is in plan xy.
    
    """
    spg = ASGDS._space_group_number
    # pyx_spg_objet = Group(spg)
    # spg_symm = pyx_spg_objet.get_spg_symmetry_object()
    # spg_symm_opas = spg_symm.opas ### list of operation analyser
    spglib_data = ASGDS.get_spglib_symmetry_data()
    spg_symm_opas = affine_from_rot_and_trans(rots=spglib_data['rotations'], trans=spglib_data['translations'])

    ### to find the mirror or glide plans in the space group.
    opas_m_ez = [] #### here, m represent mirror, ez represent the z axis. m's axis is perpendicular to m. here m_ez means m's axis is along z axis.
    opas_glide_m_ez = []

    ### difference from these case and the case above is just the mirror's normal vector in plan.
    opas_m_inplan = []
    opas_glide_m_inplan = []
    for affine_matrix in spg_symm_opas:
        axis, det, angle = find_rotation_axis(affine_matrix)
        
        tran_vec = affine_matrix[:,-1]
        #### tran_vec_in_plan need to be true, cause we study the case of materiaux 2D. And dont forget the atoms objet can be consider as multiple layers.
        tran_vec_in_plan = tran_vec[2] == 0 and (tran_vec[0] != 0 or tran_vec[1] != 0)
        
        if det == -1: ### all det of mirror(include glide plan) are -1
            
            if axis[0] == 0 and axis[1] == 0 and np.array_equal(tran_vec, np.zeros((3,))): ### mirror which normal axis parallel to Z
                opas_m_ez.append(affine_matrix)
            if axis[0] == 0 and axis[1] == 0 and tran_vec_in_plan : #### glide plan who have glide direciton in plan xy.
                opas_glide_m_ez.append(affine_matrix)
            if axis[2] == 0 and np.array_equal(tran_vec, np.zeros((3,))): ### mirror which normal axis parallel in plan
                opas_m_inplan.append(affine_matrix)
            if axis[2] == 0 and tran_vec_in_plan:
                opas_glide_m_inplan.append(affine_matrix)
    return opas_m_ez, opas_glide_m_ez, opas_m_inplan, opas_glide_m_inplan




def class_polar_site_in_diff_point_group(sites_info : list[Site_point_group], dipole_in_plan : bool = False):
    """
    ### Warning : no validated

    this function is used to firstly class the polar site point group in different case to apply it to finally get pairs of polar site.

    We can have different cases:
            - Polar axis perpendicular to plan : if we find correspond sectors, sector is a plan in plan
                1. Cn or Cnv (n > 2) : the polar axis is perpendicular to the plan
                2. C2 or C2v and axis C2 is perpendicular to the plan
                3. Cs (it's just a mirror) where mirror's normal axis in plan. So polar axis is mostly possible out of the plan.
                4. C1  : the polar axis have most possible componenent in plan and component perpendicular to plan.
            - Polar axis in plan : if we find correspond sectors, sector is just a line in plan  
                5. C2 or C2v and axis C2 is in plan
                6. Cs where mirror's normal axis perpendicular to the plan, so polar axis is 100% in plan.
                7. C1 : the polar axis have most possible componenent in plan and component perpendicular to plan.
    So the function need to return 5 case, 
    (1, 2 have not pratical difference)
    (4 = 7 are the same case but will be treat in either when we are intressted by component out of plan or in plan.)

    If dipole_in_plan is true :  Find 5,6,7 case
    If dipole_in_plan is false : Find 1,2, 3, 4 case
    
    Each case is a dict : we distingust the keys by Z number and wyckoff position letter


    Note : don't forget, for a axis of a symmetry element in 2D materials, it can just be in plan or perpendicular to the plan.
    """
    dic_atoms_n_bigger_2 = {} #### case 1 or 2
    dic_atoms_Cs_inplan = {} #### case 3
    dic_atoms_Cs_perpen_plan = {} #### case 6
    dic_atoms_C1 = {} #### case 4 or 7
    dic_atoms_n_2_in_plane = {} ### case 5 
    for site in sites_info :
        # site_polar = site._polar
        # site_order_max = site._order_max
        # site_type_principal_axis = site._type_principal_axis
        # site_point_group = site._site_point_group_Sch

        if site._site_point_group_Sch == 'C1': ### case 4 and 7
            Z_number = site._Z_number
            wpl = site._wyckoff_letter ### wyckoff position letter
            # site_point_group = site._site_point_group_Sch
            combin_Z_wp = f"{Z_number} {wpl}"
            scaled_position = site._scaled_position
            try:
                dic_atoms_C1[combin_Z_wp].append(scaled_position)
            except :
                dic_atoms_C1[combin_Z_wp] = [scaled_position]
        elif site._site_point_group_Sch == 'C2' or site._site_point_group_Sch == 'C2v' : ### case 5
            principal_axis = site._principal_axis
            pa_in_plan = principal_axis[2] == 0. ## principal axis in plan.
            if pa_in_plan:
                Z_number = site._Z_number
                wpl = site._wyckoff_letter ### wyckoff position letter
                # site_point_group = site._site_point_group_Sch
                combin_Z_wp = f"{Z_number} {wpl}"
                scaled_position = site._scaled_position
                try:
                    dic_atoms_n_2_in_plane[combin_Z_wp].append(scaled_position)
                except :
                    dic_atoms_n_2_in_plane[combin_Z_wp] = [scaled_position]

        elif site._site_point_group_Sch == 'Cs' : #### case 6
                principal_axis = site._principal_axis
                pa_in_plan = principal_axis[2] != 0. ## principal axis perpendicular to plan. 
                if pa_in_plan:
                    Z_number = site._Z_number
                    wpl = site._wyckoff_letter ### wyckoff position letter
                    # site_point_group = site._site_point_group_Sch
                    combin_Z_wp = f"{Z_number} {wpl}"
                    scaled_position = site._scaled_position
                    try:
                        dic_atoms_Cs_perpen_plan[combin_Z_wp].append(scaled_position)
                    except :
                        dic_atoms_Cs_perpen_plan[combin_Z_wp] = [scaled_position]
        if dipole_in_plan :
            if site._polar and site._site_point_group_Sch != 'C1': #### case 1 and 2
                site_principal_axis = site._principal_axis
                site_principal_axis = np.array(site_principal_axis)
                principal_axis_out_plane = np.array_equal(site_principal_axis * np.array([0,0,1.]), site_principal_axis)
                # ii = site._order_max
                ### the if below select all Cn or Cnv with (n >= 2 and Cn perpendicular to plan <=> C2 or C2v perpendicular to plan)
                if site._order_max >= 2 and site._type_principal_axis == 'rotation' and principal_axis_out_plane: #### if site have a polar point group and the order of its principal axis > 2
                    Z_number = site._Z_number
                    wpl = site._wyckoff_letter ### wyckoff position letter
                    # site_point_group = site._site_point_group_Sch
                    combin_Z_wp = f"{Z_number} {wpl}"
                    # polar_axis = site_principal_axis
                    
                    scaled_position = site._scaled_position
                    scaled_position = np.array(scaled_position)
                    # carte_position = cell @  scaled_position ### cartesian position
                    
                    try:
                        dic_atoms_n_bigger_2[combin_Z_wp].append(scaled_position)
                    except :
                        dic_atoms_n_bigger_2[combin_Z_wp] = [scaled_position]
            
            elif site._site_point_group_Sch == 'Cs' : #### case 3
                principal_axis = site._principal_axis
                pa_in_plan = principal_axis[2] == 0. ## principal axis in plan.
                if pa_in_plan:
                    Z_number = site._Z_number
                    wpl = site._wyckoff_letter ### wyckoff position letter
                    # site_point_group = site._site_point_group_Sch
                    combin_Z_wp = f"{Z_number} {wpl}"
                    scaled_position = site._scaled_position
                    try:
                        dic_atoms_Cs_inplan[combin_Z_wp].append(scaled_position)
                    except :
                        dic_atoms_Cs_inplan[combin_Z_wp] = [scaled_position]
        


def get_dipole_sum_for_Cn_bigger_2(ASGDS : Atoms_symmetry_group_direct_space) :
    """
    In a material, for a set of atoms where Z and site point group are same and site point group is polar, when polar axis Cn > 2, find out if the dipole sum is 0. if it's 0 find the sectors who will help us to find the hidden spin splitting in type Rashba.

    Para :
        ASGDS :  a Atoms_symmetry_group_direct_space objet. 
        From its attribute ._site_point_group, we can get the site point group and the dipole direction if site is polar (excepte point group C1 and Cs).

    Return:
        dipol_Z_wp_non_null :  list of string [f'{Z number} {wyckoff position letter'}] which Z at this wyckoff position have dipole // ez
        dipol_null_pairs : dict : {f'{Z number} {wyckoff position letter'} : [(pos1, pos2),...]}: for each Z at this wyckoff position, find out all pairs who are related by the mirror or glide plane of the space group. 
        Z_wp_sectors_pairs : dict : from dipol_null_pairs, find out all pairs of sectors (sector is define by a plan having a fix ez)
    
    
    """
    cell = ASGDS._cell_basis_vectors
    cell = np.transpose(cell)
    # ase_objet = ASGDS.get_ase_objet()
    dic_atoms_n_bigger_2 = {}
    sites_info = ASGDS._sites_point_group
    spg = ASGDS._space_group_number
    
    # pyx_spg_objet = Group(spg)
    # spg_symm = pyx_spg_objet.get_spg_symmetry_object()
    # spg_symm_opas = spg_symm.opas ### list of operation analyser

    
    ### to find the mirror or glide plans in the space group.
    # opas_m_ez = [] #### here, m represent mirror, ez represent the z axis. m's axis is perpendicular to m. here m_ez means m's axis is along z axis.
    # opas_glide_m_ez = []

    # for opa in spg_symm_opas:
    #     affine_matrix = opa.affine_matrix
    #     tran_vec = affine_matrix[:,-1]
    #     tran_vec_in_plan = tran_vec[2] == 0 and (tran_vec[0] != 0 or tran_vec[1] != 0)
    #     type_axis = opa.type
    #     if type_axis == 'rotoinversion':
    #         axis = opa.axis
    #         if axis[0] == 0 and axis[1] == 0 and np.array_equal(tran_vec, np.zeros((3,))): ### parallel to Z
    #             opas_m_ez.append(opa)
    #         if axis[0] == 0 and axis[1] == 0 and tran_vec_in_plan : #### glide plan who have glide direciton in plan xy.
    #             opas_glide_m_ez.append(opa)
    opas_m_ez, opas_glide_m_ez, opas_m_inplan, opas_glide_m_inplan = get_plan_glideplan_opas_from_sg(ASGDS=ASGDS)

    dipol_Z_wp_non_null = []
    dipol_null_pairs = {}
    
    for site in sites_info :
        # site_polar = site._polar
        # site_order_max = site._order_max
        # site_type_principal_axis = site._type_principal_axis
        # site_point_group = site._site_point_group_Sch
        if site._polar and site._site_point_group_Sch != 'C1':
            site_principal_axis = site._principal_axis
            site_principal_axis = np.array(site_principal_axis)
            principal_axis_out_plane = np.array_equal(site_principal_axis * np.array([0,0,1.]), site_principal_axis)
            # ii = site._order_max
            ### the if below select all propre Cn or Cnv with (n >= 2 and Cn perpendicular to plan <=> C2 or C2v perpendicular to plan)
            if site._order_max >= 2 and site._type_principal_axis == 'rotation' and principal_axis_out_plane: #### if site have a polar point group and the order of its principal axis > 2
                Z_number = site._Z_number
                wpl = site._wyckoff_letter ### wyckoff position letter
                # site_point_group = site._site_point_group_Sch
                combin_Z_wp = f"{Z_number} {wpl}"
                # polar_axis = site_principal_axis
                
                scaled_position = site._scaled_position
                scaled_position = np.array(scaled_position)
                # carte_position = cell @  scaled_position ### cartesian position
                
                try:
                    dic_atoms_n_bigger_2[combin_Z_wp].append(scaled_position)
                except :
                    dic_atoms_n_bigger_2[combin_Z_wp] = [scaled_position]
    
    # df_atoms_n_bigger_2 = pd.DataFrame(dic_atoms_n_bigger_2) 
    
    for Z_wp in dic_atoms_n_bigger_2 : #### iteration for each pair of (Z number, wyckoff position)
        list_positions = dic_atoms_n_bigger_2[Z_wp]
        
        n_Z = len(list_positions)

        ##### find pairs related by glide plan or mirror plan
        pairs_mirror = []
        pairs_glide_plan = []
        modulo = n_Z%2
        if  modulo == 1 :
            dipol_Z_wp_non_null.append(Z_wp)
        else : ### if modulo == 0, test if pos are related by plane or glide plane
            pairs_atom = find_pairs(list_positions)
            for pair in pairs_atom :
                for opa in opas_m_ez :
                    is_symmetry_related, _ = symmetry_related(pair[0], pair[1], opa)
                    if is_symmetry_related :
                        pairs_mirror.append(pair)
                for opa in opas_glide_m_ez :
                    is_symmetry_related, _ = symmetry_related(pair[0], pair[1], opa)
                    if is_symmetry_related :
                        pairs_glide_plan.append(pair)
                    # else:
                    #     ii = pair[1]
                    #     iii = opa.operate(pair[0])
                    #     print(ii)
                    #     print(iii)
        
        dipol_null_pairs[Z_wp] = {'m' : pairs_mirror, 'm glide' : pairs_glide_plan}

        ###identifie sector for each 'Z wp', calculated distance between each pair of sectors
    
    Z_wp_sectors_pairs = get_sectors_from_ez_coords(dipol_null_pairs)
    return dipol_Z_wp_non_null, dipol_null_pairs, Z_wp_sectors_pairs





def find_pairs_sectors(list_pairs : list[tuple]):
    """
    find sector from the pairs (tuple of 2 element) of positions(ndarray of shape (3,))
    """
    # Step 1: Normalize tuples based on z components
    normalized_tuples = [tuple(sorted((a[2], b[2]))) for a, b in list_pairs]  # Sort z components to handle (z1, z2) and (z2, z1)

    # Step 2: Remove duplicates using a set
    unique_normalized_tuples = list(set(normalized_tuples))
    
    # for i, pair in enumerate(unique_normalized_tuples) :
    #     distance = abs(pair[0] - pair[1])
    #     unique_normalized_tuples[i] = (pair[0], pair[1], distance)
    return unique_normalized_tuples

def get_sectors_from_ez_coords(dipol_null_pairs):
    """
    Only valide for sectors in plan.
    Para:
        dipol_null_pairs, see get_dipole_sum_for_Cn_bigger_2
    Return:
        dict : {'{Z number} {wyckoff position letter}'  : {'m' : {tuple of pair of ec (idem to ez) component }}} . e.g {'8 a' : {'m' : (0.5,0.7), 'm glide' : (0.5,0)}}
            Note:  m : mirror plan
                m glide : glide plan
    """
    sectors_pairs = {}
    for Z_wp in dipol_null_pairs :
        pairs_mirror = dipol_null_pairs[Z_wp]['m']
        pairs_glide_plan = dipol_null_pairs[Z_wp]['m glide']
        pairs_sectors_mirror = find_pairs_sectors(pairs_mirror)
        pairs_sectors_glide_plan = find_pairs_sectors(pairs_glide_plan)
        
        sectors_pairs[Z_wp] = {'m' : pairs_sectors_mirror, 'm glide' : pairs_sectors_glide_plan}
    
    return sectors_pairs
        


def find_Rashba_Dresshauls_effet_of_1_crystal (direct_space_data : Atoms_symmetry_group_direct_space):
    """
    Classification of space group and site point group with the idea from (Zhang .al , 2014, DOI : 10.1038/nphys2933 ). 
    The objective is find the R-1, R-2, D-1, D-2 effect accroding to (Zhang .al , 2014, DOI : 10.1038/nphys2933 )

    Parameter : direct_space_data : class Atoms_symmetry_group_direct_space


    Return : SOC_type : list[str]. A list of string denote by 'R-2' or 'R-1' or 'D-1' or 'D-2'
            'R-2 and D-2' : R-2 and D-2
            'D-2'        : D-2 pure
            'no R or D' : No R or D SOC
            'D-1 no any polar site' : D-1 in case no any polar site
            'R-1 and D-1' : R-1 and D-1 with space group polar
            'D-1 polar site exist' : D-1 with at least one polar site
            'unexpected case' : it shouln't existe.

    ## To do : find progress of symmetry analyse of hidden Rashba & Dresshauls effet after this artical. B
                But normally the analyse in this artical containt all criterion of symmetry classification of R-1, R-2, D-1, D-2
    
    Abridge :
        Space group : sg
        site Point group : spg
                (no) Dipole add up to zero: (N)DZ
        (no) centrosymmetric : (N)CS
        (no) polar : (N)P


    Actual Criterion (version Zhang .al, 2014): 
                            spg_NCS                     spg_CS
                spg_NP     spg_P_DZ     spg_P_NDZ     

    sg_NCS        D-1          D-1        R-1 & D1      No possible
    
    sg_CS         D-2         D-2&R-2      D-2&R-2     No R or D SOC
                    
    

    #########
    !!!!!!!!!
    Important : For now this function can just identifie the 2nd line (case space group is centrosymmetric) and the (sg_NCS and spg_NP and spg_NCS)
                For now the 2 case sp_NCS and spg_NCNS and (spg_P_DZ or spg_P_NDZ) , we have no complet methode to distingust them 
                    But at least in these 2 indeterminate cases. We know that if the point group of crystal is polar belong to spg_P_NDZ.
                    so, if output == ['indeterminate cases']. spg_P_DZ or spg_P_NDZ except crystal polar point group case.


    Note :
        The analyse symmetry here is not the end of D-(1)2 or R-(1)2. They can be check with degenerate band of spin up and spin down in band structure.

        Editer think sg_CS and spg_P_NDZ can't coexisted. But we keep the posibility to find this case in the code.
    """

    sites_info = direct_space_data._sites_point_group
    space_group_HM = direct_space_data._space_group_HM
    space_group = SpaceGroup(space_group_HM)
    space_group_num = space_group.int_number
    # sg_polar, sg_centrosymmetric, sg_chiral = is_group_polar_centrosymmetric_chiral(space_group_HM) ## some symbol dont marche
    sg_polar, sg_centrosymmetric, sg_chiral = get_point_group(space_group_num)[2:]

    at_least_one_site_non_centrosymmetric = False
    at_least_one_site_polar = False
    for i in range(len(sites_info)):
        site = sites_info[i]
        polar = site._polar
        centrosymmetric = site._centrosymmetric
        if polar :
            at_least_one_site_polar = True
        if centrosymmetric == False :
            at_least_one_site_non_centrosymmetric = True
    
    SOC_type = None

     
    ## R-2 and D-2
    if sg_centrosymmetric == True and at_least_one_site_polar == True :
        SOC_type = 'R-2 and D-2'
    ## D-2 pure
    elif sg_centrosymmetric == True and at_least_one_site_non_centrosymmetric == True and at_least_one_site_polar == False :
        SOC_type = 'D-2'
    ## No R or D SOC
    elif sg_centrosymmetric == True and at_least_one_site_non_centrosymmetric == False :
        SOC_type = 'no R or D'
    ## D-1 in case no any polar site
    elif sg_centrosymmetric == False and at_least_one_site_non_centrosymmetric == True and at_least_one_site_polar == False :
        SOC_type = 'D-1 no any polar site'
    ## R-1 and D-1 with space group polar
    elif sg_centrosymmetric == False and sg_polar == True :
        SOC_type = 'R-1 and D-1'
    ## case R-1 and D-1 or pure D-1 with space group non polar but at least one site polar.
    elif sg_centrosymmetric == False and at_least_one_site_non_centrosymmetric == True and sg_polar == False:
        SOC_type = 'D-1 polar site exist'
    else:
        SOC_type = 'unexpected case'
    return SOC_type



def find_max_Z_at_no_centrosymmetric_sites(material : Atoms_symmetry_group_direct_space):
    """
    find the maximum atomic number with no centrosymmetric site in a material 

    Para :
        material : Atoms_symmetry_group_direct_space
    
    Return Z_number_max. If all site are centrosymmetric, return 0
    """

    sites_info = material._sites_point_group
    Z_number_max = 0
    for site in sites_info :
         if not site._centrosymmetric :
            Z_number =  site._Z_number
            if Z_number > Z_number_max:
                Z_number_max = Z_number
    
    return Z_number_max

def find_max_Z_at_polaire_principale_axis(material : Atoms_symmetry_group_direct_space):
    """
    L'axe polaire unique dans les groupe polaire induit les dipôle, je veux le numéro atome maximum qui se situe sur cette axe polaire (sauf dans les C1 et Cs)
    Il est ok si les axe polaire d'un site et l'axe polaire de cristal sont les mêmes
    Arg :
    ------
    material : Atoms_symmetry_group_direct_space
    
    Return :
    --------
    Z_number_max : le numéro atome maximum qui se situe sur l'axe polair
    """
    sites_info = material._sites_point_group
    polar, centro, chiral = material.is_polar_centro_chiral()
    if polar:
        polar_axis, ordre_polar,type_axis = material._principal_axis
        if polar_axis[0] is not None and 'improper' not in type_axis:
            polar_axis = np.array(polar_axis)
            Z_number_max = 0
            for site in sites_info :
                polar_site = site._polar
                if polar_site :
                    axis_site = np.array(site._principal_axis)
                    type_axis_site = site._type_principal_axis
                    if axis_site[0] is not None and 'improper' not in type_axis_site:
                        order_axis_site = site._order_max
                        boolean = np.round(abs(axis_site @ polar_axis), 5) == np.round(np.linalg.norm(axis_site) * np.linalg.norm(axis_site), 5)
                        if boolean and order_axis_site == ordre_polar:
                            Z_number = site._Z_number
                            if Z_number > Z_number_max:
                                Z_number_max = Z_number
            if Z_number_max != 0:
                return Z_number_max
    
############################################
############################################

#### to find out the number of occupied bands, we firstly need the number of electron in the unit cell. 
#### and we need to find if the atom is on vertex, arris, surface or inside the unit cell.

### Note : we don't use this. cause the ase objet always count the atoms as one atoms. no a half atoms on surface...
def atom_on_vertex_arris_surface_inside(scale_position : np.ndarray):
    """
    to confirme if the atom is on vertex or arris or surface

    no used for now, cause finally the ase atoms objet don't use the representation for a factorized atom (at vertex, arris, surface)
    """
    ##from scaled position, find how many component of the coordinate is equal to 0 or 1

    assert np.shape(scale_position) == (3,), "a coordinate is a (3,) ndarray"
    N_edge_component = 0
    for component in scale_position:
        assert component <= 1. and component >= 0., f"all component of a scale position <= 1 and >= 0. enter coordinate : {scale_position}"
        if abs(component - 1) < 1e-2 or abs(component - 0.) < 1e-2: ### i chose 0.01 cause in this scale of a usual crystal, in 0.01a (or b or c), we can't have 2 atoms
            N_edge_component += 1
    
    if N_edge_component == 3:
        return 'vertex'
    if N_edge_component == 2:
        return 'arris'
    if N_edge_component == 1 :
        return 'surface'
    if N_edge_component == 0 :
        return 'inside'





def if_C2_proper_inplan__centro_ornot(asgds : Atoms_symmetry_group_direct_space):

    """
    for a crystal with a C2 axis, return if its axis C2 is in plan, or not. centrosymmetric or not, polar or not.
    """
    axis, order ,type_axis = asgds._principal_axis
    
    
    if type_axis == 'C2.0 proper':
        axis_inplan = (axis[0] != 0. or axis[1] != 0.) and axis[2] == 0.
        centrosymmetric = asgds._pg_centrosymmetric
        polar = asgds._pg_polar

        return axis_inplan, centrosymmetric, polar





def interation_in_df_asgds (df : pd.DataFrame, function : callable, **kwargs):
    """
    iteration in my dataframe


    """










############################################
############################################
############################################
############################################ 
#### We need to find the high symmetry points in brillouin zone and their symmetry element to identyfie their point group.

def product_vect(a : np.ndarray, b : np.ndarray):
    """
    product vectoriel of 2 (3,) matrix.
    """
    assert np.shape(a) == (3,), "a is a (3,) ndarray"
    assert np.shape(b) == (3,), "b is a (3,) ndarray"
    comp1 = a[1] * b[2] - a[2] * b[1]
    comp2 = b[0] * a[2] - b[2] * a[0]
    comp3 = a[0] * b[1] - a[1] * b[0]
    result = np.array([comp1,comp2,comp3])
    return result

def get_reciprocal_basis_vectors(cell : np.ndarray):
    """
    input direct space basis vector should have form of array([a,
                                                                b,
                                                                 c])    
    where a, b ,c are basis vectors  

    Dont forget the dimension of these basis vectors in inverse of meter.
    """
    assert np.shape(cell) == (3,3)
    a = cell[0]
    b = cell[1]
    c = cell[2]

    _2pi_ = 2 * np.pi
    volume = np.linalg.det(cell)
    a_rec = _2pi_ * product_vect(b, c) / volume
    b_rec = _2pi_ * product_vect(c, a) / volume
    c_rec = _2pi_ * product_vect(a, b) / volume
    cell_reciprocal = np.array([a_rec, b_rec, c_rec])
    cell_reciprocal = np.transpose(cell_reciprocal)
    return cell_reciprocal

def merge_kpoint_same_ka_kb(kpoints : np.ndarray):

    """
    merge the kpoints with the same ka and kb, means from 3D to 2D.
    """
    assert np.shape(kpoints)[1] == 3 , "kpoints is a (n, 3) ndarray, if not, no need to merge"
    # Create a dictionary to store groups
    from collections import defaultdict
    groups = defaultdict(list)

    # Group by first two components
    for row in kpoints:
        key = (row[0], row[1])
        groups[key].append(row[2])

    # Convert back to array format
    result = np.array([[k[0], k[1]] for k, v in groups.items()], dtype=object)
    # print(result)
    return result

def is_kpoint_invariant_under_op(op : np.ndarray, kpoint : np.ndarray):
    """
    Check the formula : op(kpoint) = kpoint + A * a + B * b + C * c (A, B, C are intergers, a, b, c are the reciprocal basis vectors)
    
    Args :
    -------
        kpoint's basis vector must base on reciprocal basis vectors. 
        op who is a affine matrix also need to respect the same basis vector.
    
    """
    do_kpoint_have_op, _ = symmetry_related(opa= op, coord1= kpoint, coord2= kpoint)
    
    return do_kpoint_have_op

def remove_same_array(array : np.ndarray):
    """
    ### Warning : no validated
    Chatgpt's code, merge the same array in the axis 0 of the array.
    """
    # Step 1: Identify unique matrices and their first occurrences
    unique_matrices, indices = np.unique(array, axis=0, return_index=True)

    # Step 2: Sort unique matrices by their first occurrence order
    sorted_indices = np.argsort(indices)
    unique_matrices_sorted = unique_matrices[sorted_indices]

    # Step 3: Replace each duplicate with its first occurrence matrix
    output_array = np.zeros_like(array)
    for i, matrix in enumerate(array):
        # Find the index of the first occurrence of the current matrix
        index = np.where((unique_matrices_sorted == matrix).all(axis=(1, 2)))[0][0]
        output_array[i] = unique_matrices_sorted[index]
    
    return output_array




def smallest_angle_difference(angles):
    """
    AI code, not validated. no use for now.
    """
    # Step 1: Sort the angles
    sorted_angles = sorted(angles)

    # Step 2: Calculate the minimum angle difference
    min_difference = 360  # Initialize with the maximum possible difference

    # Step 3: Calculate differences between consecutive angles and apply circular logic
    for i in range(1, len(sorted_angles)):
        diff = sorted_angles[i] - sorted_angles[i - 1]
        # Ensure the shortest difference is used
        diff = min(diff, 360 - diff)
        min_difference = min(min_difference, diff)

    # Step 4: Check the circular difference (from the last angle to the first angle)
    circular_diff = sorted_angles[-1] - sorted_angles[0]
    circular_diff = min(circular_diff, 360 - circular_diff)
    min_difference = min(min_difference, circular_diff)

    return min_difference

# def complete_spg_inplane_order_2_ops(opas : list[OperationAnalyzer]):
#     """
#     pyxtal dont list all vertical mirror or 2 fold axis in plan if there is a n>2 fold axis perpendicular to the plane.
    

#     Note : Dont forget the code is for 2D layer(s).
#     """
#     Order_max = 0
#     opa_order_max = None ### there can have multiple operation of the same axis, like C3 is contain in C6. we need the max out of plane ab.
#     mirrors_vec_angle = []
#     proper_axis_order_2 = []
#     for opa in opas:
#         order = opa.order
#         if order > Order_max:
#             Order_max = order
#             opa_order_max = opa
        
#         if order == 2:
#             if opa.det == -1 :
#                 mirrors_vec_angle.append(opa.angle)
#             if opa.det == 1:
#                 proper_axis_order_2.append(opa.angle)
    
#     angle_min_rot = 2 * np.pi / Order_max
#     mirror_min_diff_angle = smallest_angle_difference(mirrors_vec_angle)
#     axis2_min_diff_angle = smallest_angle_difference(proper_axis_order_2)

#     if mirror_min_diff_angle > angle_min_rot:
        
def transform_matrix_rotation_from_r_to_k_space(primitive_cell : np.ndarray):
    """

    
    The idea is not a transformation matrix from direct to reciprocal space. but to convert the rotation matrix from k to r space, with hypothesis a aligne to a'.

    Example :
    ---------
    For a hexagonal symmetry, the rotation matrix of 120 degree is different in direct and reciprocal space. Cause angle between a (a') and b (b') is respectly 120 and 60 degree.

    All we need is a, b. I don't care the c cause we are in 2D layers. It shoudln't represent a translation symmetry.
    Don't need to condiser norm(a) != norm(b) when angle(a, b) != 90 cause monoclinic and triclinic can't have rotation matrix with axis // c.
    So if you entre an monoclinic or triclinic type , it print the message and return None.



    if angle(a,b) == 90, don't need to transform the matrix. in this case, return the identity matrix. 

    Arg :
    ----
    primitive_cell : np.ndarray, shape (3, 3), it's angle error can be slight and symprec.

    Return:
    -------
    transform_real_reciprocal : np.ndarray, the transformation matrix from the direct space to the reciprocal space. r = transform_real_reciprocal @ k


    
    """

    a, b, c = primitive_cell 
    angle_a_b = angle_2_vec(a, b)
    if np.rint(angle_a_b) == 90:
        return np.identity(3)
    if np.rint(angle_a_b) == 120:
        return np.array([[1, 1, 0], [0, 1,0], [0, 0, 1]])
    else:
        print("The angle between a and b is not 90 or 120 degree. so this bravais lattice don't need this function.")

    


def find_point_group_of_kpoint(kpoint : np.ndarray, 
                            #    reciprocal_basis_vectors : np.ndarray , 
                               primitive_cell : np.ndarray ,
                               pg : list[np.ndarray] ### list of scaled rotation matrix
                            #    spg : Group, 
                               ):
    """
    Arg :
    ---------

        kpoint : np.ndarray, the kpoint in the reciprocal space. It's a (3,) ndarray.
        pg : list[np.ndarray], list of rotation matrix (3,3) of the point group of the crystal but basis in the reciprocal space. The matrix is a (3,3) ndarray.

    the reciprocal_basis_vectors and primitive_cell must have basis vectors in column. Check before enter

    Note :
    ----------
    #### Respect to the defintion the of group of kpoint same as atom site, 
    #### the operation in kpoint's group belong to the space group of the crystal.
    #### But in this version we don't think the non promitive translation play in the reciprocal space. So we just consider the rotation matrix.

    TODO : 
    Possible Error:
    ------------
    ###### matrice of rotation in reciprocal space and real space may should be the same.

    Return :
    ----------
        point_group : tuple of spglib.get_pointgroup()'s return
    """
    
   
    # ops_cartesian = []
    
    # for op in pg:
        # op_cartesian = change_matrix_basis_from_eucli_to_frac(op.rotation_matrix, primitive_cell_inv)
        # op_cartesian = expand_rot_mat_to_affine_mat(op_cartesian)
        # op_cartesian = SymmOp(op_cartesian)
        # ops_cartesian.append(op_cartesian)
    
    # ops = spg.get_spg_symmetry_object().opas

    # mirroirs = []
    
    # for op in ops_cartesian:
        
    #     # rot = op.rotation_matrix
    #     # rotation_expand_affine = expand_rot_mat_to_affine_mat(rot)
    #     # op_no_tran = SymmOp(rotation_expand_affine)
    #     # opa = OperationAnalyzer(op_no_tran)
    #     det = op.det
    #     order = op.order
    #     if det == -1 and order == 2:
    #         mirroirs.append(op)

    ops_validate = []
    for rotation in pg:
        
        
        rotation_expand_affine = expand_rot_mat_to_affine_mat(rotation)
        
        if is_kpoint_invariant_under_op(rotation_expand_affine, kpoint):
            ops_validate.append(rotation)
    # ops_validate_np = np.array(ops_validate)
    # ops_validate_remove_repeated = remove_same_array(ops_validate_np)


    #### I'm not sure if we need to transforme those matrice to real space. But hell i dont think so
    # ops_validate_as_transform_in_real_space = [] ### spglib.get_pointgroup use the basis vector of the real space.
    # Tmat = transform_matrix_rotation_from_r_to_k_space(primitive_cell)
    # for mat in ops_validate:
    #     mat_in_real_space = change_basis(mat, Tmat)
    #     ops_validate_as_transform_in_real_space.append(mat_in_real_space)



    # for mat in ops_validate:
    #     mat_scaled = change_matrix_basis_from_eucli_to_frac(mat, primitive_cell)
    #     ops_validate_as_transform_in_real_space.append(mat_scaled)
    # ops_validate_reci_basis = []
    # for mat in ops_validate:
    #     mat_scaled_reci = change_matrix_basis_from_eucli_to_frac(mat, reciprocal_basis_vectors)
    #     ops_validate_reci_basis.append(mat_scaled_reci)
    try :
        point_group = spglib.get_pointgroup(ops_validate)
    except:
        raise ValueError (f"The kpoint's point group cannot be determined by {ops_validate}")
    return point_group



# def find_some_rashba_out_plan():
    









####################
####################
#################### Some thing to read the remote data in database aiida MC2D, or more



def reparse_qe_recip_syms(PW):    
    """
    read the aiida bandsData's calcul remote data file. and get the k point symmetry operation matrix.

    
    return :
        Rotation matrixs validated for BZ.
    """                                                                                                                                                     
    try:
        ret = PW.outputs.retrieved
    except:
        ret = PW.called_descendants[-1].outputs.retrieved
    with ret.open("aiida.out") as f:
        lines = f.readlines()
    rots = []
    for i in range(len(lines)):
        if "cryst.   s(" in lines[i]:
            end = 0
            end2 = 0
            if "f =" in lines[i]:
                #print "frac. trans."
                end, end2 = 4, 3
            l1 =  np.array([float(x) for x in lines[i].split()[-4-end:-1-end]])
            l2 =  np.array([float(x) for x in lines[i+1].split()[-4-end2:-1-end2]])
            l3 =  np.array([float(x) for x in lines[i+2].split()[-4-end2:-1-end2]])
            s = np.array([l1,l2,l3])
            rots.append([s, 0.])
    return rots






# %%
if __name__ == "__main__":
    pass


# %% generate and save dataframe of Atoms_symmetry_group_direct_space for 2D materials in MC2D database
    group = load_group('structure_2D')
    data_frame_asgds = generate_database_symmetry_directspace_from_aiida_group_matuuid(group)
    save_dataframe_ASGDS_json(data_frame_asgds)


# %% 
    
    
        
        



