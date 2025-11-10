
import numpy as np
import ase.atoms
from ase.atoms import Atoms
import ase.spacegroup
try:
    import spglib 
except ImportError:
    from pyspglib import spglib as spg
import ase 
# from pymatgen.io.cif import CifParser
import pyxtal as pyl
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.symmetry import SymmOp, site_symmetry
from pyxtal.symmetry import site_symm
from pyxtal.operations import OperationAnalyzer

# from pymatgen.symmetry.analyzer import PointGroupAnalyzer
# from pymatgen.symmetry.analyzer import PointGroupOperations
from pymatgen.symmetry.groups import PointGroup


import pymsym as pys

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


import os
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from ase.io import read
from ase.io import cif


### to customize the function is_orthogonal in the pyxtal.operations
import pyxtal.operations

def new_is_orthogonal(m, rtol=1e-3, atol=1e-4):
    m1 = np.dot(m, np.transpose(m))
    m2 = np.dot(np.transpose(m), m)
    return np.allclose(m1, np.identity(3), rtol=rtol, atol=atol) and \
           np.allclose(m2, np.identity(3), rtol=rtol, atol=atol)

pyxtal.operations.is_orthogonal = new_is_orthogonal




symprec = 1e-5
dimension = 3
# PBC = [1,1,0]
point_group_hm_sch = {
    "1": "C1",          # Triclinic
    "-1": "Ci",         # Triclinic
    "2": "C2",          # Monoclinic
    "m": "Cs",          # Monoclinic
    "2/m": "C2h",       # Monoclinic
    "222": "D2",        # Orthorhombic
    "mm2": "C2v",       # Orthorhombic
    "mmm": "D2h",       # Orthorhombic
    "4": "C4",          # Tetragonal
    "-4": "S4",         # Tetragonal
    "4/m": "C4h",       # Tetragonal
    "422": "D4",        # Tetragonal
    "4mm": "C4v",       # Tetragonal
    "-42m": "D2d",      # Tetragonal
    "4/mmm": "D4h",     # Tetragonal
    "3": "C3",          # Trigonal
    "-3": "C3i",        # Trigonal
    "32": "D3",         # Trigonal
    "3m": "C3v",        # Trigonal
    "-3m": "D3d",       # Trigonal
    "6": "C6",          # Hexagonal
    "-6": "C3h",        # Hexagonal
    "6/m": "C6h",       # Hexagonal
    "622": "D6",        # Hexagonal
    "6mm": "C6v",       # Hexagonal
    "-62m": "D3h",      # Hexagonal
    "-6m2" : "D3h",     ### same group as -62m but there are some confusion in the HM notation. They are both D3h.
    "6/mmm": "D6h",     # Hexagonal
    "23": "T",          # Cubic
    "m-3": "Th",        # Cubic
    "432": "O",         # Cubic
    "-43m": "Td",       # Cubic
    "m-3m": "Oh"        # Cubic
}

from ase.spacegroup import Spacegroup , get_spacegroup, get_point_group

# from analyse_symmetry import all_space_group

"""
This module is to do math of symmetry operations. 

Basis : change_basis function, when you use it, check calfully your transformation matrix

If not , the reste will be idolt

"""



class Position :
    def __init__(self, position : np.array):
        shape = np.shape(position) 
        assert shape == (3,), "the vector is in 3 dimension"
        assert position.dtype == np.float64, "the vector of position is a fractional coordinate (might out of a unit cell), dtype of np.array should be float, here float64" 
        
        self._vector = position


class Position_in_unit_cell :
    def __init__(self, position : Position):
        pt = position._vector
        pt -= np.floor(pt)
        self._vector = pt
        
class Positions :
    def __init__(self, positions : np.array):
        shape = np.shape(positions) 
        assert len(shape) == 2 and shape[1] == 3 , "positions is a array containing arraies of shape (3), so np.shape(positions) == (n,3)"
        self._positions = positions


class Positions_in_unit_cell :
    def __init__(self, positions : Positions):
        pos = positions._positions
        frac_pos = []
        shape = np.shape(pos)
        for i in range(shape[0]):
            pt = pos[i,:]
            pt -= np.floor(pt)
            frac_pos.append(pt)
        frac_pos = np.array(frac_pos)
        self._positions = frac_pos
        
class Rotation_matrix : 
    def __init__(self, matrix : np.array):
        shape = np.shape(matrix) 
        assert shape == (3,3), "the rotation matrix has shape 3x3"
        self._matrix = matrix

class Rotation_matrix_frac :
    def __init__(self, matrix : Rotation_matrix):
        mat = matrix._matrix
        mat = np.ndarray.flatten(mat)
        for a in mat :
            assert np.abs(a) == 1 or np.abs(a) == 0, "Rotation_matrix_frac objet have basic vector along the translation of a unit cell. So all matrix element should have absolute value 1 or 0"
        self._matrix = matrix._matrix    

class Cell : 
    def __init__(self, positions : Positions_in_unit_cell, Z_numbers : list):
         
         self._positions = positions._positions
         self._Z_numbers = Z_numbers
         
    
    def __str__(self):
        print(f"{self._positions[i]}: {self._Z_numbers[i]}" for i in range(len(self._Z_numbers)))  
    

    

def is_cif_file(filepath):
    """
    check a file if its cif file."""
    with open(filepath, 'r') as file:
            for line in file:
                if line.strip().startswith("#  This is a CIF file.") :
                    return True
            return False
    
def ask_file_or_directory_cif():
    INPUT = input("you want to work with a file? [y/n]")
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    

    if INPUT == 'y':
        
        filepath = askopenfilename() 
        # if filepath.find('.cif') != -1: 
        #     return filepath # ici la fichier est de format .cif
        is_cif = is_cif_file(filepath)
        if is_cif:
            return [filepath]
        else :
            print("that is not a cif, retry")
            ask_file_or_directory_cif()

    elif INPUT == 'n':
        folder_path = askdirectory() # 
        files = os.listdir(folder_path)
        files_cif = []
        for file in files :
            file = folder_path + '/' + file
            is_cif = is_cif_file(file)
            if is_cif:
                files_cif.append(file)
        if len(files_cif) != 0:
            return files_cif
        else :
            print("don't have any cif, retry")
            ask_file_or_directory_cif()
    
def read_cif_ase(filepath):
    atoms = cif.read_cif(filepath)
    return atoms


def angle_2_vec(vec1, vec2):
    """
    angle between two vectors in degree
    """
    direct_product = vec1 @ vec2
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos = direct_product / (norm1 * norm2)
    angle = np.arccos(cos)
    angle = np.degrees(angle)
    return angle

def check_what_are_possible_scaled_translate_component():
    """
    just check what are the possible value of the translation component if the translation part of the affine matrix is scaled.
    from all 230 space group
    """
    components_non_zero = []
    for i in range(1, 231):
        spg_objet = Group(i)
        spg_opas = spg_objet.get_spg_symmetry_object().opas
        for opa in spg_opas:
            affine_matrix = opa.affine_matrix
            translation = affine_matrix[:3,3]
            for com in translation:
                if com != 0:
                    components_non_zero.append(com)
    set_components_non_zero = set(components_non_zero)
    return set_components_non_zero

### if translation is scale. It's same as the check_what_are_possible_scaled_translate_component() plus 0
### i wrote this istance just avoid each time some function recall check_what_are_possible_scaled_translate_component(), and round value to 5th decimal.
scale_tran_component_possible = {round(1/2, 5),
                                 round(1/3, 5),
                                 round(2/3 ,5),
                                 round(1/4 ,5),
                                 round(3/4,5),
                                 round(1/6 ,5),
                                 round(5/6 , 5),
                                 0.,}
def check_if_translation_is_scaled(tran : np.ndarray):
    """
     check if the translation part is scaled (same as if its basis vector is the bravais lattrice vector).
     We all know if it's scaled, the translation compoent can only be 0, 1/2, 1/3, 1/4, 1/6, or their multiples. These values are not possible if it's not scaled(means unity are angstrom )
     else, it's not scaled. In crystal data, the unity is often angstrom.
    """
    assert np.shape(tran) == (3,) , "the transition vector is not a 3D vector"
    
    for com in tran:
        com_round = round(com, 5)
        if com_round not in scale_tran_component_possible:
            return False
    
    return True

# def check_if_rotation_is_scaled(rot: np.ndarray):
#     """
#     same as check_if_translation_is_scaled() but for rotation part.
#     possible value if scaled : -1, 1, 0
#     """
#     assert np.shape(rot) == (3,3), "the rotation matrix is not a 3x3 matrix"
#     for com in rot.flatten():
#         if com not in [-1., 1., 0.]:
#             return False
        
#     return True






def change_basis(matrix : np.ndarray, basis_trans : np.ndarray):
    """

    {coor in old basis} = basis_trans @ {coor in new basis}
    {Matrix in new basis} = {inversion of basis_trans} @ {Matrix in old basis} @ {basis_trans}
    
    if entring a rotation matrix, apply directly the formula below
    if entring a affine matrix, expand the basis_trans from 3x3 to 4x4, the 4th column and the 4th row is 1, else is 0

    e.g.
    basis_trans = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    basis_trans_after_expend = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])

    
    
    Parameters :
    ----------
        matrix (np.ndarray) : the matrix to be changed basis.
        basis_trans (np.ndarray) : the basis transformation matrix. (3x3 or 4x4) , in (3x3) part, vectors a, b, c representing new basis but writing in old basis are along the column
    
    Return :
    -------
        It will return a affine matrix in new basis.
    Note :
    -------
        the unit_cell should be more correctlly called as the primitive cell. But it's just a code
        This function is correct in math, 
       
        
    
    

    """
    assert np.shape(basis_trans) == (3,3) or np.shape(basis_trans) == (4,4), "shape of entering matrix should be (3,3) or (4,4)"
    assert np.shape(matrix) == (3,3) or np.shape(matrix) == (4,4), "shape of entering matrix should be (3,3) or (4,4)"
    # basis_trans_inv = np.eye(3)
    ### below we firstly check if the matrix is 4 x4 or 3x3, then we check if the basis transformation is 3x3 or 4x4. If it's 4x4, we check if excepte the 3x3 part is 0 and 1 in diagonal, than we inverse the 3x3 part of it.
    if np.shape(matrix) == (4,4):
        if np.shape(basis_trans) == (3,3):
            basis_trans_inv = np.linalg.inv(basis_trans)
            basis_trans = expand_rot_mat_to_affine_mat(basis_trans)
            basis_trans_inv = expand_rot_mat_to_affine_mat(basis_trans_inv)
        elif np.shape(basis_trans) == (4,4):
            ### basis tranformation should only in 3x3 part. below we check that.
            row_0_part = basis_trans[-1,:-1]
            last_col_0_part = basis_trans[:-1,-1]
            element_should_be_1 = basis_trans[-1,-1]
            assert np.all(row_0_part == 0) and np.all(last_col_0_part == 0) and element_should_be_1 == 1, "the 4x4 matrix should have the 4th row and 4th column is 1, else is 0"

            rotation_part = basis_trans[:3,:3]
            rotation_part_inv = np.linalg.inv(rotation_part)
            basis_trans_inv = expand_rot_mat_to_affine_mat(rotation_part_inv)
    else :
        matrix = expand_rot_mat_to_affine_mat(matrix)
        ### it's not beauty, but i just repeat the same code as above.
        if np.shape(basis_trans) == (3,3):
            basis_trans_inv = np.linalg.inv(basis_trans)
            basis_trans = expand_rot_mat_to_affine_mat(basis_trans)
            basis_trans_inv = expand_rot_mat_to_affine_mat(basis_trans_inv)
        elif np.shape(basis_trans) == (4,4):
            ### basis tranformation should only in 3x3 part. below we check that.
            row_0_part = basis_trans[-1,:-1]
            last_col_0_part = basis_trans[:-1,-1]
            element_should_be_1 = basis_trans[-1,-1]
            assert np.all(row_0_part == 0) and np.all(last_col_0_part == 0) and element_should_be_1 == 1, "the 4x4 matrix should have the 4th row and 4th column is 1, else is 0"

            rotation_part = basis_trans[:3,:3]
            rotation_part_inv = np.linalg.inv(rotation_part)
            basis_trans_inv = expand_rot_mat_to_affine_mat(rotation_part_inv)
    # print(basis_trans_inv)
    ###
    # basis_trans_inv = np.linalg.inv(basis_trans)
    # rotation_part_scaled = change_matrix_basis_from_eucli_to_frac(rotation_part, basis_trans) ### change the affine matrix basis
    
    #### above are just validate the entrance, below we apply the change.
    matrix_new_basis =basis_trans_inv @ matrix @ basis_trans
    return matrix_new_basis
    
######## rotation matrix's principal axis and order resolution in scaled basis.
def find_rotation_axis(mat : np.ndarray) -> np.ndarray:
    """
    find the axis of rotation matrix. get eigenvector of character equation (R - \lamda * I) @ axis = 0 if proper rotation, (R + \lambda * I) @ axis = 0 if improper rotation.
    
    for rotations angle, the character's equation (R - \lamda * I) @ vector = 0 give some complexe eigenvalue, their angles are so the rotation angle.
    
    Note : for the rotation angle decribed above. for a Cn rotation in + or - direction, their eigenvalues and eigenvectors are the same, excepte the eigenvectors change the correspondance of the eigenvalues.
    
    
    Args:
    ------
        mat (np.ndarray): rotation matrix. np.ndarray (3,3)

    Return :
    ------
        axis : np.ndarray (3,). 
        det : determinant, denote if the rotation matrix is proper (det == 1) or improper (det == -1).
        angle : angle of proper or improper rotation. in radian unit.
        exception for identity :  axis = None, det = 1, angle = 2 * pi.

    Note :
    #### All output are affine to a close interger with toleranc 1e-5. Except for the angle, it's first affine to an integer in degree unit, then to radian.
    #### in fact the rotoinversion angle delta is equal to rotation angle 2 * delta (2 in crystal case) and mirror perpendicular to axis. So the eigenvalue in matrix is -1 and the rest correcpond to the 2 * delta rotation.
    """ 
    assert np.shape(mat) == (3,3), "mats should be a 3x3 matrix"
    # assert np.shape(prim_to_cart) == (3,3), "the matrix should be a 3x3 matrix"
    if not np.allclose(mat, np.identity(3)):
        # axis = np.linalg.solve(mat - np.eye(3), np.zeros(3))
        eigenvalues, eigenvectors = np.linalg.eig(mat)

        det_init = np.linalg.det(mat)
        det = np.rint(det_init)
        # Find which eigenvalue is (closest to) 1
        # (due to numerical precision, it might not be exactly 1)
        if det == 1:
            idx = np.argmin(np.abs(eigenvalues - 1.0))
        elif det == -1:
            idx = np.argmin(np.abs(eigenvalues + 1.0))
        else :
            print(mat)
            raise ValueError(f"determinant is {det_init}, too far from 1 or -1")
        
        # The corresponding eigenvector is the rotation axis
        axis = np.real(eigenvectors[:, idx])  # Take real part
        # if np.allclose(axis, np.rint(axis), atol=1e-5):
        #     axis = np.rint(axis)
        # else :
        #     raise ValueError(f"axis is {axis}, some element too far from an integer")
        
        if idx != 0: #### we know we have 3 eigenvalues but just one is real, and they represent the same angle but in opposite sense., so the real one is in an index or others. 
            eg_complex = eigenvalues[0]
        else :
            eg_complex = eigenvalues[1]
        angle = np.angle(eg_complex) ### angle is the angle of the eigenvalue
        degree_angle =  180. / ( np.pi)  * angle

        #### refind the value of det, axis, or angle cause they should all be interger. With a tolerance 1e-6
        if abs(det - np.rint(det)) < 1e-5:
            det = np.rint(det_init)
        else :
            raise ValueError(f"determinant is {det_init}, too far from 1 or -1")
        
        if abs(degree_angle - np.rint(degree_angle)) < 1e-5:
            # angle = np.rint(angle)
            pass
        else :
            raise ValueError(f"angle is {angle}, too far from an integer")
        
        
        ### some post processing to get the angle in radian unit
        if angle == 0 : ### this case can only be in pi rotation. cause in the if above i have already exclude the C1.
            angle = np.pi
        

        if det == -1 :
            

            if angle == np.pi/2 :
                pass
            elif angle != np.pi and angle != np.pi/2: #### in fact the rotoinversion angle delta is equal to rotation angle 2 * delta (2 in crystal case except C2i : mirror) and mirror perpendicular to axis. So the eigenvalue in matrix is -1 and the rest correcpond to the 2 * delta rotation.
                angle = angle / 2
        
        
    else :
        axis = None,
        det = 1
        angle = 2* np.pi
    return axis, det, angle









###########
########### operations with choise of basis below. and give the possibility to check if the spglib find good space group.


    

def operate_to_unit_cell(coord : np.ndarray, affine_matrix : np.ndarray , basis_trans : np.ndarray = None) -> np.ndarray:
    """
    return the coordinate (with unit cell vectors basis) in unit cell after applying the operation to coord
    ## This function can't treated : the rotation matrix in cartesian basis but the translation part is scaled.
    ###  e.g. SymmOp from pyxtal.symmetry.Group() objet.

         
    Parameters :
    ----------
        coord (np.ndarray) : the coordinate in unit cell vectors basis, it should be expanded to 4D array, the 4th element is 1.
        affine (np.ndarray) (4x4) : the operation {R | t} affine matrix to apply. Normally its affine matrix is cartesian coordinate.
                                                                    If not , the matrix init | after are same. 
        basis_trans : transformation the basis of affine_matrix from init affine matrix basis to coord's basis, by default is None, means don't need to apply the transformation.

                    ## So it's for user to make sure the basis_trans respect definition of above.
    
    Return :
    -------
    coord_target_unit_cell : np.ndarray : the coordinate in unit cell vectors basis after applying the operation to coord and round the coord into the unit cell.
    coord_target_out_unit_cell : Bool, if the  coordinate in unit cell vectors basis after applying the operation to coord is out of the unit cell.


    # Note :
    --------
        the unit_cell should be more correctlly called as the primitive cell. But it's just a code
        This function is correct in math, 
        but if the bravais lattice's basis vector (unit_cell in parameter) 's a or b is both no colinear to x or y in cartesian coordinate, 
        for operation having axis in plane xy (or ab) the matrix output 's matrix element can be diffrent to 0 and -1 and 1.
        
    ##    so it's suggest that at least a basis vector a or b is colinear to x or y in cartesian coordinate.
    --------
        e.g.
        ----------------
            For a hexagonal cell array([[ 8.90167342, -4.45083671,  0.        ],
                                        [ 0.        ,  7.70907532,  0.        ],
                                        [ 0.        ,  0.        , 23.8718379 ]])
            it's a is collinear to x.
            if rotation of this cell by +10 degree : it give array([[ 8.766437  , -5.72188538,  0.        ],
                                                                    [ 1.54575937,  6.81907746,  0.        ],
                                                                    [ 0.        ,  0.        , 23.8718379 ]])
            
            Mirror in plane yz in cartesien give : 
            array([[-1.,  0,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0.,  1.]])
            
            in cell non rotated by +10 degree, result is : 
            array([[-1.,  1.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0.,  1.]])
            if cell is rotated by +10 degree, result is :
            array([[-0.7422272 ,  1.13715804,  0.        ],
                    [ 0.39493084,  0.7422272 ,  0.        ],
                    [ 0.        ,  0.        ,  1.        ]])
    """
    assert np.shape(coord) == (3,), "coord (coordinate) should be a 1D array of length 3"
    # coord_carte = unit_cell @ coord
    coord = np.array(coord)
    coord = coord % 1 #### the initial coordinate can have value out of invertalle[0,1] , note: 1 % 1 = 0
    coord_affine = np.append(coord, 1.)
    # unit_cell_affine = np.append(unit_cell, 1)
    assert np.shape(affine_matrix) == (4,4), "the affine matrix should be a 4x4 matrix"

    # rotation_part = affine_matrix[:3, :3]
    # translation_part = affine_matrix[:3, 3]

    
    # ###
    # rotation_part_scaled = change_matrix_basis_from_eucli_to_frac(rotation_part, unit_cell) ### change the affine matrix basis

    # #### if translation part is base on basis vector of bravais lattrice
    # ## normally, all SymmOp's translation part are scaled
    # is_translation_part_scaled = check_if_translation_is_scaled(translation_part)
    # if not is_translation_part_scaled:
    #     translation_part_scaled = np.linalg.inv(unit_cell) @ translation_part
    # else :
    #     translation_part_scaled = translation_part
    
    # affine_matrix_scaled = np.zeros((4,4))
    # affine_matrix_scaled[:3,:3] = rotation_part_scaled
    # affine_matrix_scaled[:3,3] = translation_part_scaled
    # affine_matrix_scaled[3,3] = 1
    # coord_target =  affine_matrix_scaled @ coord_affine
    # coord_target_unit_cell = coord_target % 1 ### the coordinate in the unit cell
    # coord_target_unit_cell = coord_target_unit_cell[:-1]


    if basis_trans is not None:
        affine_matrix = change_basis(affine_matrix, basis_trans)

    coord_target = affine_matrix @ coord_affine
    coord_target_unit_cell = coord_target % 1 ### the coordinate in the unit cell
    coord_target_out_unit_cell = False
    if not np.allclose(coord_target_unit_cell[:3], coord_target[:3],atol=symprec):
        coord_target_out_unit_cell = True

    
    coord_target_unit_cell = coord_target_unit_cell[:-1]
    return coord_target_unit_cell, coord_target_out_unit_cell

def set_value_close_to_1_to_0(coord : np.ndarray):
    """
    we know that at fractionnal coordinate, zero is same as 1. But operate_to_unit_cell() can give as coordinate component close to 1 but <1. 
    cause i set the absolute tolerance to 1e-5 for fractionnal coordinate component. So, i set the value between 1-symprec and 1 to 0.
    
    
    
    """
    coordd = coord.copy()
    for i in range(3):
        if coordd[i] > 1-symprec:
            coordd[i] = 0
    return coordd


def symmetry_related(coord1 : np.ndarray, coord2 : np.ndarray, opa : np.ndarray, basis_trans : np.ndarray = None, tol : float = symprec) -> bool:
    """
    if two points (with unit cell vectors basis) in unit cell are symmetry related by opa with a relative tolerance (tol) for each component, return True. \

    So first, apply opa to coord1 to see if it find coord2, and inversly. if one of them works, than return True. 
    e.g. :  coord1 +120 to coord2 or coord2 +120 to coord1. means they are symmetry related by 120 degree rotation 
    In fact, this tell you these coordinates are equivalent, cause the definition of equivalent point is below 
        op(pos) = pos + A * a + B * b + C * c (a, b, c are basis vectors of the unit cell, A, B, C are integers)  

    You set a basis_trans matrix only if the opa is not in the same coordinate to coord1 and coord2. 


    Parameters  :
    ---------
    coord1 : np.ndarray : the first coordinate  (3,)
    coord2 : np.ndarray : the second coordinate (3,)
    opa : np.ndarray (4,4): the affine matrix of the operation whom's basis can be different to the coord1 and coord2. If it's different, set a basis_trans who is not the identity. 
    basis_trans : np.ndarray (3,3). transformation the basis of affine_matrix from init affine matrix basis to coord's basis, by default is None, means don't need to apply the transformation.
    
    Return :
    -------
    bool_equal_1 or bool_equal_2 : True if the two points are symmetry related, False otherwise.

    was_out_of_unit_cell : bool : True if the in one of the operation validated made the unit cell out of the unit cell. False otherwise.
    
    
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    
    assert np.shape(coord1) == (3,), "coord1 (coordinate) should be a (3,) nd.array"
    assert np.shape(coord2) == (3,), "coord2 (coordinate) should be a (3,) nd.array"
    assert np.shape(opa) == (4,4), "the affine matrix should be a 4x4 matrix"
    coord1_target, was_out_of_unit_cell_1 = operate_to_unit_cell(coord1, opa, basis_trans=basis_trans)
    coord1 = coord1 % 1 
    coord1 = set_value_close_to_1_to_0(coord1)
    
    
    coord1_target = set_value_close_to_1_to_0(coord1_target)
    # opa_inverse = np.transpose(opa) #### inverse the operation. like +120 to -120 degrees. it works also for mirrors. but since the operation matrix is orthogonal, its transpose is equal to inversion
    coord2_target, was_out_of_unit_cell_2 = operate_to_unit_cell(coord2 , opa, basis_trans=basis_trans)
    coord2_target = set_value_close_to_1_to_0(coord2_target)
    coord2 = coord2 % 1
    coord2 = set_value_close_to_1_to_0(coord2)
    
    bool_equal_1 = np.allclose(coord1_target, coord2,  atol=tol)    
    bool_equal_2 = np.allclose(coord2_target, coord1,  atol=tol)
    was_out_of_unit_cell = False
    if bool_equal_1:
        bool1_out_1 = was_out_of_unit_cell_1
    else :
        bool1_out_1 = False
    if bool_equal_2:
        bool1_out_2 = was_out_of_unit_cell_2
    else :
        bool1_out_2 = False
    
    was_out_of_unit_cell = bool1_out_1 or bool1_out_2
    return bool_equal_1 or bool_equal_2, was_out_of_unit_cell



# def apply_op(affine_mat : np.ndarray, coord1 : np.ndarray):
#     """
#     opa : SymmOp 
#     coord1 : np.ndarray
#     their basis vectors are basis vectors of the unit cell.

#     return : coordinate of coord1 in unit cell after applying affine_mat
#     """
#     coord1_uc = coord1
#     coord1_uc -= np.floor(coord1) ### coord1 dans unit cell
    
#     op_coord1_uc = affine_mat @ coord1_uc
#     op_coord1_uc -= np.floor(op_coord1_uc)

#     return op_coord1_uc
    



def group_Z_numbers_index(numbers : list[int]):
    """
    return the same elements and their index : {Z : index}
    """
    result = {}
    for index, num in enumerate(numbers):
        if num in result:
            result[num].append(index)
        else:
            result[num] = [index]
    return result


def affine_from_rot_and_trans(rots : np.ndarray, trans : np.ndarray) -> list[np.ndarray]:
    """
    just combine a list of rotation matrix and translation vector (according to the same index in their list) to a list of affine matrix.
    
    """
    assert np.shape(rots)[1] == 3 and np.shape(rots)[2] == 3 and len(np.shape(rots)) == 3, "rots should be a set of 3x3 matrix arranged along the axis 0 of trans"
    assert np.shape(trans)[1] == 3 and len(np.shape(trans)) == 2, "trans should be some 3x1 matrix arranged along the axis 0 of trans"
    assert np.shape(rots)[0] == np.shape(trans)[0], "should have the same number of rotation matrix and translation vector, even the translation vector is a vector 0"
    affine_matrixs = []
    for i in range(np.shape(rots)[0]):
        affine_matrix = np.identity(4)
        affine_matrix[:3,:3] = rots[i]
        affine_matrix[:3,3] = trans[i]
        affine_matrixs.append(affine_matrix)
    return affine_matrixs

def all_remain_same_after_some_op(atoms : Atoms, opas : list[np.ndarray]) -> bool:
    """
    if a unit cell under all affine matrix in operations in opas is itself. 
    "is itself" : all atoms under operation op will go to the position of the inital position add a linear combination of basis vectors of the unit cell.
                    op(pos) = pos + A * a + B * b + C * c (a, b, c are basis vectors of the unit cell, A, B, C are integers)
    
    This funciton will be used to check if the space group is correctly find by spglib, ase ,etc.
    The idea is for each operation, transform an atom, see if it's position after transformation have a atoms of the same Z number.
    This should be validate for all operations in the space group.

    #### In this function, coordinates are in primitive cell of ase objet, but since the transformation matrix are given by the spglib.
    #### So, the basis of the transformation matrix are the basis of the standard cell define in spglib. There is a transformation_matrix in spglib.symmetry_dataset who is define as
    #### std_cell = prim_std_trans @ prim_cell + origine shift, but we don't care the origine shift, cause all we care are symmetry operation. and usually the origine shift is a vector 0.
    #### you can see below the prim_std_trans matrix is given as define above.
    """
    
    scal_positions = atoms.get_scaled_positions()
    Z_numbers = atoms.get_atomic_numbers()
    Z_indexs = group_Z_numbers_index(Z_numbers)
    
    # prim_std_trans = spglib.get_symmetry_dataset((atoms.get_cell(), scal_positions, Z_numbers), symprec=symprec).transformation_matrix

    #### loop on all operations
    for opa in opas:
        ### calculate all scales position after the operation opa
        # scal_positions_after = []
        # for pos in scal_positions:
        #     pos_op = operate_to_unit_cell(opa=opa, coord=pos, unit_cell=cell )
        #     scal_positions_after.append(pos_op)

        # scal_positions_after = np.array(scal_positions_after)

        for Z in Z_indexs:
            indexs = Z_indexs[Z]
            
            # Z_scal_pos_after = scal_positions_after[indexs]
            Z_scal_pos = scal_positions[indexs]
            ### compare the positions befor after of the same Z number. If one match (one symmetry_related return true), the op is good for this position befor
            is_close = False
            for pos in Z_scal_pos:
                for pos_op in Z_scal_pos:
                    is_sym, _ = symmetry_related(opa= opa, coord1=pos, coord2=pos_op) ### opa are affine matrix in basis of atoms object's cell. so dont need to convert basis.
                    is_close = is_close or is_sym
                    if is_close:
                        break
            
                else : ### code will execute else if inner for loop is finished. Then continue will be executed. Than break will be skip
                    continue
                break #### so, inversly, if the inner loop is break above, else will not be execute. Than this break will be executed
            if not is_close:
                print(f"pos : {pos}, Z_scal_pos : {Z_scal_pos}")
                print(f"opa : {opa}")
                pos_target, did_pos_in_unitcell = operate_to_unit_cell(coord=pos, affine_matrix=opa)
                print(f"pos_target : {pos_target}")
                
                print("")
                print("")
                return False ### if a atom can't find its corresponding position, the symmetry of opa is not validated, return False

        ### ancien version
        # for pos in scal_positions: 
        #     pos_op = apply_op(opa, pos)
        #     scal_positions_after.append(pos_op)

        # #### check if under this opa the unit cell remain the same

        # # for pos_op in scal_positions_after:
        # #     for pos in scal_positions:
        # #         is_close = np.allclose(pos, pos_op)

        # for Z in Z_indexs:
        #     indexs = Z_indexs[Z]
        #     Z_scal_pos_after = scal_positions_after[indexs]
        #     Z_scal_pos = scal_positions[indexs]
        #     ### compare the positions befor after of the same Z number. If one match, the op is good for this position befor
        #     is_close = False
        #     for pos in Z_scal_pos:
        #         for pos_op in Z_scal_pos_after:
        #             is_close = is_close or np.allclose(pos, pos_op, rtol=1e-3)
        #             if is_close:
        #                 break
        #         else : ### code will execute else if inner for loop is finished. Then continue will be executed. Than break will be skip
        #             continue
        #         break #### so, inversly, if the inner loop is break above, else will not be execute. Than this break will be executed
        #     if not is_close:
        #         return False ### if a atom can't find its corresponding position, the symmetry of opa is not validated, return False
        
    
    return True ### if all atoms find its corresponding position after opa is the same element.
                




        

def get_space_point_group_spglib(atoms : Atoms, tol = symprec):
    """
    the final version to read a cif file.

    Parameter : 
    ---------
            atoms : ase.atoms.Atoms objet.
                tol :  tolerance (see doc spglib get_symmetry_dataset)
    
    Return :
    --------
            space_group : tuple (HM symbol, international number)
            point_group : HM symbol
            site_wyckoff_symmetry :  list of [Z number of the atoms at a site, site wyckoff position letter, site symmetry symbol in hm notation, fractional coordinate of the atom site] 
            abc_vector : crystal lattrice basis vector who form a primitive cell in coordinate cartesien.
            spg_ops : list of space group operation on the basis of the cell from atoms.get_cell()(see doc spglib get_symmetry_dataset)
            prim_std_transformation : transformation matrix from the primitive cell to the standarize cell (std_cell = P @ prim_cell + origine shift, but we dont care the origin shift, cause all we care are symmetry operation.) define by spglib : https://spglib.readthedocs.io/en/stable/definition.html#rotation-introduced-by-idealization. 
    """
    ### acte meilleur que py
    ### entrée : atoms : sortie de read_cif_ase()
    
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    # abc_vector = np.transpose(cell[0])
    abc_vector = cell[0]
    symmetry_data = spglib.get_symmetry_dataset(cell,symprec=tol) ### well, finalement j'ai choisi symprec par défaut. Mais n'oublie pas ce tolérance peut affecté l'analyse de symétrie
    space_group = (symmetry_data['international'], symmetry_data['number'])
    point_group = symmetry_data['pointgroup']

    rot_mats = symmetry_data['rotations']
    translation_mats = symmetry_data['translations']

    for i, trans in enumerate(translation_mats):
        translation_mats[i] = set_value_close_to_1_to_0(trans)


    spg_ops = affine_from_rot_and_trans(rot_mats, translation_mats)
    #### the transformation from the primitive cell to the standarize cell is necessary :
    ##### cause all rotation matrix respect to the standardize cell. But the cell above is the primitive cell.

    prim_std_transformation = symmetry_data['transformation_matrix']
    

    # site_symmetrys = symmetry_data['site_symmetry_symbols']
    site_symmetrys_hm = symmetry_data.site_symmetry_symbols
    wyckoff_pos = symmetry_data.wyckoffs
    Z_number = atoms.get_atomic_numbers()
    site_wyckoff_symmetry = []
    
    scaled_positions = cell[1].tolist()
    
    for i in range(len(wyckoff_pos)):
        
        site_wyckoff_symmetry.append([Z_number[i], wyckoff_pos[i], site_symmetrys_hm[i], scaled_positions[i]])
    return space_group, point_group, site_wyckoff_symmetry, abc_vector, spg_ops, prim_std_transformation


def get_symm_ops_of_site_symmetry_from_wp(group_number : int, wp_letter : str):
    """
    ## Deprecated, see get_site_point_group_from_scale_rots who use directely the point group matrix in fractionnal basis.
    return the site symmetry symbol (HM) and all rotation matrix of this site.

    Note, initaly used to connect the spglib dataframe to the symmetry matrix.

    entrer : group_number : space group number in ITA, 
            wp_letter : wyckoff position on letter.

    return : site_symm_symbol : SymmOp objet of pymatgen
            all_rotation_matrix_site_symm : ndarry:  all rotation matrix of all symmetry element on this site (in fractionnal coordinate).
            
            Note  : all rotation matrix of an axe of rotation (proper or improper) is 2*pi/n in direction + and -. 
                    rotation matrix of  m*2*pi/n (0<m<1) is not include. It's not necessary.
    """
    g = Group(group_number, dim=dimension)
    wp_pos = g.get_wyckoff_position(wp_letter)
    site_symm_objet= wp_pos.get_site_symmetry_object()
    site_symm_symbol = site_symm_objet.symbols
    # are_matrix_euclidien = wp_pos.euclidean
    all_rot_mat_site_symm_eucli = []
    site_symmetry_ops_eucli = wp_pos.get_site_symm_ops()
    for op in site_symmetry_ops_eucli:
        rot_mat_eucli = op.rotation_matrix
        all_rot_mat_site_symm_eucli.append(rot_mat_eucli)
    all_rot_mat_site_symm_eucli = np.array(all_rot_mat_site_symm_eucli)

    wp_pos.euclidean = False ### ensure the matrix are in the crystal unit cell basis
    site_symmetry_ops = wp_pos.get_site_symm_ops()

    
    # symmetry_elements = site_symm_objet.to_matrix_representation()
    # operation_analysers = site_symm_objet.opas
    # for opa in operation_analysers :
    #     axis = opa.axis ## remember that all mirror is rotoinversion with a -2 axis (so order = 2) perpendicular to the mirror and point to the side with the initial position
    #     orders = opa.order ## the order of the rotation axis.
        # print(opa)
        # print(axis)
        # print(orders)
        # print("xx")

    all_rotation_matrix_site_symm = []
    for op in site_symmetry_ops:
        rotation_matrix_site_symm = op.rotation_matrix
        all_rotation_matrix_site_symm.append(rotation_matrix_site_symm)
    all_rotation_matrix_site_symm = np.array(all_rotation_matrix_site_symm)
    return site_symm_objet, all_rotation_matrix_site_symm, all_rot_mat_site_symm_eucli


def find_principal_axis_from_site(site_symm_objet : site_symmetry ):
    """
    ### Deprecated, see find_principal_axis_and_order_from_scaled_rots()
    get site's principal axis in cartesian basis. 

    Here we use the site_symmetry object from pymatgen. It's a object who contain all symmetry operation of a site.

     
    """
    opas = site_symm_objet.opas ### a list of OperationAnalyser
    
    order_max = 0
    principal_axis = np.array([])
    type_axis = ''
    for opa in opas :
        axis = opa.axis ## remember that all mirror is rotoinversion with a -2 axis (so order = 2) perpendicular to the mirror and point to the side with the initial position
        order = opa.order ## the order of the rotation axis.
        # rot_matrix = opa.rotation_matrix
        affine_matrix = opa.affine_matrix
        tran_vec = affine_matrix[:,-1]
        

        if order > order_max:
            order_max = order
            # axis = axis.tolist()
            # if axis == None :
            #     principal_axis = [] ### show that if principal_axis == None, the group is C1
            # else :
            #     principal_axis = axis.tolist()
            try :
                principal_axis = axis.tolist()
            except :
                if axis == None :
                    principal_axis = [] ### show that if principal_axis == None, the group is C1
            type_axis = opa.type
    return order_max, principal_axis, type_axis




def change_matrix_basis_from_eucli_to_frac(matrix, abc_vector):
    """
    #### this function is not for affine matrix but for rotation matrix.

    simply convert the matrix in euclidian basis to fractionnal basis
    The abc_vector need to put a, b ,c vector along the column.

    the function firstly check if matrix is orthogonal. If not, return the input matrix without changing the basis by abs_vector.
    
    Parameter :
        matrix : 3x3 matrix, we can't tell you if the matrix is orthogonal or not if it's a 4x4 matrix. So this function is not for affine matrix.
        abc_vector : 3x3 matrix with a, b, c vectors in old basis along the column.
    """
    # abc_vector = abc_vector/(np.linalg.norm(abc_vector[0])) ### normalize the lentgh of vector abc cause the error in the translation.
    # det = np.linalg.det(abc_vector)
    # assert abs(det) > 1 ,"unit cell basis vector have volum to small (smaller than 1, unity is angstrom^3)"
    # cond = np.linalg.cond(abc_vector)
    # assert cond < 50, f"unit cell basis vector are too dependant between them, condition is {cond}"
    assert np.shape(matrix) == (3,3), "the matrix should be a 3x3 matrix"
    is_matrix_orthogonal = new_is_orthogonal(matrix)
    
    if  is_matrix_orthogonal:
        # abc_vector_invert = np.linalg.inv(abc_vector)
        # # matrix_abc = np.dot(np.dot(abc_vector_invert, matrix), abc_vector)
        # matrix_abc = abc_vector_invert @ matrix @ abc_vector
        matrix_abc = change_basis(matrix, abc_vector)
        matrix_abc = matrix_abc[:3,:3]
        
    else :
        matrix_abc = matrix
    return matrix_abc


def site_point_group_from_a_material(atoms : Atoms,  test = False):
    """
    for each site in the crystal find its site point group.

    Parameter  : 
    ---------
                atoms : ase.atoms objet

    Return : 
    -------
            space_group : see get_space_point_group_spglib()
            point_group : see get_space_point_group_spglib()
            site_wyckoff_symmetry : see get_space_point_group_spglib()
            abc_vector : see get_space_point_group_spglib()
            all_sites_info : list of information list of a atom site site_info:  #### can see in the code below the comment
                                                                            [[Z number, 
                                                                            site symmetry symbol in point group of HM notation, 
                                                                            site symmetry symbol in point group of Sch notation, 
                                                                            site symmetry symbol in ITA, 
                                                                            wyckoff letter, 
                                                                            fractional coordinate,
                                                                            order of principal axis,
                                                                            principal axis,
                                                                            type of principal axis: inversion, rotation....,
                                                                            have_non_symmorphic_site_ops : True if the site can be tranform to itself or equivalent pos under non symmorphic operation,
                                                                            was_out_of_unit_cell : if site have some non symmorphic operation, did this operation transforme site pos to a equivalent pos (True) or itself (False)
                                                                            ], ...]
            spg_ops_matx_scaled : see get_space_point_group_spglib()
            prim_std_trans : see get_space_point_group_spglib()


            if test == True, return also the following
                error_eucli_abc_convert : int. Number of site symmetry operation for which the conversion from euclidian to fractional basis (method initial) failed.
                N_op_total : number of total symmetry operation from each site
            
            ##### This test is commanded
    """

    space_group, point_group, site_wyckoff_symmetry, abc_vector, spg_ops_matx_scaled, prim_std_trans= get_space_point_group_spglib(atoms)
    # space_group_number = space_group[1]
    # abc_vector = atoms.get_cell() ### just ensure that 
    # abc_vector_tran_mat = np.transpose(abc_vector) ### in abc_vector get from spglib, a, b, c vector are placed on each ligne. but in the transformation matrix. a, b, c should be at the column.


    

    ## test : visualize matrix resulted by get_symmetry(). To delect
    # cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    # symmetry = spg.get_symmetry(cell)

    # error_eucli_abc_convert = 0
    # N_op_total = 0
    N_site_point_group_wrong = 0
    ### iteration to calculer site point group of each site.
    have_non_symmorphic_site_ops = False
    all_sites_info = []
    for site in site_wyckoff_symmetry : 
        wp_letter = site[1]
        Z_number = site[0]
        site_symmetry_ITA = site[2]
        scaled_position = site[3] ## fractional coordinate of the site
        # site_symm_objet, site_symm_ops_matrix_scaled, site_symm_ops_matrix_eucli = get_symm_ops_of_site_symmetry_from_wp(space_group_number,wp_letter)
        # # site_symm_ops_abc = []
        # order_max, principal_axis, type_axis = find_principal_axis_from_site(site_symm_objet)

        
        # # if principal_axis != []:
        # #     # prim_std_trans = np.rint(prim_std_trans)
        # #     # try :
        # #     #     prim_std_trans_inv = np.linalg.inv(prim_std_trans)
        # #     # except :
        # #     #     print(prim_std_trans)
        # #     #     print(atoms)
        # #     principal_axis = np.linalg.inv(prim_std_trans) @ np.array(principal_axis)
        # #     principal_axis = principal_axis.tolist() 
        
        # # site_symm_ops_abc = np.array(site_symm_ops_abc) 
        
        # # print(site_symm_ops_abc)
        
        # site_point_group_HM = spglib.get_pointgroup(rotations = site_symm_ops_matrix_scaled)[0]
        site_point_group_HM, site_point_group_rots, have_non_symmorphic_site_ops, was_out_of_unit_cell = get_site_point_group_from_scale_affines(scaled_position, spg_ops_matx_scaled)
        principal_axis, order_max,  type_axis = find_principal_axis_and_order_from_scaled_rots(site_point_group_rots)
        
        

        site_point_group_sch = point_group_hm_sch[site_point_group_HM]

        
        
        site_info = [Z_number, 
                     site_point_group_HM, 
                     site_point_group_sch, 
                     site_symmetry_ITA, 
                     wp_letter, 
                     scaled_position, 
                     order_max, 
                     principal_axis, 
                     type_axis, 
                     have_non_symmorphic_site_ops,  ### do this site point group have validated non symmorphic operation
                     was_out_of_unit_cell ### if this site point group have validated non symmorphic operation, did it make site position out of unit cell?
                     ] 
        all_sites_info.append(site_info)


        ### just for test, this if dont depend on the code above
        #### convert each rotation matrix to basis abc. This standard suit to the spg.get_pointgroup funciton below
        # if test :
        #     index = 0
        #     site_symm_ops_abc = []
        #     for op_eucli in site_symm_ops_matrix_eucli: 
        #         op_abc = change_matrix_basis_from_eucli_to_frac(op_eucli, abc_vector_tran_mat)
        #     ### here it can produce the error compare to the scaled matrix in site_symm_objet. In fact we can get the matrix in the get_symm...from_wp by setting the attribut of wykoff position .euclidiean = False to get the scaled matrix.
        #         op_abc_rint = np.rint(op_abc)
        #         # idem_matrix = np.allclose(op_abc, op_abc_rint, rtol = 1e-2)
        #         idem_matrix = np.allclose(op_abc_rint, site_symm_ops_matrix_scaled[index], rtol = 1e-2)
        #         if not idem_matrix :
        #             error_eucli_abc_convert += 1
        #         index += 1
        #         N_op_total += 1
        #         site_symm_ops_abc.append(op_abc_rint)
        #     site_symm_ops_abc = np.array(site_symm_ops_abc) 
        #     error_site_point_group_HM = spglib.get_pointgroup(rotations=site_symm_ops_abc)[0]
        #     if error_site_point_group_HM != site_point_group_HM:
                # N_site_point_group_wrong += 1
    
    # if test :
    #     return space_group, point_group, site_wyckoff_symmetry, abc_vector, all_sites_info, point_group_rot_matx_scaled, error_eucli_abc_convert, N_op_total, N_site_point_group_wrong
    
    return space_group, point_group, site_wyckoff_symmetry, abc_vector, all_sites_info, spg_ops_matx_scaled, prim_std_trans, have_non_symmorphic_site_ops

        
def check_if_string_disorded(str1 : str, str2 : str):
    """
    check if the site point group in HM symbol is a string simplify from site symmetry in ITA (who is result by spglib, defintly correct.)

    here simplify mean removing the '.' and the order or each character are disordered. '-n' is consider as a charecter.

    i have a doute if the site symmetry is directly the site point group. at least in the site_point_group_from_a_material() i 
    identify the site point group by the symmetry operation of each site.
    """
    
    str2 = str2.replace('.', '')
    Bool = True
    for i in range(len(str2)):
        c = str2[i]
        if c != '-':
            index = str1.find(c)
        else :
            c = c + str2[i+1]
            index = str1.find(c)

        if index == -1:
            Bool = False
            return Bool
    
    
    return Bool
    
    
def bi_check_if_string_disorded(str1 : str, str2 : str):
    Bool1 = check_if_string_disorded(str1, str2)
    Bool2 = check_if_string_disorded(str1, str2)

    return Bool1 and Bool2


###############
###############
############### to find site point group from a material, i've decide to don't use the pymatgen matrix. cause i dont know the basis vector's of the matrix.


### error case to global
wear_spg_ops = []


def get_site_point_group_from_scale_affines(pos : np.ndarray, scaled_affines : list[np.ndarray]):
    """
    Arg :
    ----
    scale_affines : (n,4,4) array of rotation matrix in fractional coordinate of a space group. This space group is the space group of crystal which the pos belong to.
    pos : a atoms position in fractional coordinate. It shoul have the same basis vector as the rotation matrix in scale_rots.
    
    The idea is the by definition of site point group: Space group operation where all operation on the position make the position into itself. (initial position or it's translation.)
    #### Note, it's the correct definition. Cause the screw axis and glide plane's rotation part is include in the point group of a space group. But a screw axis or glide plane can't not include into the site point group.
    
    Return : 
    -------
        site_point_group : str : the site point group in HM notation.
        site_point_group_ops : list of all rotation matrix of the site point group.
        have_site_ops_non_symmorphic : do validated space group operation have some nonprimitive translation part.
        was_out_of_unit_cell : True if have_site_ops_non_symmorphic is True and one of non symmorphic operation (contain nonprimitive translation part) operated the coordinate out of unit cell.
    """
    assert np.shape(pos) == (3,), "pos should be a 1D array of length 3"
    assert np.shape(scaled_affines)[1] == 4 and np.shape(scaled_affines)[2] == 4, "scaled_affines should be a 3D array of shape (n,4,4)"
    global wear_spg_ops
    site_point_group_ops = []
    
    
    have_site_ops_non_symmorphic = False
    for i, affine in enumerate(scaled_affines):
        
        
        is_sym_related_to_itself, was_out = symmetry_related(coord1=pos, coord2=pos, opa=affine)
        was_out_of_unit_cell =  False
        if is_sym_related_to_itself:
            site_point_group_ops.append(affine[:3,:3])
            if np.linalg.norm(affine[:3,3]) > symprec: ### if translation part is not 0, i cant understant for now.
                have_site_ops_non_symmorphic = True
                wear_spg_ops.append(affine)
                was_out_of_unit_cell = was_out_of_unit_cell or was_out

                # print(f"the affine matrix {affine} is not a pure rotation matrix. But valide to {pos}")
            
            
    
    try :
        site_point_group = spglib.get_pointgroup(rotations=site_point_group_ops)[0]
    except :
        print(f"all spg ops : {scaled_affines}")
        print('\n')
        print(f"all spg ops site : {site_point_group_ops}")
        raise ValueError(f"site_point_group_ops : {site_point_group_ops} is not a valid point group for {pos}")
    return site_point_group, site_point_group_ops, have_site_ops_non_symmorphic, was_out_of_unit_cell

def find_principal_axis_and_order_from_scaled_rots(scaled_rots : np.ndarray):

    """
    scale_rots : 3D array (n,3,3) of rotation matrix in fractional coordinate of a point group. This point group is the point group of crystal which the pos belong to.
    
    Return :
    -------
        principal_axis : principal axis in rotations's basis. have no direction (+ or -)
        order : the order of the principal axis.
        type_principale_axis : str :  f"C{order_max} improper" or f"C{order_max} improper" depending on the determinant of operation who's axis is principal axis.
    
    """
    assert np.shape(scaled_rots)[1] == 3 and np.shape(scaled_rots)[2] == 3, "scaled_rots should be a 3D array of shape (n,3,3)"
    try : 
        point_group_hm = spglib.get_pointgroup(rotations=scaled_rots)[0]
    except:
        
        raise ValueError(f"The entering scaled_rots {scaled_rots} can't form a valid point group.")
    ##### intitialize
    order_max = 0
    principal_axis = np.zeros((3,3))
    det_principal_axis = 0 ### it should be -1 and 1
    for rot in scaled_rots:
        axis, det, angle = find_rotation_axis(rot)
        order = np.rint(2*np.pi / angle)
        if order > order_max:
            order_max = order
            principal_axis = axis
            det_principal_axis = det
        elif order == order_max: ### this can check the case like C2i appear first and C2 appear secondly. But i want C2 as principal axis but no a mirror C2i
            if det_principal_axis == -1 and det == 1:
                principal_axis = axis
                det_principal_axis = det

    type_principale_axis = None
    if det_principal_axis == -1:
        type_principale_axis = f"C{order_max} improper"
    elif det_principal_axis == 1:
        type_principale_axis = f"C{order_max} proper"
    else:
        raise ValueError (f"The determinant {det_principal_axis} of principal axis's operation is not -1 or 1.")
    return principal_axis, order_max, type_principale_axis


def find_asymmetry_directions_by_atoms( atoms : Atoms):
    """
    just inversion atoms' scaled positions and check if the inversion give a element to another element.
    if not the atom's scaled position is a vector of asymmetry direction.

    Return 
    ------
        asymmetry_directions : list[np.ndarray] : list of asymmetry directions of atoms. For each vector representing the direction, their norm is Z number.

    """
    Z_numbers = atoms.get_atomic_numbers()
    scaled_positions = atoms.get_scaled_positions()
    Z_numbers_index = group_Z_numbers_index(Z_numbers)
    inversion_affine = expand_rot_mat_to_affine_mat(-1 * np.identity(3))

    asymmetry_directions = []
    for Z in Z_numbers_index:
        indexs = Z_numbers_index[Z]
        poss = scaled_positions[indexs]
        inversion_affine = expand_rot_mat_to_affine_mat(-1 * np.identity(3))

        have_inversion_partener = False
        for pos in poss:
            for pos_1 in poss :
                is_this_partener, _ = symmetry_related(pos, pos_1, inversion_affine)
                have_inversion_partener = have_inversion_partener or is_this_partener
                if have_inversion_partener:
                    break


            else : ### code will execute else if inner for loop is finished. Then continue will be executed. Than break will be skip
                continue
            break #### so, inversly, if the inner loop is break above, else will not be execute.


        if not have_inversion_partener:
            unit_vec = pos / np.linalg.norm(pos)
            Z_asymmetry_direction = unit_vec * Z
            asymmetry_directions.append(Z_asymmetry_direction)
    

    
    return asymmetry_directions
















###########
########### some manually set of matrix operations
def set_a_rotation_mat(angle_degree):
    """
    rotation matrix in cartesian coordinate.
    """
    angle_rad = np.deg2rad(angle_degree)
    mat = np.array([[np.cos(angle_rad), -1* np.sin(angle_rad), 0],
                    [np.sin(angle_rad), np.cos(angle_rad) , 0],
                    [0, 0, 1]])
    return mat

def expand_rot_mat_to_affine_mat(mat):
    assert mat.shape == (3, 3), "Input matrix should be 3x3"
    affine_mat = np.eye(4)
    affine_mat[:3, :3] = mat
    return affine_mat

def set_a_hex_cell(a_length, c_length):
    """
    hexagonal cell with a_length and c_length. a, b, c vectors are in columns of output matrix.
    angle between a, b is 120 degree.
    a is algne to x. c is algne to z.
    """
    a_vec = [a_length, 0, 0]
    b_vec = [a_length*np.cos(2*np.pi/3), a_length*np.sin(2*np.pi/3),0]
    c_vec = [0,0,c_length]
    cell_hex_eg = np.array([a_vec, b_vec, c_vec])
    cell_hex_eg = np.transpose(cell_hex_eg)
    return cell_hex_eg































############
############
############ 
############ Funcitons below are not validated or not used.


# def op_on_pos1_result_class_Position(pos1 : Position_in_unit_cell, ROT : Rotation_matrix_frac):
#     """
#     Op matrix on pt1
#     we assume here the op is in point group. so the op is a matrix array (3,3), so op is a Rotation_matrix class objet
#     """
#     op = ROT._matrix
#     pt1 = pos1._vector
#     pt1 -= np.floor(pt1)
#     pt1_op = np.dot(op,pt1)
#     pt1_op -= np.floor(pt1_op)

#     # diffs = pt2 - pt1_op
#     # dist = np.linalg.norm(diffs, axis=1)
#     # if dist < tol or dist == tol:
#     #     # print(dists)
#     #     return True
#     return pt1_op





# def is_op_endomorphisme_to_positions(E : Positions_in_unit_cell , ROT : Rotation_matrix_frac, tol = symprec*10):

#     """
#     check if the rotation matrix combine modulo 1 of the corrdinate is endomorphisme for a list of coordinate in a unit cell.
#     """
#     Bool = True
#     positions = E._positions
#     shape = np.shape(positions)
#     for i in range(shape[0]): ### iteration on all position
#         pt1 = positions[i,:]
#         pt1 = Position_in_unit_cell(Position(pt1))
#         pt1_op = op_on_pos1_result_class_Position(pt1, ROT)
#         pt1_op -= np.floor(pt1_op)
#         for j in range(shape[0]): ### iteration on all position execpt the position on index i
#             if  i != j : ### execpt the position on index i
#                 pt2 = positions[j,:]
#                 diffs = pt2 - pt1_op
#                 dist = np.linalg.norm(diffs)
#                 if dist > tol :
#                     Bool = False
#                     return Bool
#     return Bool




# def is_a_pointgroup_site_point_group(coord : np.array, mate : Cell, point_group_symbol_HM : str, tol = symprec):
#     """

#     Waring : never check if it's correct. To find the site point group, use site_point_group_from_a_material()



#     Check if all symmetrie operation of a point group make a site position coord invariant (coord remain the same before and after the symmetry operation)
#     Basicly, we select the atome of coord as a origine of the roation matrix. To see if all the others' positions are tranformed into the equivalent position (by using the is_op_endomorphisme_to_positions() function) in or ouside the unit cell. 
#             If all atoms are transform to their equivalent position, the site have such symmetry. If the site have all symmetry of a point group, the site symmetry is this point group.
#     Para:
#         coord : np.array represent the coordinate of the atom site. With basic vector of the unit cell
#         mate : mate : cell objet Who can give the information of the other atoms.
#         point_group_symbol_HM : point group symbol in HM.

#     Return bool : if true, the site point group is the point group in the argument.
#     """

#     positions = mate._positions
    
#     Z_numbers = mate._Z_numbers
#     exists = np.any(np.all(coord == positions, axis=1))

#     if not exists:
#         print("please enter a coordinate who existe in the objet mate")
#     ### convert the coordinates of all atoms in the cell
#     positions_convert =  []
#     for pos in positions : 
#         positions_convert.append(np.ndarray.tolist(pos - coord))

#     dico = {} ### un dictionnaire qui prend de format : {Z_number1 : [[coordinate1], coordinate2]...., Z_number2 : ..... }
#     for index, valeur in enumerate(Z_numbers):
#         if valeur not in dico:
#             dico[valeur] = []
#         dico[valeur].append(positions_convert[index])


#     Bool=True
#     point_group = PointGroup(int_symbol=point_group_symbol_HM)
#     symmops_point_group = point_group.symmetry_ops
#     for op in symmops_point_group :  #### iteration on different symmetry operation of the point group
#         # rotation_matrix = op.rotation_matrix
#         # coord_1 = np.dot(rotation_matrix, coord)
#         # diff = np.linalg.norm(coord_1 - coord)
#         # if diff > tol :
#         #     Bool = False
#         #     break
#         op = op.rotation_matrix
#         ROT = Rotation_matrix_frac(Rotation_matrix(op))
#         for Z in dico: ### iteration on different element
#             all_coord = dico[Z]
#             all_coord = np.array(all_coord)
#             all_coord = Positions_in_unit_cell(Positions(all_coord))
#             Bool = is_op_endomorphisme_to_positions(all_coord,ROT)
#             if not Bool :
#                 return Bool



#         #### check if equivalent for the transformation op in symmops_point_group
            

    
#     return Bool




# def liste_subgroup_of_pointgroup(point_group_symbol_HM):

#     """

#     Warning : may not used in the following code.


#     find all the point group which are subgroup of point group in entre:

#     Entre : Point group symbol in HM

#     Return : Bool

#     use function : is subgroup method


#     There is a warning of is_subgroup() into is_supergroup() who seem suiteable to our function: 
#         This will not work if the crystallographic directions of the two groups are different." --> NotImplementedError

#     Application here :    
#     In is_subgroup it ignore some operation represent by the affine matrix. For exemple, -6m2 have 2 fold rotation axsis 2 x,2x,0 (bilbao server).  
#     But you can't find point group 2 in the result. Because the representation of matrix is diff. So we need the representation of symbol of symmetry element

#     This funciton is not fully correct, we cans ignore some point group in return
    
    
#     ####
#     I just used the line set(pg.symmetry_ops).issubset(point_group.symmetry_ops) in the is_subgroup to skip the NotImplementedError who is make for the warning message above.

    

#     """

    

#     point_group = PointGroup(point_group_symbol_HM)
#     list_subgroups = []
#     for pg_sym in point_group_hm_sch:
#         pg = PointGroup(pg_sym)
#         # if point_group.is_supergroup(pg): ### "This is not fully functional. Only trivial subsets are tested right now. "
#         #                                     #####"This will not work if the crystallographic directions of the two groups are different." --> NotImplementedError
#         #     
#         if set(pg.symmetry_ops).issubset(point_group.symmetry_ops): ### i think can actually ignor the difference of crystallographic direction. Because all i want is the math subgroup, i ignore if the subgroup find can actually applied on the unit cell.
#             list_subgroups.append(pg_sym)


#     #### 2nd method : try to get the symbol of symmetry element or symmetry operation with pymsym and find the subgroup with these symbol.
#     # context = pys.Context()
#     # point_group_symbol_sch = point_group_hm_sch[point_group_symbol_HM]
#     # context._set_point_group(point_group_symbol_sch)
#     # ops = context._symmetry_operations
#     # context._update_character_table()
#     # character_table = context._character_table
#     # sym_species = character_table._fields_[3][1]
#     # print(sym_species)
#     # print(ops)
#     # for pg_sym in point_group_hm_sch:
#     #     pg = point_group_hm_sch[pg]
#     #     context_i = pys.Context()
#     #     context_i._set_point_group(pg)
#     #     ops_i = context_i._symmetry_operations
#     # return ops
#     return list_subgroups


    

#### site point group sur les différent site atome.

# def get_site_point_group_pyxtal(atoms):
#     ### pas du tout précise. abandonnée
#     all_xyz_positions = atoms.get_positions() 
#     all_abc_positions = atoms.get_scaled_positions()
#     all_atomic_numbers = atoms.get_atomic_numbers()
#     lattice = atoms.get_cell()
#     cell = (lattice, all_abc_positions, all_atomic_numbers)
#     group_symbol_ITA,  group_number = get_space_point_group_spglib(atoms, tol=symprec)[0]
    
#     g = Group(group_number, dim=dimension)

#     atoms_site_symmetry = []
#     for i in range(len(all_abc_positions)):
#         pos = all_abc_positions[i]
#         wp_pos = g.get_wyckoff_position_from_xyz(pos,decimals=7) 
#         atomic_number = all_atomic_numbers[i]
#         site_symmetry_ops = wp_pos.get_site_symm_ops()

#         site_symmetry_HM = wp_pos.get_site_symmetry_object()
#         site_symmetry_HM = site_symmetry_HM.symbols

        
        
#         atoms_site_symmetry.append([atomic_number, pos, site_symmetry_HM,site_symmetry_ops])
#         # site_symmetry_symbol_HM = 
#     return atoms_site_symmetry, group_symbol_ITA, group_number



    



    
    
    
    

    



    


    

    

    

    
    


    