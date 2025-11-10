# In[]
from aiida import load_profile, get_profile
if get_profile() == None:
    load_profile()


from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, load_node
from aiida.orm.nodes.process.calculation.calcfunction import CalcFunctionNode, CalcFunctionNodeLinks
from aiida.tools.visualization import Graph



# In[]
import find_groups
import importlib
importlib.reload(find_groups)
from find_groups import *

import band_analyse 
importlib.reload(band_analyse)
from band_analyse import *

import analyse_symmetry
importlib.reload(analyse_symmetry)
from analyse_symmetry import *

import time

# In[]
#### Example code to Find the reciprocal space's special point and their point group
##### Example of MoS2 version -6m2

node_MoS2_6 = load_node('d25962ea-3c39-4e86-9333-2baf8f31a1a8')
ase_obj_MoS2_6 = node_MoS2_6.get_ase()


bravais_lattice = ase_obj_MoS2_6.cell.get_bravais_lattice()
special_points = bravais_lattice.get_special_points()
print(special_points)

# In[]
spg_number = get_space_point_group_spglib(ase_obj_MoS2_6)[0][1]
spg_objet = Group(spg_number)
symms_spg = spg_objet.get_spg_symmetry_object()

#In[]
pg_objet = PointGroup(spg_objet.point_group)
ops_pg = pg_objet.symmetry_ops

# In[]

# In[]

node_Hg4Mo2O8 = load_node('daaee44e-78b0-4002-85c9-25f55fcf3b31')
ase_obj_Hg4Mo2O8 = node_Hg4Mo2O8.get_ase()
bravais_lattice_Hg = ase_obj_Hg4Mo2O8.cell.get_bravais_lattice()