from pwworkflow import workflow_scf_nscf

from aiida.engine import submit

from aiida.tools.visualization.graph import Graph

from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group, load_node, load_computer
from aiida import engine
from aiida.orm.nodes.process.calculation.calcjob import CalcJobNode

from aiida.engine import calcfunction, workfunction

from aiida import load_profile, get_profile
if get_profile() == None:
    load_profile()



def 