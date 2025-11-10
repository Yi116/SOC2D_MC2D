# In[]
from visualisation_stat import *


def test_stat_R2_D2_from_json():
    path_json = '/home/jyin/workspace/gene_data/direct_space_data.json'
    stat = stat_R2_D2_from_json(path_json)
    print(stat)

    labels = ["no R or D", "R-2 or D-2", "R-1 or D-1"]
    data = []
    for lab in labels:
        data.append(stat[lab])

    plot_circular(labels=labels, data=data)

# In[]
import importlib
import visualisation_stat
importlib.reload(visualisation_stat)
from visualisation_stat import *

# In[]
if __name__ == '__main__':
    test_stat_R2_D2_from_json()
    load_node()

# In[]
    spglib_data = None
    aiida_group_struc2D = load_group('structure_2D')
    for i, node in enumerate(aiida_group_struc2D.nodes):
        ase_objet = node.get_ase()
        chemical_formula = ase_objet.get_chemical_formula()
        
        if i == 5:
            print(ase_objet)
            cell = (ase_objet.get_cell(), ase_objet.get_scaled_positions(), ase_objet.get_atomic_numbers())
            spglib_data = spglib.get_symmetry_dataset(cell, symprec=symprec)
            break
            
# In[]
    data = [143, 89 + 33, 848 - 143 - 89 - 33] 
    type_R1_colors = {'perpendicular' :'red', 'in plan' : 'green', 'C1 or Cs with axe in plan : undefined' : 'blue'}
    plot_circular(data=data,
                  labels_colors=type_R1_colors,
                  title_graphe='Rashba material\'s dipole direction')
# In[]
    data = [143, 89 + 33, 848 - 143 - 89 - 33] 
    type_R1_colors = {'dans le plan' :'red', 'perpendiculaire' : 'green', 'Non défini : Cs avec l\'axe dans le plan ou C1' : 'blue'}
    plot_circular(data=data,
                  labels_colors=type_R1_colors,
                  title_graphe='Polarisation de spin des matériaux Rashba',
                  title_categorie='Polarisation de spin')