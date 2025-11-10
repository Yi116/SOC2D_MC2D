import matplotlib.pyplot as plt
import numpy as np

from analyse_symmetry import *
from statistique import *

from typing import Callable

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def plot_histo(data : list[int], title : str, ylabel :str, xlabel : str):
    
    dic = {}
    ### cause the values of the keys cant be the same. so we merge the same values of the data.
    for d in data:
        dic[d] = None
    n_Z = len(dic.keys()) 

    ### plot
    fig, ax = plt.subplots()
    ax.hist(data, bins = n_Z)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.show()


def plot_circular(data : list, labels_colors : dict, title_graphe : str ,percentage = True, title_categorie = "Categories"):
    """
    plot a circular diagramme

    Para :
        data : list
        labels : list[str]
        percentage : True if you want a representation of percentage
    
    Return : None
    """
    labels = list(labels_colors.keys())
    colors = list(labels_colors.values())
    fig, ax = plt.subplots()
    if percentage :
        autopct = '%1.1f%%'
    else :
        autopct = None
        
    # ax.pie(x = data, labels=labels, autopct=autopct, labeldistance= 0.1, wedgeprops={'width': 0.3}, textprops=dict(color="b"), center=(3 ,3))
    legend_labels = [f'{label}: {d}' for label, d in zip(labels, data)]
    wedges, texts, autotexts = ax.pie(x = data, colors=colors ,autopct=autopct, pctdistance= 0.6 )
    
    ax.legend(wedges, legend_labels,  title=title_categorie, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title_graphe)
    plt.show()



def plot_circular_2levels(data : list, outer_labels : list[str] , inner_labels = None, percentag = True):
    """
    plot a circal diagramme with maximum 2 level of categorie. Respect to outer and inner labels.
    

    Parameter:
            data : list of maximum 2 dimension. 
            outer_labels :  main categorie.
            inner_labels : subset of main categorie. If some main categorie dont have subset. their place in inner_labels is 0.

    """


def plot_score_2D(data : pd.DataFrame, 
                  labelx : str, 
                  labely :str, 
                  title : str, 
                  labelx_to_print : str = None,
                  labely_to_print : str = None,
                  print_chemical_symbol : bool = False, 
                  print_criterion : Callable[[float, float], float] = None, 
                  print_contour_criterion : bool = True,
                  print_type_SOC_color : bool = False, ### if true, you must specifiy the labels_colors and type_SOC_file. And this true turn autocolor_selection to False
                  type_SOC_file : str = None, ### path to a json who storage pks with their SOC type. see the format of storage from the function save_dataframe_ASGDS_json()
                  labels_type_SOC_colors = {}, ### a dict with type of SOC and their color : e.g. {'R-2 & D-2' : 'red'}
                  critical_constant : float = -1.,
                  print_highest_score = False,
                  autocolor_selection = False
                  ):
    """
    from a dataframe, select 2 columns with labelx and labely, plot a 2D scatter plot. 

    Para:
        data : pd.DataFrame, index of line are identifier on the plot. It should be a identifier to a material in anyway (like pk or uuid of aiida database)
        labelx : str, name of column in dataframe to plot on axe x
        labely : str, name of column in dataframe to plot on axe y
        title : str, plot title
        print_chemical_symbol : bool, if True, and if print_criterion is not None 
                                            we suppose that the index are pk or uuid of aiida database. 
                                               Load the node of such pk or uuid who satifty the print_criterion (who return a bool). print chemical symbol who satify this criterion.
        print_criterion :  a function have 2 enter of float who return a float. cause we want to select some point who satify the print_criterion to plot the index of line or chemical symbol on the graphe. 
                            if it's not non, we do the calcul of selection scores by the function print_criterion.
        print_contour_criterion : bool: if you want to print the courbe where print_criterion == critical_constant.
        critical_constant :  a float to enter to the last entry of print_criterion
        
        print_highest_score : bool, if True, print the highest score on the graphe
        autocolor_selection : bool, if True, color the point randomly of the selected points on the graphe
        print_type_SOC_color : bool, if True, color point by their type of SOC. You must specifiy the labels_type_SOC_colors. And this true turn autocolor_selection to False
        labels_type_SOC_colors : dict, a dict with type of SOC and their color : e.g. {['R-2,D-2'] : 'red'} Note, the key of the dict correspond to output of find_Rashba_Dresshauls_effet_of_1_crystal
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ### select 2 serie from dataframe
    serie_x = data[labelx]
    serie_y = data[labely]
    index_satisty_cri = []
    plt.scatter(serie_x, serie_y, s = 20, color = 'grey') ### just initilize the scatter plot with grey
    
    #### print soc type colors
    if print_type_SOC_color: 
        autocolor_selection = False
        labels_type_SOC = list(labels_type_SOC_colors.keys())
        for i in range(len(labels_type_SOC)):
            labels_type_SOC[i] = str(labels_type_SOC[i])
        colors_type_SOC = list(labels_type_SOC_colors.values())
        df = get_dataframe_ASGDS_json(type_SOC_file)
        N_total_point = 0
        count_type_SOC = {}
        for idx in data.index:
            N_total_point += 1
            x = data.loc[idx, labelx]
            y = data.loc[idx, labely]
            # node_struc2D = load_node(idx)
            # ase_objet = node_struc2D.get_ase()
            # ASGDS_objet = Atoms_symmetry_group_direct_space().get_site_point_group(ase_objet)
            # type_SOC = find_Rashba_Dresshauls_effet_of_1_crystal(ASGDS_objet)
            type_SOC = df.loc[idx,"Rashba or Dressehauls type"]
            try :
                count_type_SOC[type_SOC] += 1
            except:
                count_type_SOC[type_SOC] = 1
            # if isinstance(type_SOC) != str:
            #     type_SOC = str(type_SOC)
            color_SOC = labels_type_SOC_colors[type_SOC]
            plt.scatter(x, y, color=color_SOC, s=30)
        
        ### calculate percentage of each type SOC, and add them to the count_type_soc
        for type_SOC in count_type_SOC:
            percent = round(count_type_SOC[type_SOC] / N_total_point * 100,2)
            count_type_SOC[type_SOC] = [count_type_SOC[type_SOC], percent]
        # # Extend x-axis to make space for labels of type color
        # x_min, x_max = plt.xlim()
        # plt.xlim(x_min, x_max + (x_max - x_min) * 0.2)  # Extend 20% to the right
        # new_x_max = plt.xlim()[1]

        # Create custom legend labels
        legend_labels = []
        for label in labels_type_SOC:
            try :
                count_string = str(count_type_SOC[label])
                count_string = count_string.replace('[', '')
                count_string = count_string.replace(']', '')
                count_string +=' %'
                string = f"{label} : {count_string}"
                legend_labels.append(string)
            except:
                string = f"{label} : 0"
                legend_labels.append(string)
        

        # Create legend patches
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=c, label=l) for c, l in zip(colors_type_SOC, legend_labels)]

        # Add the legend outside the plot
        ax.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5), fontsize =20)
        # plt.rcParams.update({'font.size': 22})
        # Adjust layout
        plt.tight_layout()

    # colors = plt.cm.get_cmap('tab10', 20)  # Generate colors dynamically
    colors = plt.cm.get_cmap('tab10',1000)
    selected_points = {}
    if print_criterion != None:
        color_idx = 0  # Keep track of color index
        scores_xy = {}
        for idx in data.index:
            x = data.loc[idx, labelx]
            y = data.loc[idx, labely]
            # print_criterion(x=8, y=9)
            score_xy, formula_str = print_criterion(x,y)
            can_print_xy = score_xy > critical_constant
            if can_print_xy:
                index_satisty_cri.append(idx)
                label_text = str(idx)
                if print_chemical_symbol: #### here print_chemical_symbol tell if we load nodes and search their chemical symbol, then print them
                    node = load_node(idx)
                    try :
                        chemical_symbol = node.get_ase().get_chemical_formula()
                        label_text = chemical_symbol
                    except:
                        chemical_symbol = f"{idx}:Unknown" ### if can't get the chemical symbol, print on the plot the index of the line and mark Unknown
                        label_text = chemical_symbol
                        print(f"can't load chemical symbol for {idx}")
                    
                        
                    #### annotation of point
                    # plt.annotate(f'{chemical_symbol}', 
                    #     (x, y),
                    #     xytext=(5, 5),
                    #     textcoords='offset points')
                # else :
                #     plt.annotate(f'{idx}', 
                #         (x, y),
                #         xytext=(5, 5),
                #         textcoords='offset points')
                    
                ###### Assign a unique color and store the label
                # point_color = colors(color_idx % 10)  # Cycle through colors
                if autocolor_selection :
                    point_color = colors(np.random.uniform(0, 1))
                    plt.scatter(x, y, color=point_color, edgecolors='black', s=30, label=label_text)
                    
                    selected_points[label_text] = point_color
                    scores_xy[label_text] = score_xy
                    color_idx += 1

        if print_contour_criterion:            
            #### print the line of selection formula
            x_max = data[labelx].max()
            x_min = data[labelx].min()
            y_max = data[labely].max()
            y_min = data[labely].min()
            
            num_points = 50
            x_range = np.linspace(x_min, x_max, num_points)
            y_range = np.linspace(y_min, y_max, num_points)
            X, Y = np.meshgrid(x_range, y_range)
            Z , _= print_criterion(X,Y)

            contour = plt.contour(X, Y, Z, levels=[critical_constant], colors='red', linewidths=2, linestyles="solid")
            # Add the equation as text near the contour
            plt.clabel(contour, inline=True, fontsize=10, fmt={critical_constant: f"{formula_str} = {critical_constant}"}, colors='red')
            # plt.pcolormesh(X, Y, Z, shading='auto', cmap='coolwarm')
        
        
        #### now try to print the chemical symbol of the highest score        
        if print_highest_score:
            scores_xy_sorted = dict(sorted(scores_xy.items(), key=lambda item: item[1]))
            highest_score_index = list(scores_xy_sorted.keys())[-1]
            try :
                node_highest_score = load_node(highest_score_index)
                chemical_symbol_highest = node_highest_score.get_ase().get_chemical_formula()
            except:
                chemical_symbol_highest = f"{idx}:Unknown"
            selected_points_sorted = {k: selected_points[k] for k in scores_xy_sorted.keys()}

            # Extend x-axis to make space for labels just for the highest score
            x_min, x_max = plt.xlim()
            plt.xlim(x_min, x_max + (x_max - x_min) * 0.2)  # Extend 20% to the right
            new_x_max = plt.xlim()[1]

            plt.text(new_x_max , y=0, s=f"highest score : {chemical_symbol_highest}", fontsize=20, verticalalignment='center')


        if print_chemical_symbol: ### here print_chemical_symbol tell us if you want to print chemical labels
            ### sort the points selected by their score. For printing them in order below
            scores_xy_sorted = dict(sorted(scores_xy.items(), key=lambda item: item[1]))
            selected_points_sorted = {k: selected_points[k] for k in scores_xy_sorted.keys()}

            # Extend x-axis to make space for labels
            x_min, x_max = plt.xlim()
            plt.xlim(x_min, x_max + (x_max - x_min) * 0.2)  # Extend 20% to the right
            
            # Add color-coded labels outside the plot
            y_max = max(serie_y) if len(serie_y) > 0 else 1
            label_y_pos = np.linspace(y_max, y_max * 0.3, len(selected_points))  # Space them vertically
            
            for (label, color), y_pos in zip(selected_points_sorted.items(), label_y_pos):
                plt.text(x_max + (x_max - x_min) * 0.05, y_pos, label, color=color, fontsize=10, verticalalignment='center')

    if labelx_to_print == None:
        labelx_to_print = labelx
    if labely_to_print == None:
        labely_to_print = labely
    ### add labels and title
    plt.xlabel(labelx_to_print, fontsize=22)
    plt.ylabel(labely_to_print, fontsize=22)
    plt.title(title, fontsize=22)
    plt.grid(True)
    plt.show()
    return index_satisty_cri

def plot_criterion_1(x : float, y : float) -> float:
    """
    return x*y
    """
    return x*y , "xy"

def plot_criterion_2(x : float, y : float) -> float:
    """
    return x/y
    """
    return x/y, "x/y"

def plot_criterion_3(x : float, y : float, r :float) :
    """
    return x - exp(y)/r
    """
    return x - np.exp(y/r), f"x - exp(y)/{r}"

def soc (jj : np.ndarray, ii : str ):
    return jj



def set_equal_scaling(ax):
    """ Ensure that all three axes have the same scale in a 3D plot. """
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def plot_brillouin_zone(a, b, c, view="oblique"):
    """
    Plots the Brillouin zone based on the reciprocal lattice vectors a, b, and c.
    
    Parameters:
        a, b, c: numpy arrays representing the reciprocal lattice vectors.
        view: str, specifies the viewing angle. Options are "oblique", "top", "right", "left".
    """
    # Step 1: Define reciprocal lattice points
    lattice_points = []
    grid_range = [-1, 0, 1]  # Expanding to ensure full hexagonal lattice is captured
    for i in grid_range:
        for j in grid_range:
            for k in grid_range:
                lattice_points.append(i * a + j * b + k * c)
    lattice_points = np.array(lattice_points)

    # Step 2: Compute the Voronoi diagram of the reciprocal lattice
    vor = Voronoi(lattice_points)

    # Step 3: Plot the Brillouin zone (Wigner-Seitz cell)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the edges of the Voronoi region surrounding the origin
    for ridge in vor.ridge_vertices:
        ridge = np.array(ridge)
        if np.all(ridge >= 0):  # Only plot valid ridges
            points = vor.vertices[ridge]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], "k-")

    # Step 4: Add lattice points to the plot
    ax.scatter(lattice_points[:, 0], lattice_points[:, 1], lattice_points[:, 2], color="red", label="Reciprocal Lattice Points")

    # Step 5: Set the view angle based on the specified parameter
    if view == "top":
        ax.view_init(elev=90, azim=0)
    elif view == "right":
        ax.view_init(elev=0, azim=0)
    elif view == "left":
        ax.view_init(elev=0, azim=180)
    else:  # Default to oblique view
        ax.view_init(elev=20, azim=30)

    # Step 6: Ensure equal scaling for all axes
    set_equal_scaling(ax)

    # Step 7: Set plot labels
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_zlabel("kz")
    ax.set_title(f"Brillouin Zone ({view.capitalize()} View)")

    plt.legend()
    plt.show()

def plot_kpoints_grid_contour(kpoints: np.ndarray, xlabel :str = 'ka', ylabel :str = 'kb'):
    """
    plot kpoint's projection on x and y. we suppose the 2 first component are kx and ky.

    
    """

    if kpoints.shape[1] == 3:
        kpoints = kpoints[:,:2]
    kpoints_xy = kpoints
    from scipy.spatial import ConvexHull
    contour_hull = ConvexHull(kpoints_xy)
    plt.plot(kpoints_xy[:,0], kpoints_xy[:,1], 'o', markersize=1)
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rcParams.update({'font.size': 20})
    for simplex in contour_hull.simplices:

        plt.plot(kpoints_xy[simplex, 0], kpoints_xy[simplex, 1], 'k-')


if __name__ == "__main__":

    soc(jj = 0)


    
