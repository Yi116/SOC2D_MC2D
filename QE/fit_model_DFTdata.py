# %%
import numpy as np
from sklearn.linear_model import LinearRegression

def C3v_1st_3rd_order_term(kxy :np.ndarray):
    """
    kxy : k point storage as [[kx, ky, 0]...] they will be transformed to sphere coordinate.
    
    """
    # Calculer k et phi
    kx = kxy[:, 0]
    ky = kxy[:, 1]
    k = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)  # Utiliser arctan2 pour gérer correctement les quadrants

    # Préparer les termes pour la régression
    termes_for_regression  = np.column_stack((
        k * np.cos(phi),           # alpha_R k cos(phi)
        -k * np.sin(phi),          # -alpha_R k sin(phi)
        k**3 * np.cos(phi),        # alpha_3R k^3 cos(phi)
        -k**3 * np.sin(phi),       # -alpha_3R k^3 sin(phi)
        k**3 * np.cos(3 * phi)     # alpha_3R' k^3 cos(3 * phi)
    ))

    return termes_for_regression


def spin_splitting_energy(energys_2_bands : np.ndarray):
    
    delta = (energys_2_bands[:,0] - energys_2_bands[:,1]) / 2
    return delta 


def mask_of_zones_effective_around_special_point(kxy, cutoff_radius, special_point):
    mask = np.linalg.norm(kxy - special_point, axis=1) <= cutoff_radius
    return mask


def regression(termes_for_regression : np.ndarray,
               energy_for_regression : np.ndarray,
               mask):
    
    termes_cutted = []
    for terme in termes_for_regression.T:
        terme_cutted = terme[mask]
        termes_cutted.append(terme_cutted)
    
    termes_cutted = np.array(termes_cutted).T
    energy_cutted = energy_for_regression[mask]
    model = LinearRegression.fit(termes_cutted, energy_cutted)

    coef = model.coef_

    return coef

def read_csv(path_csv):
    from numpy import genfromtxt
    my_data = genfromtxt(path_csv, delimiter=',')
    my_data_good = np.delete(my_data, 0, axis=0)
    return my_data_good

# %%
if __name__ == '__main__':

# %%
    path_csvs = '/home/jyin/workspace/test_garbage'
    kxy = read_csv(path_csv=path_csvs + '/' +'GeTe_kpoints_mate.csv')
    energys_2_bands = read_csv(path_csv=path_csvs + '/' +'GeTe_2CB.csv').T
    
# %%
    termes = C3v_1st_3rd_order_term(kxy=kxy)
# %%
    mask = mask_of_zones_effective_around_special_point(kxy=kxy, cutoff_radius=0.2 + 1e-6, special_point=np.array([0,0,0]))
# %%
    splitting_E = spin_splitting_energy(energys_2_bands=energys_2_bands)
# %%
    coefs = regression(termes_for_regression=termes,energy_for_regression=splitting_E, mask=mask)

# %%
