# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import itertools

from sympy.physics.quantum.dagger import Dagger ### for matrix vector congugate calculation

# def C3v_1st_3rd_order_term(kxy :np.ndarray):
#     """
#     kxy : k point storage as [[kx, ky, 0]...] they will be transformed to sphere coordinate.
    
#     TODO : to write the correct term with the factors who is the eigenvalue of pauli matrix at each term
#     """
#     # Calculer k et phi
#     kx = kxy[:, 0]
#     ky = kxy[:, 1]
#     k = np.sqrt(kx**2 + ky**2)
#     phi = np.arctan2(ky, kx)  # Utiliser arctan2 pour gérer correctement les quadrants

#     # Préparer les termes pour la régression
#     termes_for_regression  = np.column_stack((
#         k * np.cos(phi) - k * np.sin(phi),           # alpha_R (k cos(phi) - k sin(phi))
#         k**3 * np.cos(phi) - k**3 * np.sin(phi),        # alpha_3R k^3 cos(phi)-alpha_3R k^3 sin(phi)
#         k**3 * np.cos(3 * phi)     # alpha_3R' k^3 cos(3 * phi)
#     ))

#     return termes_for_regression

import sympy as sp


#######################################
### 1st part is the kp method's hamiltonian.
#######################################



# Symbolic variable of sympy to defin. 
kx, ky = sp.symbols('kx ky', real = True)
k, theta = sp.symbols('k, theta', real = True)
##here the first number label the order of k, the second label the termes of this order. But usually, we stop at 3rd order
alpha1, alpha3, alpha3_2 = sp.symbols('alpha1 alpha3 alpha3_2', real = True) ## C3v

# Définir les matrices de Pauli
sigma_x = sp.Matrix([[0, 1], [1, 0]])
sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sigma_z = sp.Matrix([[1, 0], [0, -1]])

spin_x, spin_y, spin_z = sp.symbols('spin_x spin_y spin_z', real = True)

# Defint Hamiltonians for each point group
H_c3v = alpha1 * (kx * sigma_y - ky * sigma_x) + alpha3 * ((kx**3 + ky**2 * kx) * sigma_y - (ky**3 + ky * kx**2) * sigma_x) + alpha3_2 * (kx**3 - 3*kx*ky**2) * sigma_z
H_c3v_scalar_spin = alpha1 * (kx * spin_y - ky * spin_x) + alpha3 * ((kx**3 + ky**2 * kx) * spin_y - (ky**3 + ky * kx**2) * spin_x) + alpha3_2 * (kx**3 - 3*kx*ky**2) * spin_z
## For each hamitonian need to write a spin scalar version, cause sympy .subs() method can change nature of an expression (from matrix to polynomial)



# dic from point group to effective kp hamiltonian
kp_Hsoc_point_group = {'C3v' : (H_c3v, H_c3v_scalar_spin)}


#########
######### Method with entering the DFT spin vectors to the model.

def termes_for_regression_with_DFT_spin_value(spin_texture : np.ndarray, kxy : np.ndarray ,H : sp.MutableDenseMatrix):

    """
    Arg :
    --------
        spin_texture : spin from DFT with shape (2, N_kpoints, 3), the 2 stade for 2 bands
        kxy : kpoints with shape (N_kpoints, 3) 
    """
    if spin_texture.shape[0] != 2 or  spin_texture.shape[2] != 3: ## [0] is the number of band, [2] is the number of spin component
        raise ValueError("please enter a matrix where axis 0 represent 2 bands with spin opposite and axis 2 is the 3 spin component")
    if kxy.shape[0] != spin_texture.shape[1] or (kxy.shape[1] != 2 and kxy.shape[1] != 3):
        raise ValueError("please enter a k mesh with axis 0 equal to spin texture's axis 1. And each k point should be express as a 2 or 3 component vector")

    if isinstance(H, (sp.Matrix, sp.ImmutableMatrix, sp.MutableDenseMatrix)) :
        raise TypeError("H should not be a tensor or matrix")
    # H_no_pauli_matrix = H
    # H_no_pauli_matrix = H_no_pauli_matrix.subs(sigma_x ,spin_x)
    # H_no_pauli_matrix = H_no_pauli_matrix.subs(sigma_y , spin_y)
    # H_no_pauli_matrix = H_no_pauli_matrix.subs(sigma_z , spin_z)
    terms_to_insert_values, _ = terms_for_regression(H)
    
    # for index, k in enumerate(kxy): ## you enter the 
    #     k_x, k_y, k_z = k
    #     spin_band0 = spin_texture[0][index]
    #     spin_band1 = spin_texture[1][index]

    #     Ek_band0 = H.subs({kx : k_x, ky : k_y, sigma_x : spin_band0[0], sigma_y : spin_band0[1], sigma_z : spin_band0[2]})
    #     Ek_band1 = H.subs({kx : k_x, ky : k_y, sigma_x : spin_band1[0], sigma_y : spin_band1[1], sigma_z : spin_band1[2]})
    #     diff_energy = Ek_band0 - Ek_band1
    #     all_diff_energy.append(diff_energy)
    kx_values = kxy[:,0]
    ky_values = kxy[:,1]
    spinx_values_band0 = spin_texture[0, : , 0]
    spiny_values_band0 = spin_texture[0, : , 1]
    spinz_values_band0 = spin_texture[0, : , 1]

    spinx_values_band1 = spin_texture[1, : , 0]
    spiny_values_band1 = spin_texture[1, : , 1]
    spinz_values_band1 = spin_texture[1, : , 1]

    terms_diff_energy_band0_minus_band1 = []
    for key in list(terms_to_insert_values.keys()):
        term = terms_to_insert_values[key]
        term_in_numpy_function = sp.lambdify(args=[kx, ky, spin_x, spin_y, spin_z], expr = term, modules='numpy')
        term_with_value_band0 = term_in_numpy_function(kx_values, ky_values, spinx_values_band0, spiny_values_band0, spinz_values_band0)
        term_with_value_band1 = term_in_numpy_function(kx_values, ky_values, spinx_values_band1, spiny_values_band1, spinz_values_band1)
        term_diff_band0_minus_band1 = term_with_value_band0 - term_with_value_band1
        terms_diff_energy_band0_minus_band1.append(term_diff_band0_minus_band1)
    
    terms_diff_energy_band0_minus_band1 = np.array(terms_diff_energy_band0_minus_band1)
    return terms_diff_energy_band0_minus_band1







#########
######### Method with diagonalization of matrix

def diagonalize_matrix(H : sp.MutableDenseMatrix ):
    # Diagonalize Hamiltonian (or something else)
    # eigenvalues = H.eigenvals()
    eigenvectors_values = H.eigenvects()
    eigenvectors = []
    eigenvalues_multi = {}
    for ei in eigenvectors_values:
        eigenvectors.append(ei[2][0])
        eigenvalues_multi[ei[0]] = ei[1]
    
    return eigenvalues_multi, eigenvectors


def change_kxky_to_k_theta(formula):
    """
    Deprecated!!! No very useful, but keep it.
    """
    # Define the transformation equations
    subs = {kx: k * sp.cos(theta), ky: k * sp.sin(theta)}
    # Substitute and simplify
    expr_polar = formula.subs(subs)
    expr_polar_simplified = sp.simplify(expr_polar)

    return expr_polar_simplified



# def substitute_k_values(eigenvalues, k_points):
#     """
#     Substitue the k values in difference of eigenvalues by assuming it's a 2 dim matrix's result
#     """
#     diff_eigenvalue = eigenvalues[0] - eigenvalues[1]

#     substituted_diff_eigenvalues = []
    
#     for k in k_points:
#         kx_val, ky_val = k
#         # Substitut the k values in eigenvalues different
#         diff_eigenvalue_values  = diff_eigenvalue.subs({kx: kx_val, ky: ky_val})
        
#         substituted_diff_eigenvalues.append(diff_eigenvalue_values)
    
#     return substituted_diff_eigenvalues

def find_alpha_monomials(expr, alpha_prefix='alpha', max_order=2):
    """
    Find all symbolic parameters in expr whose names start with alpha_prefix,
    and generate all unique monomials up to max_order in those parameters,
    including powers and products.

    Parameters:
    - expr: sympy expression containing alpha parameters
    - alpha_prefix: string prefix for alpha parameters (default 'alpha')
    - max_order: max order of monomials to generate (default 2)

    Returns:
    - list of sympy monomials (symbols and their products) up to max_order
    """

    # Find all alpha parameter symbols
    # alpha_syms = sorted([s for s in expr.free_symbols if s.name.startswith(alpha_prefix)], key=lambda s: s.name)
    alpha_syms = [s for s in expr.free_symbols if s.name.startswith(alpha_prefix)]

    if max_order == 1:
        return alpha_syms

    monomials = alpha_syms[:]  # start with first order monomials

    # Generate higher order monomials up to max_order
    for order in range(2, max_order + 1):
        # Use combinations_with_replacement to generate powers and products
        combos = itertools.combinations_with_replacement(alpha_syms, order)
        for c in combos:
            monomial = sp.Mul(*c)
            monomials.append(monomial)
    
    return monomials, alpha_syms


def terms_for_regression(expr : sp.Mul):
    """
    Find termes to regression in the expr who is a sympy polynomial

    Return :
    --------
        terms: {alpha dependent para  : k dependent terms}, 
        alphas : alpha symbols

    """
    syms, alphas = find_alpha_monomials(expr)
    terms = sp.collect(expr=expr, syms=syms, evaluate=False) ## evaluate = False to output a dict like {alpha dependent para : k dependent terms}

    return terms, alphas

def add_values_to_terms_for_regression(kpoints : np.ndarray, terms : dict):

    """
    add values of k point into the outputs terms of terms_for_regression
    """

    if kpoints.shape[1] != 2 and kpoints.shape[1] != 3:
        raise ValueError("kpoints array should have a form of (n_sample, 2 or 3)")
    kx_values = kpoints[:,0]
    ky_values = kpoints[:,1]

    terms_in_values = []
    for key in list(terms.keys()):
        term = terms[key]
        term_in_numpy_function = sp.lambdify(args=[kx, ky], expr = term, modules='numpy')
        term_as_value_array = term_in_numpy_function(kx_values, ky_values)
        terms_in_values.append(term_as_value_array)
    
    terms_in_values = np.array(terms_in_values)
    return terms_in_values









def generate_lambdify_function(expr, variables=(sp.symbols('kx ky'))):
    """
    for the eigenvalues expression, take the lambdify expression to use in curve_fit of sklearn (with k points values)
    
    Deprecated!!!
    """
    # Find all free symbols in the expression
    all_symbols = expr.free_symbols
    
    # Define variables set (kx, ky)
    variables_set = set(variables)
    
    # Parameters are all symbols except variables
    params = sorted(all_symbols - variables_set, key=lambda s: s.name)
    
    # Create the argument tuple: ((kx, ky), alpha1, alpha3, ...)
    args = (variables,) + tuple(params)
    
    # Generate the lambdified function
    model_func = sp.lambdify(args, expr, modules='numpy')
    
    return model_func, params


def create_model_to_fit(expr):
    """
    Here, to_fit means to use in curve_fit of sklearn. 

    Deprecated!!!
    """
    model_func, parameters = generate_lambdify_function(expr)
    def model_to_fit(coords, *params_vals):
        return model_func(coords[0], coords[1], *params_vals)
    return model_to_fit, parameters



def fit_DFT(k_points : np.ndarray, 
            y_data : np.ndarray, 
            expr):
    """
    To fit the curve of bands (or differences like spin splitting) using curve_fit of sklearn. parameters are parameters to fit in your hamiltonians. 
    Parameters should normally be given in the function generate_lambdify_function

    Deprecated!!! Usually it take to long time to fit a curve (I don't know how long because the 1st test i took 7min with 9 kpoints and i dont want to continu)
    """
    if k_points.shape[0] != y_data.shape[0]:
        raise ValueError("the data of k and energy should have the same number of samples : length of axis 0")
    kx_data = k_points[:, 0]
    ky_data = k_points[:, 1]
    # Initial guess for parameters, ordered as in `parameters`
    model_to_fit, parameters = create_model_to_fit(expr)
    initial_guess = [1.0] * len(parameters)
    popt, pcov = curve_fit(model_to_fit, (kx_data, ky_data), y_data, p0=initial_guess)
    return popt, pcov

# def Rashba_1st_order_term(kxy : np.ndarray):

#     # Calculer k et phi
#     kx = kxy[:, 0]
#     ky = kxy[:, 1]
#     k = np.sqrt(kx**2 + ky**2)
#     phi = np.arctan2(ky, kx) 

#      # Préparer les termes pour la régression
#     termes_for_regression  = k * np.cos(phi) * np.sin(phi) - k * np.sin(phi) * np.cos(phi), 
#     termes_for_regression = np.array(termes_for_regression).T
    
#     return termes_for_regression

################################
### end of kp method's part
################################











def spin_splitting_energy(energys_2_bands : np.ndarray):
    """
    energy of 2 bands should have shape (N kpoint, 2)
    
    """
    delta = (energys_2_bands[:,0] - energys_2_bands[:,1]) / 2
    return delta 


def mask_of_zones_effective_around_special_point(kxy, cutoff_radius, special_point):
    mask = np.linalg.norm(kxy - special_point, axis=1) <= cutoff_radius
    return mask


def regression(termes_for_regression : np.ndarray,
               y_for_regression : np.ndarray,
               mask):
    
    """
    Here a linear regression for the termes who are some list of value in ndarray (normally k dependent) and y for regression .

    In SOC usage, y_for_regression is usally the energy ** 2. 
    """

    # termes_cutted = []
    # for terme in termes_for_regression:
    #     terme_T = terme.T
    #     terme_cutted = terme_T[mask]
    #     termes_cutted.append(terme_cutted)
    termes_cutted = termes_for_regression.T[mask]
    # termes_cutted = np.array(termes_cutted)
    # terme_cutted = terme_cutted.tolist()
    energy_cutted = y_for_regression[mask]
    mm = LinearRegression()
    model = mm.fit(X=termes_cutted, y=energy_cutted)

    # coef = model.coef_

    return model

def read_csv(path_csv):
    from numpy import genfromtxt
    my_data = genfromtxt(path_csv, delimiter=',')
    my_data_good = np.delete(my_data, 0, axis=0)
    return my_data_good





def process_diag_until_linearmodel(H : sp.MutableDenseMatrix,
                  energys_2_bands : np.ndarray,
                  kxy : np.ndarray,
                  radius_to_fit = 0.2,
                  special_point = np.array([0,0,0])):
    
    if kxy.shape[0] != energys_2_bands.shape[0] :
        raise ValueError('The 0 axis of 2 samples array must be the number of sample.')

    eigvals_mul, eigvecs = diagonalize_matrix(H)
    eigvals = list(eigvals_mul.keys())
    diff_eigvals = eigvals[0] - eigvals[1]
    diff_eigvals_square = diff_eigvals ** 2


    mask = mask_of_zones_effective_around_special_point(kxy=kxy, cutoff_radius=radius_to_fit + 1e-6, special_point=special_point)
    splitting_E = spin_splitting_energy(energys_2_bands=energys_2_bands)
    splitting_E_square = splitting_E**2
    terms, alphas = terms_for_regression(diff_eigvals_square)
    terms_as_values_array_for_regression = add_values_to_terms_for_regression(kxy, terms=terms)

    model_line = regression(termes_for_regression=terms_as_values_array_for_regression,
               y_for_regression=splitting_E_square,
               mask=mask)
    socre_R_square = model_line.score(terms_as_values_array_for_regression.T[mask], splitting_E_square[mask])
    
    return model_line, terms , eigvals_mul, eigvecs, alphas, socre_R_square





def process_using_DFT_spin_until_linearmodel(H : sp.MutableDenseMatrix,
                                             energys_2_bands : np.ndarray,
                                             kxy : np.ndarray,
                                             spin_texture : np.ndarray,
                                             raidus_to_fit = 0.2,
                                             special_point = np.array([0,0,0])):
    if kxy.shape[0] != energys_2_bands.shape[0] :
        raise ValueError('The 0 axis of 2 samples array must be the number of sample.')

    mask = mask_of_zones_effective_around_special_point(kxy=kxy, cutoff_radius=raidus_to_fit + 1e-6, special_point=special_point)
    splitting_E = spin_splitting_energy(energys_2_bands=energys_2_bands)
    
    terms_as_values_with_DFT_spins = termes_for_regression_with_DFT_spin_value(spin_texture=spin_texture, kxy=kxy, H=H)
    
    model_line = regression(termes_for_regression=terms_as_values_with_DFT_spins,
               y_for_regression=splitting_E,
               mask=mask)
    socre_R_square = model_line.score(terms_as_values_with_DFT_spins.T[mask], splitting_E[mask])
    return model_line, socre_R_square



def solve_alphas(alphas, terms : dict, models_line : LinearRegression):
    """
    simply solve the all alphas' value with the coef_ in models_line

    The linear regression can not give the value of alpha perfectly mach like alpha1  ** 2 = l, alpha3  ** 2 = m and alpha1 * alpha3 = sqrt(l*m) 
    So single variable terms for calculation, multiple variable terms for checking with a relative error below 10 % of reference values given by Linear regression  

    Arg :
    --------  
        corresponding output of process_until_linearmodel

    Return :
    --------
        Checked_sols : list of length equal to number of solutions. Each solution is a dict :{name of para : value}
    """
    ### ith coef in models_line.coef_ is the ith values of terms.keys() 
    coefs = models_line.coef_
    eqs = []
    checking_terms = {}
    for index, term in enumerate(list(terms.keys())):
        if len(term.free_symbols) == 1:
            eq = sp.Eq(term, coefs[index])
            eqs.append(eq)
        else :
            checking_terms[term] = coefs[index]
    
    solutions = sp.solve(eqs, alphas, dict=True)

    Checked_sols = []
    for sol in solutions:
        Good_sol = True
        for key in list(checking_terms.keys()):
            value_reference = checking_terms[key] ## here the key is the term for checking
            alphas_correspond = key.free_symbols
        
            vals_alphas_correspond = {key: sol[key] for key in alphas_correspond}
            
            value_from_solution = key.subs(vals_alphas_correspond)
            abso_diff = abs(value_from_solution - value_reference)
            if abso_diff > abs(value_reference) * 0.1:
                Good_sol = False
        if Good_sol:
            Checked_sols.append(sol)

    return Checked_sols




def convert_factor_k_alat_on_bohr_angs(latt_len_bohr):

    # lattice vector length on angstrom = latt_len_bohr / 0.529177211
    factor_k_QE_to_ang = (latt_len_bohr/0.529177211)/(2 * np.pi)

    return factor_k_QE_to_ang




# %%
if __name__ == '__main__':

# %%
    path_csvs = '/home/jyin/workspace/test_garbage'
    # path_csvs = '/home/junlin/workspace/data/generated_data'
    kxy = read_csv(path_csv=path_csvs + '/' +'GeTe_kpoints_mate.csv')
    energys_2_bands = read_csv(path_csv=path_csvs + '/' +'GeTe_2CB.csv').T

# %% lattice vector visually found
    latt_len_bohr = 7.3574
    latt_len_angs = latt_len_bohr / 0.529177211
    factor_k_QE_to_ang = (latt_len_bohr/0.529177211)/(2 * np.pi)
    
# %%
    # termes = C3v_1st_3rd_order_term(kxy=kxy)
    # termes = Rashba_1st_order_term(kxy=kxy)
# %%
#     mask = mask_of_zones_effective_around_special_point(kxy=kxy, cutoff_radius=0.2 + 1e-6, special_point=np.array([0,0,0]))
# # %%
#     splitting_E = spin_splitting_energy(energys_2_bands=energys_2_bands)
# # %%
#     # model = regression(termes_for_regression=termes,energy_for_regression=splitting_E, mask=mask)

# # %%
#     eigvals_mul, eigvecs = diagonalize_matrix(H_c3v)
#     eigvals = list(eigvals_mul.keys())
#     diff_eigvals = eigvals[0] - eigvals[1]
#     diff_eigvals_square = diff_eigvals ** 2

#     # model_func, parameters = generate_lambdify_function(diff_eigvals_square)
#     splitting_E_square = splitting_E**2
#     # popt, pcov =  fit_DFT(k_points=kxy[mask], y_data= splitting_E_square[mask], expr=diff_eigvals_square)
# # %%
#     terms = terms_for_regression(diff_eigvals_square)
#     terms_as_values_array_for_regression = add_values_to_terms_for_regression(kxy, terms=terms)

# # %%
#     model_line = regression(termes_for_regression=terms_as_values_array_for_regression,
#                y_for_regression=splitting_E_square,
#                mask=mask)

# %% 
    model, terms, eigvals, eigvecs, alphas = process_diag_until_linearmodel(H=H_c3v,
                                                              energys_2_bands=energys_2_bands,
                                                              kxy=kxy)
# %%
    sols = solve_alphas(alphas=alphas,
                        terms=terms,
                        models_line=model)