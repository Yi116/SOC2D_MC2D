###
## Hypothesis :
The materaux 2D to analys have bondery condition in 3 dimension. But just the C vector is manualy set to a big value who forbidden most of interaction between layers.


###
module find_group is the basis module to find the symmetry groups in direct space. 
And the analyse_symmetry is there to find the apply and symmetry operations to illustrate some proprety physics of materials 2D

#### find_group

    change_matrix_basis_from_eucli_to_frac base on the funcition

#### analyse_symmetry
Use find_group.change_matrix_basis_from_eucli_to_frac

## Important
### Ase spglib convention. In ase objet Atoms, the fractionnales coordinates are in primitive cell given by Atoms.get_cell() , but since the transformation matrix are given by the spglib,
### So, the basis of the transformation matrix are the basis of the standard cell define in spglib. : https://spglib.readthedocs.io/en/stable/definition.html There is a transformation_matrix in spglib.symmetry_dataset who is define as
### std_cell = prim_std_trans @ prim_cell + origine shift, but we don't care the origine shift, cause all we care are symmetry operation. and usually the origine shift is a vector 0.
### by math definition, for a affin matrix in std_cell basis, to change it to the prim_cell basis, the formula is M_prim_basis = np.linagl.inv(prim_std_trans) @ M_std_basis @ prim_std_trans

## math :  old_basis_coord = T @ new_basis_coord
## math :  new_basis_matrix = T_inv @ old_basis_matrix @ T

## But in spglib symmetry_dataset, the rotation matrix and translation vector are given in primitive cell basis. So don't need to transformation of basis.







analyse_symmetry : 
## TODO : we know the for monoclinic lattice, we have propre C2 axis, but in standard lattice (contrast to primitive cell), the C axis is in plane. So write an if for monoclinic lattice to set the axis correctly.
    ### Done

## TODO : Abandance of SymmOP objet in code. 
        #### Direct space, find site group symmetry with scaled matrix of symmetry operations. (result : done)
        #### reciprocal space, find kpoint symmetry with scaled matrix of symmetry operations. 
        #### From scaled rotation matrix axis. Use it to find principal axis. #### Done!!!!

## TODO : find asymetry directions. ## MayBe not need to. Cause invariance methode (without group theory) can help us determin termes (like k coupled by pauli matrice) existe in hamiltonian.

## TODO Check for all spglib symmetry database for 2759 structure 2D.  
        All element found are validated, but didn't check if spglib missed to find some element.

## TODO :
    For the logic of finding the materials of having site polaire conected by the mirror or the glide plan, change the logic to the definition of pair of site d'atomes: Site link by the inversion. 



Scientific idea :

## for R-2 and D-2, how to find the electric field necessary to apply to obtain the SOC local useful?  We can in fact give some parameter to this question. A scientific research need to be done.


## For dresselhaus effet : study how to know the polarisation of spin by symmetry. ## Done :  invariance methode (without group theory)

## For 








