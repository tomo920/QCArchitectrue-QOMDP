import numpy as np

from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermionpyscf import run_pyscf

from quantumcircuit.qc import get_gate_matrix, get_tensor_product

def get_hamiltonian(molecule, bond_length):
    '''
    Get hamiltonian of molecule.

    Args
    ----------
    molecule: str
        Molecule to calculate hamiltonian.
    bond_length: float
        Distance between atoms.

    Returns
    ----------
    hamiltonian: list
        Hamiltonian.
        Shape is (2**N, 2**N), where N is number of qubits.
    '''

    # prepare gates
    I = get_gate_matrix('i')
    X = get_gate_matrix('x')
    Y = get_gate_matrix('y')
    Z = get_gate_matrix('z')
    pauli = {'X': X, 'Y': Y, 'Z': Z}

    if molecule == 'h2':
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        geometry = [("H",(0, 0, 0)), ("H", (0, 0, bond_length))]
        description = ""
    elif molecule == 'n2':
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        geometry = [("N",(0, 0, 0)), ("N", (0, 0, bond_length))]
        description = ""
    elif molecule == 'li2':
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        geometry = [("Li",(0, 0, 0)), ("Li", (0, 0, bond_length))]
        description = ""
    elif molecule == 'hhe':
        basis = "sto-3g"
        multiplicity = 1
        charge = 1
        geometry = [("H",(0, 0, 0)), ("He", (0, 0, bond_length))]
        description = ""
    elif molecule == 'lih':
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        geometry = [("Li",(0, 0, 0)), ("H", (0, 0, bond_length))]
        description = ""
    elif molecule == 'h2o':
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        geometry = [("H",(0, 0, -0.1)), ("H", (0, 0, 0.1)), ("O", (bond_length, 0, 0))]
        description = ""
    elif molecule == 'beh2':
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        geometry = [("H",(0, 0, -bond_length)), ("H", (0, 0, bond_length)), ("Be", (0, 0, 0))]
        description = ""

    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)

    # qubit
    N = molecule.n_qubits

    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))

    hamiltonian = np.zeros((2**N, 2**N)).astype(np.complex128)

    for term in jw_hamiltonian.terms.items():
        gate_term, const_term = term
        # get gate list to calculate tensor product
        gate_list = [I] * N
        for i in range(N):
            if i < len(gate_term): gate_list[gate_term[i][0]] = pauli[gate_term[i][1]]
        # update hamiltonian
        hamiltonian += const_term * get_tensor_product(gate_list[::-1])

    return hamiltonian
