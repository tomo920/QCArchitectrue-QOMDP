import numpy as np


Circuit_config = {
    # gate type list applied in each qubit
    # 'gate_type_list': ['single', 'target', 'cz', 'swap', 'toffoli', 'cswap', 'cont_targ', 'cont_toffoli', 'cont_swap'],
    # 'gate_type_list': ['single', 'target', 'cont_targ'],
    'gate_type_list': ['i', 'single', 'target', 'cont_targ'],

    # single rotation gate list
    'single_rotation_gate': ['rx', 'ry', 'rz'],
    # rotation angle
    'rotation_angle': np.pi/9.,

    # single nonparametric gate
    # 'single_gate_list': ['i', 'h'],
    'single_gate_list': ['h'],

    # control gate
    'target_gate_list': ['cx'],
}


def check_unitary_constraints(u):
    '''
    Check whether the unitary is legal.

    Args
    ----------
    u: list
        Each element is the gate type applied in the qubit.

    Returns
    ----------
    bool
        True if the unitary is legal, false else.
    '''

    n_single = u.count('single')
    n_targ = u.count('target')
    n_cz = u.count('cz')
    n_swap = u.count('swap')
    n_toffoli = u.count('toffoli')
    n_cswap = u.count('cswap')
    n_c_targ = u.count('cont_targ')
    n_c_toffoli = u.count('cont_toffoli')
    n_c_swap = u.count('cont_swap')

    # check constraints
    if n_swap % 2 == 1 or n_cswap % 2 == 1 or n_cz % 2 == 1:
        return False
    if n_c_targ != n_targ or n_c_toffoli != n_toffoli * 2 or n_c_swap != int(n_cswap/2):
        return False

    # additional constraints
    if n_single != 0 and n_targ != 0:
        return False
    if (n_single != 1 and n_targ == 0) or (n_targ != 1 and n_single == 0):
        return False

    return True