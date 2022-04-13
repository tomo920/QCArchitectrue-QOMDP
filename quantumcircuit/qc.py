import itertools
import numpy as np
import copy

s_0 = np.array([[1],[0]]).astype(np.complex128) #|0>
s_1 = np.array([[0],[1]]).astype(np.complex128) #|1>
s_base = [s_0, s_1]

M_0 = np.dot(s_0, s_0.conj().T) #|0><0|
M_1 = np.dot(s_1, s_1.conj().T) #|1><1|
M = [M_0, M_1]

R = np.dot(s_0, s_0.conj().T) + np.dot(s_0, s_1.conj().T) #|0><0|+|0><1|

def get_gate_matrix(gatename, theta=None):
    if gatename=='rx':
        gate = np.array([[np.cos(theta / 2), -1.0j * np.sin(theta / 2)],
                         [-1.0j * np.sin(theta / 2), np.cos(theta / 2)]]).astype(np.complex128)
    elif gatename=='ry':
        gate = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2), np.cos(theta / 2)]]).astype(np.complex128)
    elif gatename=='rz':
        gate = np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]])
    elif gatename=='cz':
        gate = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., -1.]]).reshape(4 * [2]).astype(np.complex128)
    elif gatename=='x':
        gate = np.array([[0, 1],[1, 0]]).astype(np.complex128)
    elif gatename=='y':
        gate = np.array([[0, -1.0j],[1.0j, 0]]).astype(np.complex128)
    elif gatename=='h':
        gate = 1 / np.sqrt(2) * np.array([[1, 1],[1, -1]]).astype(np.complex128)
    elif gatename=='z':
        gate = np.array([[1,0],[0, -1]]).astype(np.complex128)
    elif gatename=='i':
        gate = np.array([[1,0],[0, 1]]).astype(np.complex128)

    return gate

def get_tensor_product(gate_list):
    '''
    Calculate gate_list[0] tensor gate_list[1] ... tensor gate_list[-1]
    '''

    M = gate_list[0]
    for g in gate_list[1:]:
        M = np.kron(M, g)

    return M

def enum_control_bn(control_bn, target_bn):
    '''
    Enumerate all combinations of control bits and target bits of control gate.

    Args
    ----------
    control_bn: list
        List of index of control bit.
    target_bn: list
        List of index of target bit.

    Returns
    ----------
    List of combinations of control bit and target bit.
    '''

    if len(target_bn) == len(control_bn) == 1:
        return [[[control_bn[0], target_bn[0]]]]
    else:
        inds = []
        for t in target_bn:
            target_bn_copy = copy.copy(target_bn)
            target_bn_copy.remove(t)
            control_bn_copy = copy.copy(control_bn)
            inds_ =  enum_control_bn(control_bn_copy[1:], target_bn_copy)
            for ind in inds_:
                ind.extend([[control_bn[0], t]])
                inds.extend([ind])
        return inds

def enum_swap_bn(bn_list):
    '''
    Enumerate all combinations of bits of swap gate.

    Args
    ----------
    bn_list: list
        List of index of bit.

    Returns
    ----------
    List of combinations of two swapped bits.
    '''

    if len(bn_list) == 2:
        return [[bn_list]]
    else:
        inds = []
        for bn in bn_list[1:]:
            bn_list_copy = copy.copy(bn_list[1:])
            bn_list_copy.remove(bn)
            inds_ =  enum_swap_bn(bn_list_copy)
            for ind in inds_:
                ind.extend([[bn_list[0], bn]])
                inds.extend([ind])
        return inds

def enum_toffoli_bn(control_bn, target_bn):
    '''
    Enumerate all combinations of control bits and target bit of toffoli gate.

    Args
    ----------
    control_bn: list
        List of index of control bit.
    target_bn: list
        List of index of target bit.

    Returns
    ----------
    List of combinations of two control bits and target bit.
    '''

    if len(target_bn) == 1 and len(control_bn) == 2:
        return [[[control_bn[0], control_bn[1], target_bn[0]]]]
    else:
        inds = []
        for c_t in itertools.product(itertools.combinations(control_bn, 2), target_bn):
            target_bn_copy = copy.copy(target_bn)
            target_bn_copy.remove(c_t[1])
            control_bn_copy = copy.copy(control_bn)
            control_bn_copy.remove(c_t[0][0])
            control_bn_copy.remove(c_t[0][1])
            inds_ =  enum_toffoli_bn(control_bn_copy, target_bn_copy)
            for ind in inds_:
                ind.extend([[c_t[0][0], c_t[0][1], c_t[1]]])
                inds.extend([ind])
        return inds

def enum_cswap_bn(control_bn, target_bn):
    '''
    Enumerate all combinations of control bit and target bits of cswap gate.

    Args
    ----------
    control_bn: list
        List of index of control bit.
    target_bn: list
        List of index of target bit.

    Returns
    ----------
    List of combinations of control bit and two target bits.
    '''

    if len(target_bn) == 2 and len(control_bn) == 1:
        return [[[control_bn[0], target_bn[0], target_bn[1]]]]
    else:
        inds = []
        for t_c in itertools.product(itertools.combinations(target_bn, 2), control_bn):
            control_bn_copy = copy.copy(control_bn)
            control_bn_copy.remove(t_c[1])
            target_bn_copy = copy.copy(target_bn)
            target_bn_copy.remove(t_c[0][0])
            target_bn_copy.remove(t_c[0][1])
            inds_ =  enum_cswap_bn(control_bn_copy, target_bn_copy)
            for ind in inds_:
                ind.extend([[t_c[1], t_c[0][0], t_c[0][1]]])
                inds.extend([ind])
        return inds

def apply_gate(qc, gate_name, qubit, theta=None):
    '''
    Apply gate in quantum circuit
    '''

    if gate_name == 'i':
        qc.i(qubit[0])
    elif gate_name == 'z':
        qc.z(qubit[0])
    elif gate_name == 'x':
        qc.x(qubit[0])
    elif gate_name == 'h':
        qc.h(qubit[0])
    elif gate_name =='rx':
        qc.rx(theta, qubit[0])
    elif gate_name =='ry':
        qc.ry(theta, qubit[0])
    elif gate_name =='rz':
        qc.rz(theta, qubit[0])
    elif gate_name == 'cx':
        qc.cx(qubit[0], qubit[1])
    elif gate_name == 'cz':
        qc.cz(qubit[0], qubit[1])
    elif gate_name == 'swap':
        qc.swap(qubit[0], qubit[1])
    elif gate_name == 'toffoli':
        qc.toffoli(qubit[0], qubit[1], qubit[2])
    elif gate_name == 'cswap':
        qc.cswap(qubit[0], qubit[1], qubit[2])