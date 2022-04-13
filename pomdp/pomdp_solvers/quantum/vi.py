import numpy as np
import itertools
from gurobipy import *


def enum_op(eta, state_dim, A, gamma, action_set, observation_set, R):
    """
    Enumerate |action_set| * |eta|^|observation_set| all next upsilon matrices.
    Each next upsilon matrix upsilon_ is calculated by
    upsilon_ = R[a] + gamma * ∑_o A[a][o]† upsilon A[a][o],
    a is from action_set,
    in each o, upsilon is from eta.

    Args
    ----------
    eta: list
        upsilon matrix set.
    state_dim: int
        Dimension of quantum state.
    A: dictionary
        Set of transition operators.
        A[a][o] is transition operator matrix of executing action a and
        getting observation o.
    gamma: float
        discount rate.
    action_set: list
        Set of actions.
        {0, 1, 2, 3, 4, ....}
    observation_set: list
        Set of observations.
        {0, 1, 2, 3, 4, ....}
    R: dictionary
        Set of reward operators.
        R[a] is reward operator matrix of executing action a.

    Returns
    ----------
    eta_: list
        Enumerated all new upsilon matrices.
    """

    eta_ = []

    observation_num = len(observation_set)

    # list of |eta|^|observation_set| combinations of choosing upsilon matrices in ∑_o,
    # in each element i_upsilon, i_upsilon[o] is the index in eta.
    i_upsilon_list = [ c for c in itertools.product(list(range(len(eta))), repeat=observation_num) ]

    # add all next upsilon matrices |action_set| * |eta|^|observation_set|
    for a in action_set:
        for i_upsilon in i_upsilon_list:
            # initialize next upsilon matrix
            eta_a = np.zeros((state_dim, state_dim)).astype(np.complex128)
            # calculate next upsilon matrix
            eta_a += R[a]
            for o in observation_set:
                eta_a += gamma * np.dot(np.dot(A[a][o].conj().T, eta[i_upsilon[o]]['v']), A[a][o])
            eta_.append( {'a':a, 'v':eta_a} )

    return eta_


def prune_op(eta, state_dim):
    """
    Prune upsilon matrix set eta.
    Remove upsilon matrices not chosen in calculating max{<s|upsilon|s> | upsilon in eta}.
    Solve the below quadratically constrained programing problem,
    max epsilon
    such that <s|(upsilon - upsolon_)|s> >= epsilon
    remove upsilon if epsilon < 0.

    Args
    ----------
    eta: list
        Upsilon matrix set.
    state_dim: int
        Dimension of quantum state.

    Returns
    ----------
    eta_: list
        Pruned upsilon matrix set.
    """

    eta_ = []

    def define_qcp_model():
        model = Model()
        model.params.NonConvex = 2
        model.params.LogToConsole = 0

        # define variable
        s_real = {}
        s_imag = {}
        for i in range(state_dim):
            s_real[i] = model.addVar(lb=-1., ub=1., vtype="C")
            s_imag[i] = model.addVar(lb=-1., ub=1., vtype="C")
        epsilon = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype="C")

        model.update()

        # add constraint |s> in S
        model.addConstr(quicksum(s_real[i]**2+s_imag[i]**2 for i in range(state_dim)) == 1.)

        return model, s_real, s_imag, epsilon

    def add_qconstr(model, s_real, s_imag, Ups, epsilon):
        '''
        add quadratic constraint <s|Ups|s> >= epsilon
        '''

        Ups_real = Ups.real
        Ups_imag = Ups.imag

        # calculate x.T A y
        calc_inner_product = (lambda x, A, y: quicksum(x[i]*A[i][j]*y[j]
                                                       for i in range(state_dim)
                                                       for j in range(state_dim)))

        # calculate real part of <s|Ups|s>
        model.addConstr(calc_inner_product(s_real, Ups_real, s_real)
                        + calc_inner_product(s_imag, Ups_imag, s_real)
                        + calc_inner_product(s_imag, Ups_real, s_imag)
                        - calc_inner_product(s_real, Ups_imag, s_imag) >= epsilon)

    removed_index = []

    for i, upsilon_i in enumerate(eta):
        model, s_real, s_imag, epsilon = define_qcp_model()
        for j, upsilon_j in enumerate(eta):
            if i != j:
                add_qconstr(model, s_real, s_imag, upsilon_i['v'] - upsilon_j['v'], epsilon)
        model.setObjective(epsilon, GRB.MAXIMIZE)
        # solve the quadratically constrained programing problem
        model.optimize()
        if model.ObjVal < 0.:
            removed_index.append(i)

    # remove the upsilon matrices
    for i, upsilon in enumerate(eta):
        if not i in removed_index:
            eta_.append(upsilon)

    return eta_


def update_value_function(eta, state_dim, A, gamma, action_set, observation_set, R):
    """
    Update value function V(|s>) = max{ <s|upsilon|s> | upsilon in eta },
    by updating set of upsilon matrices eta.

    Args
    ----------
    eta: list
        Set of upsilon matrices before update.
        upsilon matrix is dictionary,
        key 'a' is action, key 'v' is upsilon matrix.
    state_dim: int
        Dimension of quantum state.
    A: dictionary
        Set of transition operators.
        A[a][o] is transition operator matrix of executing action a and
        getting observation o.
    gamma: float
        discount rate.
    action_set: list
        Set of actions.
        {0, 1, 2, 3, 4, ....}
    observation_set: list
        Set of observations.
        {0, 1, 2, 3, 4, ....}
    R: dictionary
        Set of reward operators.
        R[a] is reward operator matrix of executing action a.

    Returns
    ----------
    eta_: list
        Set of upsilon matrices after update.
        upsilon matrix is dictionary,
        key 'a' is action, key 'v' is upsilon matrix.
    """

    # enumerate all upsilon matrices
    eta_ = enum_op(eta, state_dim, A, gamma, action_set, observation_set, R)

    # prune some upsilon matrices
    eta_ = prune_op(eta_, state_dim)

    return eta_
