import sys

import numpy as np

from .calculation import apply_operator, get_observation_probability, state_update
from .backup import matrix_backup


def point_based_value_update(state_point_set, eta, state_dim, A, gamma, action_set, observation_set, R):
    """
    Update value function V(|s>) = max{ <s|upsilon|s> | upsilon in eta },
    by updating set of upsilon matrices eta over some finite set of state points.

    Args
    ----------
    state_point_set: list
        Set of state points.
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

    # generate next step upsilon matrices
    eta_a_o = {}
    for a in action_set:
        eta_a_o.setdefault(a, {})
        for o in observation_set:
            eta_a_o[a][o] = [apply_operator(A[a][o], upsilon['v']) for upsilon in eta]

    eta_ = []

    # update upsilon matrices
    for s in state_point_set:
        eta_s_max = matrix_backup(s, eta_a_o, state_dim, gamma, action_set, observation_set, R)

        # not add upsilon matrix already added
        if not any([np.all(eta_s_max['v'] - ups_['v'] == 0) for ups_ in eta_]):
            eta_.append(eta_s_max)

    return eta_


def state_point_expand(S, action_set, observation_set, A, metric='fubini-study'):
    """
    Update state point set.

    Args
    ----------
    S: list
        Set of state points.
    action_set: list
        Set of actions.
        {0, 1, 2, 3, 4, ....}
    observation_set: list
        Set of observations.
        {0, 1, 2, 3, 4, ....}
    A: dictionary
        Set of transition operators.
        A[a][o] is transition operator matrix of executing action a and
        getting observation o.
    metric: string
        Metric used in calculation selecting new state point.

    Returns
    ----------
    S_: list
        Set of state points after update.
    """

    def calc_d(s, s_target, metric):
        if metric == 'l1norm':
            d = np.linalg.norm(s - s_target, ord=1)
        elif metric == 'l2norm':
            d = np.linalg.norm(s - s_target, ord=2)
        elif metric == 'fidelity':
            d = np.abs(np.dot(s.conj().T, s_target))**2
            d = d[0][0]
        elif metric == 'fubini-study':
            s_s_target = np.dot(s.conj().T, s_target)
            s_target_s = np.dot(s_target.conj().T, s)
            s_ip = np.dot(s.conj().T, s)
            s_target_ip = np.dot(s_target.conj().T, s_target)
            d = np.arccos(np.sqrt(s_s_target*s_target_s / (s_ip*s_target_ip)))
            d = d[0][0]
        elif metric == 'bures':
            f = np.abs(np.dot(s.conj().T, s_target))**2
            if f > 1:
                if f - 1 < 1e-10:
                    f = [[1]]
                else:
                    print('fidelity is over 1')
                    sys.exit()
            d = np.sqrt(2 - 2*np.sqrt(f))
            d = d[0][0]
        return d

    S_new = []

    for s in S:
        n_s = []
        d_s = []
        for a in action_set:
            # one step simulation
            o = np.random.choice(observation_set, p = get_observation_probability(s, a, A, observation_set))
            s_a = state_update(s, a, o, A)
            # check whether state has already been added
            if not any([np.all(np.abs(s_a - s_) < 1e-10) for s_ in S + S_new]):
                n_s.append(s_a)
                d_s.append([calc_d(s_a, s_, metric) for s_ in S])
        # check whether at least one point has been added
        if len(d_s) == 0:
            continue
        # select new state point which is farthest away from any state point in S
        if metric == 'fidelity':
            max_d = np.max(d_s, axis=1)
            a_min = np.argmin(max_d)
            if max_d[a_min] < 1:
                S_new.append(n_s[a_min])
        elif metric in ['l1norm', 'l2norm', 'fubini-study', 'bures']:
            min_d = np.min(d_s, axis=1)
            a_max = np.argmax(min_d)
            if min_d[a_max] > 0:
                S_new.append(n_s[a_max])

    S_ = S + S_new

    return S_
