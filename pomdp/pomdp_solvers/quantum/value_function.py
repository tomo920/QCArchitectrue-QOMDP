import sys

import numpy as np

from .calculation import apply_operator, get_observation_probability, state_update


def get_best_action(s, eta):
    """
    Get best action corresponding to upsilon chosen in V(|s>) = max{<s|upsilon|s> | upsilon in eta}.

    Args
    ----------
    s: list
        Quantum state.
    eta: list
        Set of upsilon matrices.
        upsilon matrix is dictionary,
        key 'a' is action, key 'v' is upsilon matrix.

    Returns
    ----------
    a: int
        Best action.
        upsilon maximizing <s|upsilon|s> correspond to a.
    """

    max_v = -np.inf
    a_best = -1

    for upsilon in eta:
        v = np.dot(np.dot(s.conj().T, upsilon['v']), s).real[0][0]
        if v > max_v:
            a_best = upsilon['a']
            max_v = v

    if a_best == -1:
        print('unexpected action is chosen')
        sys.exit()

    return a_best


def calc_value_function(s, eta):
    """
    Calculate V(|s>) = max{<s|upsilon|s> | upsilon in eta}.

    Args
    ----------
    s: list
        Quantum state.
    eta: list
        Set of upsilon matrices.
        upsilon matrix is dictionary,
        key 'a' is action, key 'v' is upsilon matrix.

    Returns
    ----------
    max_v: float
        Optimal Value function in quantum state s.
    max_upsilon: dict
        Upsilon matrix maximizing value function.
        upsilon matrix is dictionary,
        key 'a' is action, key 'v' is upsilon matrix.
    """

    max_v = -np.inf
    max_upsilon = None

    for upsilon in eta:
        v = np.dot(np.dot(s.conj().T, upsilon['v']), s).real[0][0]
        if v > max_v:
            max_v = v
            max_upsilon = upsilon

    return max_v, max_upsilon


def apply_bellman_operator(s, V, A, gamma, action_set, observation_set, R):
    """
    Apply bellman operator H to V,
    calculate HV(s) = max{ <s|R_a|s> + gamma * sigma_o p(o|s, a)V(s_) }.

    Args
    ----------
    s: list
        Quantum state.
    V: func
        Method calculating value fuction in s.
        V(s) -> float
        where s is quantum state.
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
    HV: float
        Optimal value function gotten by applying bellman operator.
    """

    q_s = []

    for a in action_set:
        p_o_s_a = get_observation_probability(s, a, A, observation_set)
        q_s_a = apply_operator(s, R[a]).real[0][0]
        for o in observation_set:
            q_s_a += gamma * p_o_s_a[o] * V(state_update(s, a, o, A))
        q_s.append(q_s_a)

    HV = np.max(q_s)

    return HV

