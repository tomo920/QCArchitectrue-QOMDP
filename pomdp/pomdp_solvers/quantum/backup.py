import numpy as np

from .calculation import apply_operator


def matrix_backup(s, eta_a_o, state_dim, gamma, action_set, observation_set, R):
    """
    Update value function V(|s>) = max{ <s|upsilon|s> | upsilon in eta },
    by updating set of upsilon matrices eta over some finite set of state points.

    Args
    ----------
    s: list
        Quantum state.
    eta_a_o: list
        Set of upsilon matrices updated by action and observation.
        eta_a_o[a][o] is list of updated upsilon matrices of executing action a and
        getting observation o.
    state_dim: int
        Dimension of quantum state.
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
    eta_s_max: dict
        Upsilon matrix maximizing <s|upsilon|s>.
        upsilon matrix is dictionary,
        key 'a' is action, key 'v' is upsilon matrix.
    """

    eta_s = []
    for a in action_set:
        # initialize next upsilon matrix
        eta_a = np.zeros((state_dim, state_dim)).astype(np.complex128)
        # calculate next upsilon matrix
        eta_a += R[a]
        for o in observation_set:
            # get upsilon matrix that maximizes expectation value
            i_ups_max = np.argmax([apply_operator(s, ups).real[0][0] for ups in eta_a_o[a][o]])
            eta_a += gamma * eta_a_o[a][o][i_ups_max]
        eta_s.append( {'a':a, 'v':eta_a} )
    i_upsilon_max = np.argmax([apply_operator(s, upsilon['v']).real[0][0] for upsilon in eta_s])

    eta_s_max = eta_s[i_upsilon_max]

    return eta_s_max
