import sys

import numpy as np
import itertools
from scipy.optimize import linprog


def belief_update(b, a, o, p_t, p_o, S):
    """
    Perform belief state update when belief state is b,
    action the agent executed is a and observation the agent observed is o.

    Args
    ----------
    b: list
        Belief state before update.
        b[s] is the probabilty the state is s.
    a: int
        Action agent executed.
    o: int
        Observation agent observed.
    p_t: list
        Transition probability.
        p_t[s'][s][a] is the probabilty of transitioning in state s',
        given startinig state s and executing action a.
    p_o: list
        Observation probability.
        p_o[o][a][s'] is the probabilty of getting observation o,
        given executing action a and transitioned state s'.
    S: list
        Set of states.
        {0, 1, 2, 3, 4, ....}

    Returns
    ----------
    b_: list
        Belief state after update.
        b_[s'] is the probabilty the state is s'.
    """

    b_ = np.zeros(len(S))

    for s_ in S:
        b_[s_] = p_o[o][a][s_] * np.sum( [p_t[s_][s][a] * b[s] for s in S] )

    # normalize
    b_ = b_ / np.sum(b_)

    return b_


def enum_op(nu, S, A, state_num, observation_num, r_sa, alpha_, gamma):
    """
    Enumerate |A| * |nu|^|O| all next alpha vector sets.
    Each next alpha vector nu_a is calculated by
    nu_a[s] = r_sa[s][a] + gamma * sigma_o( alpha_[i_alpha][a][o][s] ),
    action a is from A,
    in each a, there are |nu|^|O| combinations of choosing alpha vector
    in each o when calculating sigma_o.

    Args
    ----------
    nu: list
        Alpha vector set.
    S: list
        Set of states.
        {0, 1, 2, 3, 4, ....}
    A: list
        Set of actions.
        {0, 1, 2, 3, 4, ....}
    state_num: int
        Number of state.
    observation_num: int
        Number of observation.
    r_sa: list
        Reward function.
        r_sa[s][a] is the reward,
        given startinig state s and executing action a.
    alpha_: list
        Next alpha vector.
        alpha_[i_alpha][a][o] is alpha vector when executing action a
        and getting observation o. i_alpha is the index of alpha vector in nu.
    gamma: float
        discount rate.

    Returns
    ----------
    nu_: list
        Enumerated all new alpha vector sets.
    """

    nu_ = []

    # |nu|^|O| combinations of choosing alpha vector in sigma_o,
    # in each element i_alpha_o[o] is index of alpha vector in o when calculating sigma_o
    i_alpha_o_list = [ c for c in itertools.product(list(range(len(nu))), repeat=observation_num) ]

    # add all next alpha vectors |A| * |nu|^|O|
    for a in A:
        for i_alpha_o in i_alpha_o_list:
            # initialize next alpha vector
            nu_a = np.zeros(state_num)
            # calculate next alpha vector in action a and in alpha vector combination i_alpha_o
            nu_a += r_sa[:, a]
            for o, i_alpha in enumerate(i_alpha_o):
                nu_a += gamma * alpha_[i_alpha][a][o]
            nu_.append( {'a':a, 'v':nu_a} )

    return nu_


def prune_op(nu, state_num):
    """
    Prune alpha vector set nu.
    Remove alpha vectors not chosen in calculating max alpha*b.
    Solve the below linear programing problem,
    max epsilon
    such that (alpha - alpha_)*b >= epsilon
    remove alpha if epsilon < 0.

    Args
    ----------
    nu: list
        Alpha vector set.
    state_num: int
        Number of state.

    Returns
    ----------
    nu_: list
        Pruned alpha vector set.
    """

    nu_ = []

    # parameters for linear programing problem
    c = np.append(np.zeros(state_num), [1.])
    A_ub_b = np.zeros((state_num, state_num+1))
    for s in range(state_num):
        A_ub_b[s][s] = -1.
    b_ub = np.zeros(state_num + len(nu) - 1)
    A_eq = [np.append(np.ones(state_num), [0.])]
    b_eq = [1.]

    removed_index = []

    for i, alpha_i in enumerate(nu):
        A_ub = A_ub_b
        for j, alpha_j in enumerate(nu):
            if i != j:
                A_ub = np.concatenate([ [np.append(-1 * (alpha_i['v'] - alpha_j['v']), [-1.])], A_ub ])
        # solve the linear programing problem
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
        if res.x[-1] > 0.:
            removed_index.append(i)

    # remove the alpha vector
    for i, alpha in enumerate(nu):
        if not i in removed_index:
            nu_.append(alpha)

    return nu_


def update_value_function(nu, p_t, p_o, r_sa, gamma, S, A, O):
    """
    Update value function V(b) = max alpha*b, by updating set of alpha vectors nu.

    Args
    ----------
    nu: list
        Set of alpha vectors before update.
        alpha vector is dictionary,
        key 'a' is action, key 'v' is alpha vector.
    p_t: list
        Transition probability.
        p_t[s'][s][a] is the probabilty of transitioning in state s',
        given startinig state s and executing action a.
    p_o: list
        Observation probability.
        p_o[o][a][s'] is the probabilty of getting observation o,
        given executing action a and transitioned state s'.
    r_sa: list
        Reward function.
        r_sa[s][a] is the reward,
        given startinig state s and executing action a.
    gamma: float
        discount rate.
    S: list
        Set of states.
        {0, 1, 2, 3, 4, ....}
    A: list
        Set of actions.
        {0, 1, 2, 3, 4, ....}
    O: list
        Set of observations.
        {0, 1, 2, 3, 4, ....}

    Returns
    ----------
    nu_: list
        Set of alpha vectors after update.
        alpha vector is dictionary,
        key 'a' is action, key 'v' is alpha vector.
    """

    state_num = len(S)
    action_num = len(A)
    observation_num = len(O)

    # initialize next alpha
    alpha_ = np.zeros((len(nu), action_num, observation_num, state_num))

    # calculate next alpha
    for i_alpha, alpha in enumerate(nu):
        for a in A:
            for o in O:
                for s in S:
                    alpha_[i_alpha][a][o][s] = np.sum( [ alpha['v'][s_] * p_o[o][a][s_] * p_t[s_][s][a] for s_ in S ] )

    # enumerate all alpha vectors
    nu_ = enum_op(nu, S, A, state_num, observation_num, r_sa, alpha_, gamma)

    # prune some alpha vectors
    nu_ = prune_op(nu_, state_num)

    return nu_


def get_best_action(b, nu):
    """
    Get best action corresponding to alpha chosen in V(b) = max alpha*b.

    Args
    ----------
    b: list
        Belief state.
        b[s] is the probabilty the state is s.
    nu: list
        Set of alpha vectors.
        alpha vector is dictionary,
        key 'a' is action, key 'v' is alpha vector.

    Returns
    ----------
    a: int
        Best action.
        alpha maximizing alpha*b correspond to a.
    """

    max_v = -np.inf
    a_best = -1

    for alpha in nu:
        v = np.dot(alpha['v'], b)
        if v > max_v:
            a_best = alpha['a']
            max_v = v

    if a_best == -1:
        print('unexpected action is chosen')
        sys.exit()

    return a_best
