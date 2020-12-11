import numpy as np
from scipy.optimize import linear_sum_assignment
import numba as nb


def get_ternary(ts):
    """Compute the ternary ancestry fractions for each individual from a ts.

    ternary = ternary ancestry fractions (N, 3) array of floats
    N = number of indiviudals
    assumes just two admixing populations
    ts = tree-sequence
    """
    L = ts.sequence_length
    max_node_age = ts.tables.nodes.asdict()['time'].max()

    # match each interval in the samples to a ind from an ancestral population
    anc = ts.tables.map_ancestors(
        samples=ts.samples(),
        ancestors=np.where(
            (ts.tables.nodes.asdict()['population'] == 0)
            & (ts.tables.nodes.asdict()['time'] == max_node_age)
        )[0]
    )

    # ancestry of each interval
    pop_of_node = dict()
    for node in ts.nodes():
        pop_of_node[node.id] = node.population
    anc.ancestry = np.vectorize(pop_of_node.__getitem__)(anc.parent)

    # ind of each child (sample)
    Nsamp = len(ts.samples())
    Nind = int(Nsamp / 2)
    ind_of_sample = dict(zip(np.arange(Nsamp), np.arange(int(Nsamp / 2)).repeat(2)))
    anc.ind = np.vectorize(ind_of_sample.__getitem__)(anc.child)

    # compute the ternary fractions
    ternary = np.zeros([Nind, 3], dtype='float64')
    for i, ind in enumerate(range(Nind)):
        # get the unique ancestry switch points for the individual
        lefts = np.take(anc.left, np.where(anc.ind == ind))
        rights = np.take(anc.right, np.where(anc.ind == ind))
        endpoints = np.unique(np.concatenate([lefts, rights]))
        # and the length of each ancestry segment
        span = np.diff(endpoints)
        #  a point that should be inside each interval
        midpoints = endpoints[1:] - 1

        # for each midpoint how many intervals it is inside?
        inside_n = np.logical_and(
            midpoints.reshape(-1, 1) > lefts,
            midpoints.reshape(-1, 1) < rights
        ).sum(1)
        # add up the intervals that contribute to each
        frac_pop1pop1 = span[np.where(inside_n == 2)].sum() / L
        frac_pop1pop2 = span[np.where(inside_n == 1)].sum() / L
        frac_pop2pop2 = 1 - (frac_pop1pop1 + frac_pop1pop2)
        ternary[i] = (frac_pop1pop1, frac_pop1pop2, frac_pop2pop2)

    return(ternary)


@nb.njit(fastmath=True, parallel=False)
def costs_emd(A, B):
    """Compute the earth mover distance (EMD) between each pair of inds.

    given two arrays (N, 3) of ternary ancestry fractions for N individuals,
    selecting one individual for each A and B.
    Transportation costs for the EMD are 1 to adjacent bins.
    """
    assert A.shape[1] == 3
    assert B.shape[1] == 3
    assert A.shape[0] == B.shape[0]

    C = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)

    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            C[i, j] = np.abs((A[i, 0] - B[j, 0])) + np.abs((A[i, 2] - B[j, 2]))

    return(C)


@nb.njit(fastmath=True, parallel=True)
def costs_emd_parallel(A, B):
    """Parallel version of costs_emd().

    uses nb.prange() and parallel=True
    """
    assert A.shape[1] == 3
    assert B.shape[1] == 3
    assert A.shape[0] == B.shape[0]

    C = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)

    for i in nb.prange(A.shape[0]):
        for j in range(B.shape[0]):
            C[i, j] = np.abs((A[i, 0] - B[j, 0])) + np.abs((A[i, 2] - B[j, 2]))

    return(C)


def delta_emd(A, B):
    """Compute the minimum earth mover distance (EMD).

    between two arrays of ternary ancestry fractions
    by matching up pairs of individuals, one from each array.
    """
    cost_mat = costs_emd(A, B)
    assignment = linear_sum_assignment(cost_mat)
    delta = cost_mat[assignment].sum()

    return(delta)
