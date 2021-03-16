import numpy as np
from scipy.optimize import linear_sum_assignment
import numba as nb


import pandas as pd
import intervaltree
import tskit
import msprime


def get_ternary(ts):
    """Compute the ternary ancestry fractions for each individual from a ts.

    ternary = ternary ancestry fractions (N, 3) array of floats
    N = number of indiviudals
    assumes just two admixing populations
    ts = tree-sequence
    """
    L = ts.sequence_length
    max_node_age = ts.tables.nodes.asdict()['time'].max()

    # match each interval in the samples to an ind from an ancestral population
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
    
    
    
    
### gen 2 summary stats

# Normal admixture fractions
def tern_to_admix(tern):
    admix = np.ones((len(tern), 2))
    admix[:,0] = tern[:,0] + tern[:,1]/2
    admix[:,1] = tern[:,2] + tern[:,1]/2
    return(admix)


# ancestry decay
def get2(iterable):
    """return pairs of itesm from a list
    https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list/5389547
    """
    a = iter(iterable)
    return(zip(a,a))

def autocorr3(x):
    '''fft, pad 0s, non partial'''
    n=len(x)
    # pad 0s to 2n-1
    ext_size = 2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    xp = x - np.mean(x)
    var = np.var(x)

    # do fft and ifft
    cf=np.fft.fft(xp, fsize)
    sf=cf.conjugate()*cf # should be the same as 
    corr=np.fft.ifft(sf).real # inverse scaled by 1/n
    corr=corr/var/n

    return(corr)

def get_ancestry_decay(ts, genetic_map):
    max_node_age = ts.tables.nodes.asdict()['time'].max()

    # only look for ancestors from population 0
    anc = ts.tables.map_ancestors(
        samples = ts.samples(),
        ancestors = np.where((ts.tables.nodes.asdict()['population'] == 0) & (ts.tables.nodes.asdict()['time']==max_node_age))[0]
    )

    # ancestry of each interval
    pop_of_node = dict()
    for node in ts.nodes():
        pop_of_node[node.id] = node.population
    anc.ancestry = np.vectorize(pop_of_node.__getitem__)(anc.parent)

    # ind of each child (sample)
    Nsamp = len(ts.samples())
    Nind = int(Nsamp/2)
    ind_of_sample = dict(zip(np.arange(Nsamp), np.arange(int(Nsamp/2)).repeat(2)))
    anc.ind = np.vectorize(ind_of_sample.__getitem__)(anc.child)
    
    iterval = 500000
    keep_interval = 100 #only keep this many points
    running = np.zeros(keep_interval)
    count = 0
    # look at the decay within each ind within each chromosome
    # notice this is diploid ancestry [0,1,2] 
    for chrom_start, chrom_stop in get2(genetic_map.get_positions()):
        ancestry_poll_points = np.arange(chrom_start, chrom_stop, iterval)
        for ind in set(anc.ind):
            lefts = np.take(anc.left, np.where(anc.ind == ind))
            rights = np.take(anc.right, np.where(anc.ind == ind))
            ancestry_poll = np.logical_and(ancestry_poll_points.reshape(-1, 1) >= lefts, ancestry_poll_points.reshape(-1, 1) < rights).sum(1)
            # not sure which of the autocovariance functions is the fastest or most appropriate
            #autocorr = autocorrelation(ancestry_poll)
            #autocov = autocovariance_1(ancestry_poll)
            autocov = autocorr3(ancestry_poll)

            if np.isnan(autocov[0]):
                # didn't work, skip
                pass
            else:
                to_add = np.zeros(keep_interval)
                #a = np.real(autocorr)[:keep_interval]
                a = np.real(autocov)[:keep_interval]
                to_add[:a.shape[0]] = a
                running += to_add
                count += 1 # could be converted to a weight 
    
    return(running, count)

# Length of ancestry tracts

def get_tract_lengths(ts, genetic_map):
    max_node_age = ts.tables.nodes.asdict()['time'].max()

    anc = ts.tables.map_ancestors(
        samples = ts.samples(),
        ancestors = np.where((ts.tables.nodes.asdict()['population'] == 0) & (ts.tables.nodes.asdict()['time']==max_node_age))[0]
    )

    # ancestry of each interval
    pop_of_node = dict()
    for node in ts.nodes():
        pop_of_node[node.id] = node.population
    anc.ancestry = np.vectorize(pop_of_node.__getitem__)(anc.parent)

    local_df = pd.DataFrame({
        'left': anc.left, 
        'right': anc.right,
        'span': anc.right - anc.left,
        'parent': anc.parent,
        'child': anc.child,
        'ancestry' : anc.ancestry,
        'CHR': None
                 })

    local_df = local_df.query('span > 1').copy()

    local_df = local_df.sort_values(['child', 'left']).reset_index(drop=True)

    CHR = 1
    for chrom_start, chrom_stop in get2(genetic_map.get_positions()):
        begin = chrom_start-1
        end = chrom_stop+1
        a = local_df.query('(left > @begin) & (right < @end)')
        local_df.loc[a.index, 'CHR'] = CHR
        CHR+=1

    # each of these is contained within a single chromosome
    to_combine = local_df.query('CHR > 0').copy()

    match_pos = to_combine['left'].values[1:] == to_combine['right'].values[:-1]
    match_child = to_combine['child'].values[1:] == to_combine['child'].values[:-1]
    match_yes = np.logical_and(match_pos, match_child)
    to_combine['match'] = np.append(match_yes, [False])

    to_combine['dummy_group'] = (to_combine['match'] != to_combine['match'].shift()).cumsum()
    #to_combine

    tracts = to_combine.groupby('dummy_group')['span'].sum().values
    #tracts


    # each of these intervals spans at least two chromosomes
    multi_CHR = local_df.query('not CHR > 0')
    len(multi_CHR)

    ends = np.array(genetic_map.get_positions(), dtype = 'int')


    it = intervaltree.IntervalTree.from_tuples(zip(multi_CHR.left, multi_CHR.right, multi_CHR.index))
    for IDX in range(0, 21):
        it.chop(ends[2*IDX+1], ends[2*IDX+2])

    # you can just take the span from each of these split intervals
    spans = []
    for l, r, index in it.items():
        spans.append(r-l)
    all_tracts = np.append(tracts, spans)
    len(all_tracts)

    #sns.displot(all_tracts*genetic_map.mean_recombination_rate,
    #            log_scale = True)

    counts, bins = np.histogram(all_tracts, bins = np.arange(0, 249250621, 5e5))
    return(counts)
