import scipy as sp
import random
import numpy as np
import pandas as pd
import itertools
import collections
import msprime
import tskit
from math import log
from math import exp

def sim_pulse(L = 1e9, Ne = 10000, Tadmix = 8, Nadmix = 500, seed = None, path = None):
    """
    simulate a simple pulse model of admixture.


    L = length of genome, in base pairs
    Ne = diploid population size for all three populations (2*source & admixed)
    Tadmix = time of admixture
    Nadmix = number of observed admixed individuals
    seed = seed to pass to msprime.simulate
    path = file path, if given will write the ts to this path
    """


    # convert to correct dtypes and catch problems
    Tadmix = int(Tadmix)
    L = int(L)
    Ne = int(Ne)
    Nadmix = int(Nadmix)

    # recombination map
    recomb_map = msprime.RecombinationMap.uniform_map(L, 1e-8, L)


    pop_configs = [
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0)
        ]

    # no ongoing migration.
    mig_mat = [
        [0,0,0],
        [0,0,0],
        [0,0,0],                             
    ]

    admixture_events = [
        # intial 50% fraction from each pop
        msprime.MassMigration(time=Tadmix, source=2, destination=0, proportion=0.5),
        msprime.MassMigration(time=Tadmix+1, source=2, destination=1,proportion=1.0),
    ]

    Nsamp = int(Nadmix)
    samps = [msprime.Sample(population=2, time=0)] * Nsamp

    ts_admix = msprime.simulate(
        population_configurations = pop_configs,
        migration_matrix = mig_mat,
        demographic_events = admixture_events,
        recombination_map = recomb_map,
        mutation_rate=0,
        model = 'dtwf',
        samples = samps,
        random_seed = seed,
        start_time = 0,
        end_time = Tadmix+2
    )

    return(ts_admix)
