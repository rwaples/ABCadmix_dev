import msprime
import tskit


def sim_pulse(rec_map = None, L = 1e9, Ne = 10000, Nadmix = 500,
    Tadmix = 8, frac_p0 = 0.5,
    seed = None, path = None, tszip = None):
    """
    simulate a simple pulse model of admixture with the disrete-time backwards wright-fisher.

    rec_map = valid msprime recombination map
    L = length of genome, in base pairs (ignored if rec_map is specified)
    
    Ne = diploid population size for all three populations (2*source & admixed)
    Tadmix = time of admixture
    Nadmix = number of observed admixed individuals
    seed = seed to pass to msprime.simulate
    path = file path, if given will write the ts to this path (NOT IMPLEMENTED)
    """


    # convert to correct dtypes and catch problems
    Tadmix = int(Tadmix)
    Ne = int(Ne)
    Nadmix = int(Nadmix)

    # recombination map
    if rec_map:
        recomb_map = rec_map
    else:
        L = int(L)
        recomb_map = msprime.RecombinationMap.uniform_map(L, 1e-8, L)


    pop_configs = [
        msprime.PopulationConfiguration(initial_size = Ne, growth_rate = 0),
        msprime.PopulationConfiguration(initial_size = Ne, growth_rate = 0),
        msprime.PopulationConfiguration(initial_size = Ne, growth_rate = 0)
        ]

    # no ongoing migration
    mig_mat = [
        [0,0,0],
        [0,0,0],
        [0,0,0],                             
    ]

    admixture_events = [
        msprime.MassMigration(time=Tadmix, source = 2, destination = 0, proportion = frac_p0),
        msprime.MassMigration(time=Tadmix + 1, source = 2, destination = 1, proportion = 1.0),
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
    
    if path:
        if tszip:
            # save compressed ts
            import tszip
            tszip.compress(ts_admix, path, variants_only = False)
        else:
            # save uncompressed ts
            ts_admix.dump(path)
            

    return(ts_admix)
