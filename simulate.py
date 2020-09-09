import msprime

# TODO: use the save_ts() function with each simulation model
# TODO: try to standardize the parameter naming across simulation models


def sim_pulse(rec_map=None, L=1e9, Ne=10000, Nadmix=500,
            Tadmix=8, frac=0.5,
            seed=None, path=None, tszip=None):
    """Simulate a simple pulse model of admixture.

    With the disrete-time backwards wright-fisher.
    rec_map = valid msprime recombination map
    L = length of genome, in base pairs (ignored if rec_map is specified)
    Ne = diploid population size for all three populations
    Tadmix = time of admixture
    Nadmix = number of observed admixed diploid individuals
    seed = seed passed to msprime.simulate()
    path = file path, if given will write the ts to this path
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
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0)
    ]

    # no ongoing migration
    mig_mat = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    admixture_events = [
        msprime.MassMigration(time=Tadmix, source=2, destination=1, proportion=frac),
        msprime.MassMigration(time=Tadmix + 1, source=2, destination=0, proportion=1.0),
    ]

    samps = [msprime.Sample(population=2, time=0)] * 2 * Nadmix

    ts_admix = msprime.simulate(
        population_configurations=pop_configs,
        migration_matrix=mig_mat,
        demographic_events=admixture_events,
        recombination_map=recomb_map,
        mutation_rate=0,
        model='dtwf',
        samples=samps,
        random_seed=seed,
        start_time=0,
        end_time=Tadmix + 2
    )

    if path:
        if tszip:
            # save compressed ts
            import tszip
            tszip.compress(ts_admix, path, variants_only=False)
        else:
            # save uncompressed ts
            ts_admix.dump(path)

    return(ts_admix)


def sim_two_pulse(rec_map=None, L=1e9, Ne=10000, Nadmix=500,
                T1=4, T2=12, frac1=.2, frac2=.2,
                seed=None, path=None, tszip=None):
    """Simulate a simple pulse model of admixture.

    Using the disrete-time backwards wright-fisher.

    rec_map = valid msprime recombination map
    L = length of genome, in base pairs (ignored if rec_map is specified)

    Ne = diploid population size for all three populations
    Tadmix = time of admixture
    Nadmix = number of observed admixed diploid individuals
    seed = seed passed to msprime.simulate()
    path = file path, if given will write the ts to this path
    """

    assert T2 > T1, "T2 must be greater than T1"

    # convert to correct dtypes and catch problems
    T1 = int(T1)
    T2 = int(T2)
    Ne = int(Ne)
    Nadmix = int(Nadmix)

    # recombination map
    if rec_map:
        recomb_map = rec_map
    else:
        L = int(L)
        recomb_map = msprime.RecombinationMap.uniform_map(L, 1e-8, L)

    pop_configs = [
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0)
    ]

    # no ongoing migration
    mig_mat = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    admixture_events = [
        msprime.MassMigration(time=T1, source=2, destination=1, proportion=frac1),
        msprime.MassMigration(time=T2, source=2, destination=1, proportion=frac2),
        msprime.MassMigration(time=T2 + 1, source=2, destination=0, proportion=1.0),
    ]

    samps = [msprime.Sample(population=2, time=0)] * 2 * Nadmix

    ts_admix = msprime.simulate(
        population_configurations=pop_configs,
        migration_matrix=mig_mat,
        demographic_events=admixture_events,
        recombination_map=recomb_map,
        mutation_rate=0,
        model='dtwf',
        samples=samps,
        random_seed=seed,
        start_time=0,
        end_time=T2 + 2
    )

    if path:
        if tszip:
            # save compressed ts
            import tszip
            tszip.compress(ts_admix, path, variants_only=False)
        else:
            # save uncompressed ts
            ts_admix.dump(path)

    return(ts_admix)


def sim_ongoing_constant(rec_map=None, L=3e9, Ne=10000, Nadmix=500,
                        Tadmix=8, frac_ongoing=0.05,
                        seed=None, path=None, tszip=None):

    """Simulate an ongoing model of admixture.

    Using the disrete-time backwards wright-fisher.
    A new population (2) is formed by splitting off from population 0.
    At time=Tadmix migration starts from population 1, with rate frac_ongoing
    admixture continues until the present (time of sampling).

    rec_map = valid msprime recombination map
    L = length of genome, in base pairs (ignored if rec_map is specified)

    Ne = diploid population size for all three populations (2*source & admixed)
    Tadmix = time of admixture
    Nadmix = number of observed admixed individuals
    seed = seed to pass to msprime.simulate
    path = file path, if given will write the ts to this path (NOT IMPLEMENTED)
    """

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
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0)
    ]

    mig_mat = [
        [0, 0, 0],
        [0, 0, 0],
        [0, frac_ongoing, 0],
    ]

    admixture_events = [
        # set migration rates back to zero with the admixed population
        msprime.MigrationRateChange(time=Tadmix, rate=0, matrix_index=(2, 1)),
        msprime.MassMigration(time=Tadmix + 1, source=2, destination=0, proportion=1.0),
    ]

    samps = [msprime.Sample(population=2, time=0)] * 2 * Nadmix

    ts_admix = msprime.simulate(
        population_configurations=pop_configs,
        migration_matrix=mig_mat,
        demographic_events=admixture_events,
        recombination_map=recomb_map,
        mutation_rate=0,
        model='dtwf',
        samples=samps,
        random_seed=seed,
        start_time=0,
        end_time=Tadmix + 2
    )

    return(ts_admix)


def sim_ongoing_interval(rec_map=None, L=3e9, Ne=10000, Nadmix=500,
                Tadmix_start=4, Tadmix_stop=12, frac_ongoing=0.05,
                seed=None, path=None, tszip=None):

    """
    Simulate an ongoing model of admixture.

    With the disrete-time backwards wright-fisher.

    A new population (2) is formed by splitting off from population 0.
    At time=Tadmix_start migration starts from population 1,
    with rate frac_ongoing admixture continues until Tadmix_stop.

    rec_map = valid msprime recombination map
    L = length of genome, in base pairs (ignored if rec_map is specified)

    Ne = diploid population size for all three populations
    Tadmix = time of admixture
    Nadmix = number of observed admixed individuals
    seed = seed to pass to msprime.simulate
    path = file path, if given will write the ts to this path (NOT IMPLEMENTED)
    """

    assert Tadmix_stop > Tadmix_start, "Tadmix_stop must be greater than Tadmix_start"

    Tadmix_start = int(Tadmix_start)
    Tadmix_stop = int(Tadmix_stop)
    Ne = int(Ne)
    Nadmix = int(Nadmix)

    # recombination map
    if rec_map:
        recomb_map = rec_map
    else:
        L = int(L)
        recomb_map = msprime.RecombinationMap.uniform_map(L, 1e-8, L)

    pop_configs = [
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0),
        msprime.PopulationConfiguration(initial_size=Ne, growth_rate=0)
        ]

    mig_mat = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    admixture_events = [
        # migration during the interval Tadmix_start - Tadmix_stop
        msprime.MigrationRateChange(time=Tadmix_start, rate=frac_ongoing, matrix_index=(2, 1)),
        msprime.MigrationRateChange(time=Tadmix_stop, rate=0, matrix_index=(2, 1)),
        # founding of pop 2
        msprime.MassMigration(time=Tadmix_stop + 1, source=2, destination=0, proportion=1.0),
    ]

    samps = [msprime.Sample(population=2, time=0)] * 2 * Nadmix

    ts_admix = msprime.simulate(
        population_configurations=pop_configs,
        migration_matrix=mig_mat,
        demographic_events=admixture_events,
        recombination_map=recomb_map,
        mutation_rate=0,
        model='dtwf',
        samples=samps,
        random_seed=seed,
        start_time=0,
        end_time=Tadmix_stop + 2
    )

    return(ts_admix)
