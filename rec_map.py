import msprime
import stdpopsim


def get_human_rec_map():
    """return a (discrete) recombination map for 22 human autosomes
    There is a 1 bp region of 0.5 recombination rate between each chromosome.
    """

    map_of_chr = {}
    species = stdpopsim.get_species('HomSap')
    for contig in [f'chr{x}' for x in range(1, 23)]:
        map_of_chr[contig] = species.get_contig(contig).recombination_map

    pos_list = []
    rates_list = []
    # shift the positions on each chromosome due to concatenation of the genome
    shifts = [0]
    for i in range(1, 23):
        chrom = f'chr{i}'
        pos = map_of_chr[chrom].get_positions()
        rates = map_of_chr[chrom].get_rates()
        rates[-1] = .5
        rates_list.extend(rates)
        if i > 1:
            shift = pos_list[-1]
            pos = [x+1+shift for x in pos]
            pos_list.extend(pos)
            shifts.append(shift)
        else:
            pos_list.extend(pos)

    num_loci = int(pos_list[-1] / 100) # loci of approx 100 bp
    human_map = msprime.RecombinationMap(
        positions=pos_list,
        rates=rates_list,
        num_loci=num_loci  # not sure the best option here
        # https://msprime.readthedocs.io/en/stable/tutorial.html#multiple-chromosomes
    )

    bp = human_map.get_sequence_length()
    M = human_map.get_total_recombination_rate()

    print('''human recombination map
        sequence length (bp) {bp}
        num_loci: {num_loci}
        total recombination rate (M):: {M:0.4}'''.format(bp=int(bp), num_loci=num_loci, M=M))

    return(human_map)
