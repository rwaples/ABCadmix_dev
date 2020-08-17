import numpy as np
import simulate
import sum_stats
import rec_map

def sim_pulse_baseline(Nrep=20):
    """simulates replciates of pulse admixture events"""
    human_map = rec_map.get_human_rec_map()
    Nrep = int(Nrep)
    for Tadmix in range(25, 40):
        res = []
        for frac in np.arange(.05, 1, .05):
            for rep in range(Nrep):
                for attempt in range(25):
                    try:
                        ts = simulate.sim_pulse(
                            rec_map = human_map,
                            Tadmix = Tadmix,
                            frac = frac,
                            Nadmix = 1000,
                            seed = None,
                            path = None
                            )
                    except:
                        continue
                    else:
                        break
                else:
                    assert False, "Too many simulation errors"

                tern = sum_stats.get_ternary(ts)
                res.append((Tadmix, frac, rep, tern.astype('float16')))
        to_save = dict(zip(
                [f'{r[0]}_{int(r[1]*100)}_{r[2]}' for r in res], # key
                [r[3] for r in res] # value
            ))
        np.savez_compressed(f'./pulse.T_{Tadmix}.ternary.npz', **to_save)
        print(f'Done with T: {Tadmix}')

def main():
    sim_pulse_baseline()


if __name__ == '__main__':
    main()
