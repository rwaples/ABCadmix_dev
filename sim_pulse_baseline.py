import numpy as np
# local imports
import simulate
import sum_stats
import rec_map


human_map = rec_map.get_human_rec_map()

Nrep = 20

for Tadmix in range(2, 50):
    res = []
    for frac in np.arange(.05, 1, .05):
        for rep in range(Nrep):
            for attemp in range(10):
                try:
                    ts = simulate.sim_pulse(
                        rec_map = human_map,
                        Tadmix = Tadmix,
                        frac_p0 = frac,
                        Nadmix = 1000,
                        seed = None,
                        path = None
                        )
                except:
                    pass
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
