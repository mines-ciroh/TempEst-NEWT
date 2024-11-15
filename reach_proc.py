import NEXT.reach_prep as reach
import pandas as pd
import os
import numpy as np
import warnings
bp = "/scratch/dphilippus/notebooks/reach_analysis/"
bp_inpcache = bp + "input_cache/"

sitefile = bp + "sitelist.txt"
if os.path.exists(sitefile):
    with open(sitefile, "r") as sf:
        sites = [x.strip() for x in list(sf)]
else:
    sites = pd.read_csv("/scratch/dphilippus/notebooks/next_validation/DevData.csv", dtype={"id": "str"})["id"].unique()
    with open(sitefile, "w") as sf:
        sf.write("\n".join(sites))
rng = np.random.default_rng(42)
full_cache = bp + "reach_cache/"
reach_coefs = bp + "reach_coefs.csv"
start = "2015-01-01"
end = "2022-12-31"
rem_sites = sites
if os.path.exists(reach_coefs):
    existing = pd.read_csv(reach_coefs, dtype={"id": "str"})["id"].unique()
    rem_sites = [s for s in sites if not s in existing]
for site in sites:
    tries = 0
    inp = None
    inpfn = full_cache + site + ".csv"
    if os.path.exists(inpfn):
        inp = pd.read_csv(inpfn, dtype={"id": "str"}, parse_dates = ["date"])
    length = int(10**(rng.uniform(0, 2)))
    buffer = 100 * int(rng.uniform(1, 10))
    which_is_short = rng.choice([1, 2], 1)[0]
    while tries < 3 and inp is None:
        try:
            inp = reach.prepare_full_data("coefs.pickle", site, length, buffer, start, end, bp_inpcache, False)
            inp.to_csv(inpfn, index=False)
        except:
            # Failed to retrieve, so try a different buffer length.  First shorten by a factor of 5, then lengthen likewise - or the reverse.
            tries += 1
            print(f"Retrying {site}", end=" ")
            if tries == which_is_short:
                length = length // (5**tries)  # squared for second try, so it undoes the first iteration
            else:
                length *= (5**tries) # likewise
            if length < 1:
                length = 1
            if length > 100:
                length = 100
    if inp is None:
        print(f"Failed {site}", end=" ")
        continue  # just skip it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = reach.search_reach_coefficients(inp, (-1, 1), (-1, 1), (-1, 1), (0, 1), log=False, tolerance=0.0001)
    desc = inp.select_dtypes("number").agg("mean").to_dict() | {"length": length, "buffer": buffer}
    output = pd.DataFrame({"id": [site]} | desc | res, index = [0])
    exists = os.path.exists(reach_coefs)
    output.to_csv(reach_coefs, index=False, mode="a", header=not exists)