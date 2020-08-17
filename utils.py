import tskit
import msprime


def save_ts(ts, path, tszip=False):
    """function to save ts to a file
    path gives the filename
    if tszip evaluates to True, the output will be comrpessed tszip
    tszip compression can can reduce files sizes,
    but adds time to import and export steps. 
    """
    if tszip:
        # save compressed ts
        try:
            import tszip
        except ImportError:
            assert False, "tszip compression requires tszip package"
        tszip.compress(ts, path, variants_only=False)
    else:
        # save uncompressed ts
        ts.dump(path)
