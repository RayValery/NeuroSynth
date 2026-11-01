"""Download EEGBCI and build the preprocessed dataset (.npz).

Thin command-line wrapper around `neurosynth.data`; run with --help for options.
"""
from neurosynth.data import main

if __name__ == "__main__":
    main()
