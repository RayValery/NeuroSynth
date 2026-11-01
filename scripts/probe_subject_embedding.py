"""Check whether a subject-conditional generator uses its subject embedding.

Thin command-line wrapper around `neurosynth.probe_subject_embedding`; run with --help for options.
"""
from neurosynth.probe_subject_embedding import main

if __name__ == "__main__":
    main()
