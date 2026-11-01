"""Train EEGNet per subject with leave-one-run-out evaluation (real-only, augmented or synthetic arm).

Thin command-line wrapper around `neurosynth.train_baseline`; run with --help for options.
"""
from neurosynth.train_baseline import main

if __name__ == "__main__":
    main()
