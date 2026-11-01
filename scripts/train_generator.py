"""Train the three run-holdout diffusion generators and export synthetic banks.

Thin command-line wrapper around `neurosynth.train_diffusion`; run with --help for options.
"""
from neurosynth.train_diffusion import main

if __name__ == "__main__":
    main()
