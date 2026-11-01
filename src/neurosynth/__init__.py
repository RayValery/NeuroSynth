"""NeuroSynth: does diffusion-generated EEG beat signal augmentation for few-shot BCI calibration?

Modules
-------
config                   paths and constants shared by the analysis scripts
data                     EEGBCI download, band-pass, epoching, z-scoring -> .npz
dataset                  within-subject leave-one-run-out splits
augmentations            noise / time-shift / channel-dropout / crop
eegnet                   EEGNet classifier
train_baseline           per-fold classifier training: real-only, augmented, synthetic (TSTR)
diffusion                1D U-Net DDPM with class / domain / subject conditioning
train_diffusion          generator training, fidelity + memorization checks, subject banks
probe_subject_embedding  is the subject embedding used, and is it subject-specific?
"""

__version__ = "1.0.0"
