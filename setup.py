"""Hi-Lo setup utility."""

from setuptools import find_packages, setup

setup(
    name =              "dadl-lab-cl",
    version =           "1.0.0",
    author =            "Gabriel C. Trahan, Ashton Andrepont",
    author_email =      "gabriel.trahan1@louisiana.edu, ashton.andrepont1@louisiana.edu",
    description =       (
                        "Research project to determine the efficacy of wavelet decomposition "
                        "in measuring image complexity."
                        ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/Image-Complexity-by-Wavelet-Decomposition",
    packages =          find_packages(),
    python_requires =   ">=3.10",
    install_requires =  [
        "matplotlib",
        "numpy",
        "pandas",
        "PyWavelets",
        "scikit-learn",
        "termcolor",
        "torch",
        "torchvision",
        "tqdm"
    ]
)