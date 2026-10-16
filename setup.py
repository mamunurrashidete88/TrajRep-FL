from setuptools import setup, find_packages

setup(
    name="trajrep_fl",
    version="1.0.0",
    description="TrajRep-FL: Learning Client Trajectory Representations for Robust Federated Optimization",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "Pillow>=9.5.0",
    ],
)
