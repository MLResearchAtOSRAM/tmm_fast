import os
from setuptools import setup

def read(fname):
    """utility function to read the README file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

descrip = ("This gym-related environment allows to interpret the optimization of angular and spectral reflectivity/transmission behavior of multi-layer thin films as a parameterized Markov decision process.\nThus, it can be solved by reinforcement learning. The multi-layer thin film computations are based on transfer-matrx method.")

data_files = ['README.md', 'LICENSE.txt', 'example.ipynb']

setup(
    name="mltf",
    version='0.0.1',
    author="Heribert Wankerl, Alexander Luce, Maike Lorena Stern",
    author_email="heribert.wankerl@osram-os.com",
    description=descrip,
    license="MIT",
    keywords="reinforcement learning, parameterized Markov decision process, multi-layer thin films, optics, transfer matrix method",
    url="https://github.com/MLResearchAtOSRAM",
    packages=['gym_mltf'],
    package_data={'gym_mltf':data_files},
    package_dir={'gym_mltf': '.'},
    long_description=read('README.md'),
    classifiers=[
        "Intended Audience :: Science/Research/Industry",
        "Topic :: Scientific/Engineering/Optics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"]
)