from distutils.core import setup

from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Vortex Lattice Method for yacht sailing'
LONG_DESCRIPTION = 'A package that calculate and visualize yacht during different wind conditions'

setup(
    name="sailingVLM",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Zuzanna Wieczorek, Grzegorz Gruszczynski",
    #license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords=['VLM', 'Vortex Lattice Method', 'sailingVLM', 'yacht'],
    python_requires='>3.10.4',
    classifiers= [
        #'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Education",
        "Software Development :: Libraries",
    ]
)
