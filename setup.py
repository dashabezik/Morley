#!/usr/bin/env python

from setuptools import setup, find_packages
import os

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name                 = 'morley',
    version              = get_version('morley/version.py'),
    description          = '''GUI software for plant morphometry''',
    long_description     = (''.join(open('README.md').readlines())),
    long_description_content_type="text/markdown",
    author               = 'Daria Emekeeva & Lev Levitsky',
    author_email         = 'dashabezik65@gmail.com',
    install_requires     = ['pyteomics', 'seaborn', 'scipy', 'scikit-learn', 'opencv-python', 'imutils'],
    classifiers          = ['Intended Audience :: Science/Research',
                            'Programming Language :: Python :: 3',
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Intended Audience :: End Users/Desktop'],
    license              = 'License :: OSI Approved :: Apache Software License',
    packages             = find_packages(),
    package_data         = {'morley': ['ethalon.png', 'logo.png']},
    entry_points         = {'console_scripts': ['morley=morley.main:main']},
    )
