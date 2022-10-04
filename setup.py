#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name                 = 'morley',
    version              = '0.0.1',
    description          = '''GUI software for plant morphometry''',
    long_description     = None,
    # long_description_content_type="text/markdown",
    author               = 'Daria Emekeeva & Lev Levitsky',
    author_email         = None,
    install_requires     = ['pyteomics', 'seaborn', 'scipy', 'scikit-learn', 'opencv-python'],
    classifiers          = ['Intended Audience :: Science/Research',
                            'Programming Language :: Python :: 3',
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Intended Audience :: End Users/Desktop'],
    license              = 'License :: OSI Approved :: Apache Software License',
    packages             = find_packages(),
    entry_points         = {'console_scripts': ['morley=morley.main:main']},
    )