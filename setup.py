#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name                 = 'morley',
    version              = '0.0.3',
    description          = '''GUI software for plant morphometry''',
    long_description     = None,
    # long_description_content_type="text/markdown",
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
