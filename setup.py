"""
Setup script for PMARLO package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "PMARLO: Protein Markov State Model Analysis with Replica Exchange"

# Read requirements
requirements = [
    'numpy>=1.24.0,<2.3.0',  # Compatible with numba and mordred
    'scipy>=1.10.0',
    'matplotlib>=3.6.0',
    'pandas>=1.5.0',
    'scikit-learn>=1.2.0',
    'mdtraj>=1.9.0',
    'openmm>=8.0.0',
    'rdkit>=2024.03.1',
]

# Development requirements
dev_requirements = [
    'pytest>=6.0.0',
]

setup(
    name='pmarlo',
    version='0.1.0',
    description='Protein Markov State Model Analysis with Replica Exchange',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='PMARLO Development Team',
    author_email='pmarlo@example.com',
    url='https://github.com/yourusername/pmarlo',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'fixer': ['pdbfixer @ git+https://github.com/openmm/pdbfixer@v1.11'],
        'all': requirements + dev_requirements + ['pdbfixer @ git+https://github.com/openmm/pdbfixer@v1.11'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='molecular dynamics, markov state models, replica exchange, protein simulation, biophysics',
    entry_points={
        'console_scripts': [
            'pmarlo=pmarlo.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Documentation': 'https://pmarlo.readthedocs.io/',
        'Source': 'https://github.com/yourusername/pmarlo',
        'Tracker': 'https://github.com/yourusername/pmarlo/issues',
    },
)