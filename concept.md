```angular2html

Build system  ─► Bias (Metadynamics) ─► Run N short trajectories
      │                           │
      ▼                           ▼
  Save periodic snapshots  ◄──────┘
      │
      ▼
Extract features (distances, dihedrals, CVs)
      │
      ▼
Cluster snapshots  ─►  state index per frame
      │
      ▼
Count transitions at lag τ  ─►  C_ij
      │
      ▼
Row‑normalise           ─►  T_ij  (transition matrix)
      │
      ├─ Eigenvalues → implied time‑scales
      │
      └─ Stationary vector π → ΔG_i = –kT ln π_i


```

all of the above is done only for proteins right now.

create a program that could create a free energy map from markov state model data.
make a option to use replica exchange method to create better markov state models for the simulaion
- allow a simulation to perform a random walk in temperature space.

1. take a protein pdb file
- make it usable
- clear the protein
2. make a simulation



created replica exhange method
fixed the issue that were occuring with units switch

better and more refined minimization
- initial minimization with steepest descent
- refined minimization
- gradual heating(60% and than 40%) temperature equilibration
manager system that saved the progression in the output
amends in the DCD system


more conservative system

open and closed systems
- gradually get better and better definisions of that state
- stitch the data together and model things


create a vector field to visulizae what the protein is doing, like a map vector with diffusion example and only 2 dimensions



do the profiling of the module to find out how to optimize the performance
- only for the cpu right now


is it possible to use openMM molecular dynamics to use markov state models to obtain freeenergy landscape and from that findout the conformation states of the protein(the population of the conformations states). so we get the conformations that were found in the simulations and maybe findout more about the conformation states that come from informaiton that is not precisely from the molecular dynamics.
test the methods in a modular way, so that
- the equilibration and minimization are done in a modular/testable way


make it a real distribution package with wheel and poetry build
- dist(check if everything is correct as source)
- build(check if everything is correct)


monkey patching on the protein class to make it work with openMM
psutil to make it efficient

things that need to be done:
created a Python package;

set up Poetry for dependency management;

added a (fairly permissive) license to the project;

configured Poetry to allow uploading to PyPI and a test version of PyPI;

tested uploading the package to a test version of PyPI;

added Scriv to help us with changelog management and generation;

tagged and published a release of our project;

wrote a bunch of tests;

automated testing and linting with tox;

checked code coverage and got it to 100%;

set up CI/CD with GitHub actions to run linting and testing;

integrated with Codecov to get coverage reports in our pull requests;

created a GitHub action to publish our package to PyPI automatically; and

added some nifty badges to the README file.
