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

create a program that could create a free energy map from markov state model data.

1. take a protein pdb file
- make it usable
- clear the protein
2. make a simulation



is it possible to use openMM molecular dynamics to use markov state models to obtain freeenergy landscape and from that findout the conformation states of the protein(the population of the conformations states). so we get the conformations that were found in the simulations and maybe findout more about the conformation states that come from informaiton that is not precisely from the molecular dynamics. 
