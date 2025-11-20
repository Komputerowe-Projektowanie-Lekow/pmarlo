### changed
- Refactored `EnhancedMSM` to compose loader, featurizer, clustering, estimation, plotting, and export components instead of inheriting a dozen mixins, making state ownership explicit and enabling targeted testing of each domain.
