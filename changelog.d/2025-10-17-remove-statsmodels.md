### Removed

- Dropped the `statsmodels` dependency by replacing toroidal KDE and autocorrelation
  routines with internal NumPy/SciPy implementations so the example Streamlit app
  runs without extra site-packages. Removed `statsmodels` from requirements.txt and
  updated poetry.lock to reflect the changes.
