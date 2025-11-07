added:
- Reusable sampling and FES validation plot helpers in `pmarlo.visualization.diagnostics`, covered by dedicated unit tests.

changed:
- Streamlit validation workflows now call the shared visualization helpers directly, eliminating mock state shims.
- Base dependencies now include `matplotlib`, and README documents how to generate validation plots outside the web app.

