## fixed
- Avoid raising when `min_count` masks every histogram cell by falling back to the unmasked density before smoothing so KDE blending can still run.
- Keep the adaptive Gaussian smoother output two-dimensional by gathering the blur kernels with `np.take_along_axis`, ensuring the smoothed surface can be compared directly with the histogram grid.
