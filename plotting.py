import matplotlib.pyplot as plt
import numpy as np

import scipy.interpolate as spi

# Open the output.txt file.
with open('output.txt') as file:
    # Import the data.
    lines = file.readlines()
    # Separate the data into numpy arrays.
    x_arr = np.fromstring(lines[0], dtype=float, sep=',')
    y_arr = np.fromstring(lines[1], dtype=float, sep=',')
    eval_arr = np.fromstring(lines[2], dtype=float, sep=',')
    lin_arr = np.fromstring(lines[3], dtype=float, sep=',')
    spline_arr = np.fromstring(lines[4], dtype=float, sep=',')

# Change the font settings and use LaTeX as the text renderer.
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['cm']})
plt.rc('text.latex', preamble=r'\usepackage{sfmath}')

# Calculate a linear interpolation using scipy.interpolate
lin_func = spi.interp1d(x_arr, y_arr)
sp_lin_arr = lin_func(eval_arr)

# Calculate a cubic spline using scipy.interpolate
spline_func = spi.CubicSpline(x_arr, y_arr, bc_type="natural")
sp_spline_arr = spline_func(eval_arr)

# Initialise the figure and set the size.
plt.figure(figsize=(6, 6))

# Plot the original data.
plt.plot(x_arr, y_arr, 'bx')
# Plot the linearlly interpolated data.
plt.plot(eval_arr, lin_arr, 'r--')
# Plot the cubic spline.
plt.plot(eval_arr, spline_arr, 'g-')

# Find the distance between the scipy interpolation and the C++ interpolation at each point.
lin_difference_arr = np.abs(sp_lin_arr - lin_arr)
spline_difference_arr = np.abs(sp_spline_arr - spline_arr)

print("The largest distance between the SciPy cubic spline interpolation and mine is ", np.max(spline_difference_arr), ".", sep="")
print("The mean distance between the SciPy cubic spline interpolation and mine is ", np.mean(spline_difference_arr), ".", sep="")
print("The largest distance between the SciPy linear interpolation and mine is ", np.max(lin_difference_arr), ".", sep="")
print("The mean distance between the SciPy linear interpolation and mine is ", np.mean(lin_difference_arr), ".", sep="")

# Include a legend.
plt.legend(['Data', 'Linear Interpolation', 'Cubic spline'], fontsize=14)

# Formatting suitable for insertion in the report.
plt.grid()
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16)

# Save the figure.
plt.savefig("interpolation.png")

# Dinplay the figure.
plt.show()
