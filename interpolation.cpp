#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "utils.h"
#include "matrix_LU.h"

// These functions are designed to be used with any user input, and therefore
// have data validation to ensure, for example, that the x and y data inputs are
// the same size.

// The first few functions in this file are used for sorting data - the
// interpolation algorithms only work if the x-data is in the correct order.
// The algorithms therefore assume that the y-data corresponds to the x-data,
// and sort the x data into ascending order, whilst also moving the
// corresponding y-data.

// Using a template allows the same function to be used to for data sets of
// different types, for example (double, double) or (unsigned, double).

template <typename T1, typename T2>
struct xy_pair {
    /**
     * struct xy_pair
     *
     * Pairs the x and y data for sorting.
     *
     * Members:
     *   T1 x_data: A single point of x data of type T1.
     *   T2 y_data: The corresponding point of y data of type T2.
     */
    T1 x_data;
    T2 y_data;
}; // Structs must end with a semicolon.

template <typename T1, typename T2>
bool sort_func(xy_pair<T1, T2> pair_1, xy_pair<T1, T2> pair_2) {
    /**
     * function sort_func
     *
     * Given a two pairs of x-y data, returns true if the first x-coordinate is less than the second.
     * Used in sorting the x and y arrays by the values of the x-data.
     *
     * Parameters:
     *   xy_pair<T1, T2> pair_1: An xy_pair struct containing data of type T1 and then data of type T2.
     *   xy_pair<T1, T2> pair_2: A second xy_pair struct containing the same data types as pair_1.
     *
     * Returns:
     *   bool: True iff pair_1.x_data > pair2.x_data;
     */
  return pair_1.x_data < pair_2.x_data;
}

template <typename T1, typename T2>
void sort_xy(std::vector<T1> & x_arr, std::vector<T2> & y_arr) {
    /**
     * function sort_xy
     *
     * Given an array of x_data and an array of y data, sorts the x array and moves the y array correspondingly.
     * The arrays are passed in by reference because the function actually changes the input arrays rather than outputting new arrays.
     *
     * Parameters:
     *   std::vector<T1> x_arr: A vector of type T1 containing the x data.
     *   std::vector<T2> y_arr: A vector of type T2 containing the y data.
     *
     * Errors:
     *   Throws an error if x_arr and y_arr are different sizes.
     */

    unsigned arr_size = x_arr.size();

    // Data validation
    try {
        // Check that the arrays are the same size.
        if (y_arr.size() != arr_size) {
          // Throw an error.
          throw 1;
        }
    } catch (int e) {
        // Print an error statement.
        std::cout << "sort_xy: x_arr and y_arr must be the same size." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Use the xy_pair struct to group together the x and y data into pairs.
    std::vector<xy_pair<T1, T2> > bundle(arr_size);

    // Loop through the arrays.
    for (unsigned i = 0; i < arr_size; ++i) {
        // Store the current x_arr element into the x_data member of the struct,
        bundle[i].x_data = x_arr[i];
        // Store the current y_arr element into the y_data member of the struct,
        bundle[i].y_data = y_arr[i];
    }

    // Use the standard sort function to sort the vector of pairs.
    // Pass the function sort_func to sort by x_data value only.
    std::sort(bundle.begin(), bundle.end(), sort_func<T1, T2>);

    // Loop through the original arrays and pass in the new, sorted elements.
    for (unsigned i = 0; i < arr_size; ++i) {
      x_arr[i] = bundle[i].x_data;
      y_arr[i] = bundle[i].y_data;
    }
}

std::vector<unsigned> sort_eval_arr (std::vector<double> & eval_arr) {
    /**
     * function sort_eval_arr
     *
     * Given an array of points, sorts the array, keeping track of where the original indices were.
     *
     * Parameters:
     *   std::vector<double> eval_arr: A vector of doubles at which to evaulate the interpolation.
     *
     * Returns:
     *   std::vector<unsigned>: A vector containing a list of the original indicex corresponding to each element in the eval_arr.
     */

    unsigned arr_size = eval_arr.size();

    // Create a vector to store the index of each element.
    std::vector<unsigned> index_arr(arr_size);

    // Generate the index array by putting 0 in the first element, 1 in the
    // second, 2 in the third, etc.
    for (unsigned i = 0; i < arr_size; ++i) {
        index_arr[i] = i;
    }

    // Sort the arrays.
    sort_xy(eval_arr, index_arr);

    return index_arr;
}

std::vector<double> reorder_arr (const std::vector<double> & unordered_arr, const std::vector<unsigned> & index_arr) {
    /**
     * function reorder_outuput.
     *
     * Given an array of points, sorts the array in the order given by index_arr
     *
     * Parameters:
     *   std::vector<double> unordered_arr: A vector of doubles to reorder.
     *   std::vector<double> index_arr: A vector
     *
     * Returns:
     *   data: A struct containing the sorted data in the x_arr, and a list of the original index order in the y_arr.
     */

    unsigned arr_size = unordered_arr.size();

    // Intialise the output vector.
    std::vector<double> output(arr_size);

    // Reorder the array using index_arr.
    for (unsigned i = 0; i < arr_size; ++i) {
        output[index_arr[i]] = unordered_arr[i];
    }

    return output;
}

// In the interpolation functions, vectors are passed by value instead of by
// const reference because the vectors are changed inside the function.

std::vector<double> linear_interp(std::vector<double> x_arr, std::vector<double> y_arr, std::vector<double> eval_arr) {
    /**
     * function linear_interp
     *
     * Performs linear interpolation on a tabulated set of x-y data.
     * This function does not require sorted data, however it does assume that the order of the x-data corresponds to the order of the y-data.
     *
     * Parameters:
     *   std::vector<double> x_arr: A vector of doubles containing the x data.
     *   std::vector<double> y_arr: A vector of doubles containing the y data.
     *   std::vector<double> eval_arr: A vector of doubles at which to evaulate the interpolation.
     *
     * Returns:
     *   std::vector<double>: A vector containing the interpolated values at x-values correspding to the input eval_arr.
     *
     * Error:
     *   Throws an error if x_arr and y_arr are different sizes.
     */

    unsigned points = x_arr.size();
    unsigned output_length = eval_arr.size();

    // Data validation
    try {
        // Check that the arrays are the same size.
        if (y_arr.size() != points) {
            throw 1;
        }
    } catch (int e) {
        // Print an error statement.
        std::cout << "linear_interp: x_arr and y_arr must be the same size." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Sort the x and y arrays.
    sort_xy(x_arr, y_arr);

    // Sort the eval_arr.
    std::vector<unsigned> index_arr = sort_eval_arr(eval_arr);

    // Create an array to store the output in the order generated by the
    // algorithm.
    std::vector<double> unordered_output(output_length);

    // Store the current index in the x_arr, starting at zero.
    unsigned x_index = 0;

    // Loop through the unordered_output array.
    for (unsigned i = 0; i < output_length; ++i) {
        // If the point at which the interpolation is requested is larger than
        // the current x-value, increase the value of x_index to move to the
        // next value in the x_arr.
        if (eval_arr[i] > x_arr[x_index + 1]) {
            ++x_index;
        }

        // Use the equation for linear interpolation to find the interpolated
        // value. See Computational Physics, Lecture 4 - Interpolation, Slide 5
        // [Y. Uchida, 2017].

        // Find the value of (x_{i+1} - x)/(x_{i+1} - x_i), where x is the value
        // at which the interpolation is being evaluated.
        double a =  (x_arr[x_index + 1] - eval_arr[i])/(x_arr[x_index + 1] - x_arr[x_index]);

        // Find the value of (x - x_i)/(x_{i+1} - x_i).
        double b = 1 - a;

        // Find the value of a*(y_i) + b*(y_{i+1}) and store it to the output
        // array.
        unordered_output[i] = a * y_arr[x_index] + b * y_arr[x_index + 1];
    }

    // Reorder the output so that its order corresponds with the input arrays.
    std::vector<double> output = reorder_arr(unordered_output, index_arr);

    return output;

}

std::vector<double> cubic_spline(std::vector<double> x_arr, std::vector<double> y_arr, std::vector<double> eval_arr) {
    /**
     * function cubic_spline
     *
     * Performs cubic spline interpolation on a tabulated set of x-y data using the natural spline boundary condition.
     * Uses a linear time point-finding algorithm.
     *
     * Parameters:
     *   std::vector<double> x_arr: A vector of doubles containing the x data.
     *   std::vector<double> y_arr: A vector of doubles containing the y data.  Each y point must correspond to the x point at the same index.
     *   std::vector<double> eval_arr: A vector of doubles at which to evaulate the interpolation.
     *
     * Returns:
     *   std::vector<double>: A vector containing the interpolated values x-values correspding to the input eval_arr.
     *
     * Error:
     *   Throws an error if x_arr and y_arr are different sizes.
     */

    unsigned points = x_arr.size();
    unsigned output_length = eval_arr.size();

    // Data validation
    try {
        // Check that the arrays are the same size.
        if (y_arr.size() != points) {
            throw 1;
        }
    } catch (int e) {
        // Print an error statement.
        std::cout << "cubic_spline: x_arr and y_arr must be the same size." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Sort the x and y arrays.
    sort_xy(x_arr, y_arr);

    // Sort the eval_arr.
    std::vector<unsigned> index_arr = sort_eval_arr(eval_arr);

    // Create a square matrix with two fewer rows than number of x and y points.
    Matrix<double> matrix(points - 2, points - 2);

    // This matrix is a tridiagonal matrix. The diagonal terms on row i-1 are
    // (x_{i+1} - x_{i-1})/3. The terms in the lower diagonal on row i-1 are
    // (x_i - x_{i-1})/6. The terms in the upper diagonal on row i-1 are
    // (x_{i+1} - x_i)/6. Computational Physics, Lecture 4 - Interpolation,
    // Slide 13 [Y. Uchida, 2017].

    // Calculate the values in the first row of the matrix.
    matrix[0][0] = (x_arr[2] - x_arr[0])/3.0;
    matrix[0][1] = (x_arr[2] - x_arr[1])/6.0;

    // Loop over the rows in the matrix containing three non-zero terms.
    for (unsigned i = 1; i < points - 3; ++i) {
        // Calculate the value of the lower diagonal term.
        matrix[i][i-1] = (x_arr[i + 1] - x_arr[i])/6.0;
        // Calculate the value of the diagonal term.
        matrix[i][i] = (x_arr[i + 2] - x_arr[i])/3.0;
        // Calculate the value of the upper diagonal term.
        matrix[i][i+1] = (x_arr[i + 2] - x_arr[i + 1])/6.0;
    }

    // Calculate the values in the last row of the matrix.
    matrix[points - 3][points - 4] = (x_arr[points - 2] - x_arr[points - 3])/6.0;
    matrix[points - 3][points - 3] = (x_arr[points - 1] - x_arr[points - 3])/3.0;

    // Calculate the LU decomposition of the matrix using Crout's algorithm
    // from the matrix_LU header file.
    matrix_LU::LU_output<double> LU = matrix_LU::crout(matrix);

    // Initialise a matrix as a column vector with two fewer rows than the
    // number of x and y points.
    Matrix<double> vector_b(1, points - 2);

    // This column vector is the solution to the product of the matrix above
    // with a column vector containing the second derivatives of the spline
    // at each of the values in the x_arr from x_arr[1] to x_arr[points - 1],
    // allowing solving for these values using forward and back substitution.

    // Loop through vector_b finding the value at each index.
    for (unsigned i = 0; i < points - 2; ++i) {
        // The vector at index i - 1 has value:
        // (y_{i+1} - y_i)/(x_{i+1} - x_i)  - (y_i - y_{i-1})/(x_i - x_{i-1}).
        // See Computational Physics, Lecture 4 - Interpolation, Slide 13
        // [Y. Uchida, 2017].
        double term_1 = (y_arr[i+2] - y_arr[i+1])/(x_arr[i+2] - x_arr[i+1]);
        double term_2 = (y_arr[i+1] - y_arr[i])/(x_arr[i+1] - x_arr[i]);
        vector_b[i][0] = term_1 - term_2;
    }

    // Use forward and back substitution to solve for the second derivatives.
    Matrix<double> vector_x = matrix_LU::linsolve_LU(LU.matrix_L, LU.matrix_U, vector_b);

    // Initialise a vector to hold the second derivatives.
    std::vector<double> y_doubleprime(points);

    // Implement the natural spline boundary condition by seting the first and
    // last points of the y_doubleprime vector to zero.
    y_doubleprime[0] = 0;
    y_doubleprime[points - 1] = 0;

    // Loop through the y_doubleprime std::vector and copy over the required
    // values from the Matrix vector_x.
    for (unsigned i = 1; i < points - 1; ++i){
        y_doubleprime[i] = vector_x[i - 1][0];
    }

    // Create an array to store the output in the order generated by the
    // algorithm.
    std::vector<double> unordered_output(output_length);

    // Store the current index in the x_arr, starting at zero.
    unsigned x_index = 0;

    // Loop through the unordered_output array.
    for (unsigned i = 0; i < output_length; ++i) {
        // If the point at which the interpolation is requested is larger than
        // the current x-value, increase the value of x_index to move to the
        // next value in the x_arr.
        if (eval_arr[i] > x_arr[x_index + 1]) {
            ++x_index;
        }

        // Use the equation for cubic splines to find the interpolated value.
        // See Computational Physics Lecture Notes, Chapter 4 - Interpolation
        // [Y. Uchida, P. Scott, et al., 2017]

        // Find the value of (x_{i+1} - x)/(x_{i+1} - x_i), where x is the value
        // at which the interpolation is being evaluated.
        double a =  (x_arr[x_index + 1] - eval_arr[i])/(x_arr[x_index + 1] - x_arr[x_index]);

        // Find the value of (x - x_i)/(x_{i+1} - x_i).
        double b = 1 - a;

        // Find the value of ((x_{i+1} - x) ^ 2)/6.
        double shared = pow((x_arr[x_index + 1] - x_arr[x_index]) , 2) / 6.0;

        // Find the value of ((a^3 - a) * (x_{i+1} - x) ^ 2)/6.
        double c = (pow(a, 3) - a) * shared;

        // Find the value of ((b^3 - b) * (x_{i+1} - x) ^ 2)/6.
        double d = (pow(b, 3) - b) * shared;

        // Find the value of a*(y_i) + b * (y_{i+1}) + c * (y''_i) +
        // d * (y''_{i+1}) and store it to the output array.
        unordered_output[i] = a * y_arr[x_index] + b * y_arr[x_index + 1] + c * y_doubleprime[x_index] + d * y_doubleprime[x_index + 1];
    }

    // Reorder the output so that its order corresponds with the input arrays.
    std::vector<double> output = reorder_arr(unordered_output, index_arr);

    return output;
}


int main(int argc, char ** argv) {

    // Generate the arrays of x values and y_values.
    std::vector<double> x_arr = {-2.1, -1.45, -1.3, -0.2, 0.1, 0.15, 0.8, 1.1, 1.5, 2.8, 3.8};
    std::vector<double> y_arr = {0.012155, 0.122151, 0.184520, 0.960789, 0.990050, 0.977751, 0.527292, 0.298197, 0.105399, 3.936690e-4, 5.355348e-7};

    // Generate an array of 10000 points at which to evaluate the interpolation.
    // Note the use of utils:: which accessess the utils namespace from utils.h
    std::vector<double> eval_arr = utils::linspace(-2.099, 3.799, 10000);

    // Generate the arrays of interpolated values.
    std::vector<double> lin_arr = linear_interp(x_arr, y_arr, eval_arr);
    std::vector<double> spline_arr = cubic_spline(x_arr, y_arr, eval_arr);

    // Create an output file.
    std::ofstream output("output.txt");

    // Store the vectors to the file using the vector_to_file function from
    // utils.h, to be imported to python for plotting.
    utils::vector_to_file(output, x_arr);
    utils::vector_to_file(output, y_arr);
    utils::vector_to_file(output, eval_arr);
    utils::vector_to_file(output, lin_arr);
    utils::vector_to_file(output, spline_arr);

  return 0;
}
