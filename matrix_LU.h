#ifndef MATRIX_LU_H
#define MATRIX_LU_H

// Define MATRIX_LU_H at compile time to ensure the module is not loaded twice.
// Only load the module if MATRIX_LU_H has not previously been defined.

// This header file contains the "matrix.h" file from the Matrices part of the
// project, as well as the functions written for that part utilising LU
// decomposition.

// Functions outside of the matrix class are wrapped in the namespace matrix_LU
// to prevent conflicts.

#include <iostream>
#include <vector>
#include <algorithm>

// Using a template allows the existence of matrices with integer, float, double
// or long double elements.

template <typename T>
class Matrix {
    /**
     * class Matrix
     * A class that implements matrix operations.
     *
     * The class is designed to work with any type T. To initialise the class, use Matrix <T>, e.g. Matrix <double>.
     *
     * Properties:
     *   Private:
     *     rows: The number of rows in the matrix.
     *     cols: The number of columns in the matrix.
     *     matrix: A vector of vectors storing the data.
     *
     * Methods:
     *   Public:
     *     Matrix: Class constructor.
     *     get_rows: Returns the number of rows in the matrix.
     *     get_cols: Returns the number of columns in the matrix
     *     print: Displays the matrix in the console.
     *     operator[]: Returns a row of the matrix given an index.
     *     operator+: Adds two matrices.
     *     operator+=: Adds to a matrix.
     *     operator-: Subtracts two matrices.
     *     operator-=: Subtracts from a matrix.
     *     operator*: Multiplies two matrices or multiplies a matrix by a scalar.
     *     operator*=: Multiplies and updates a matrix.
     *     operator/: Divides a matrix by a scalar.
     *     operator/=: Divides and updates a matrix.
     */
private:
    unsigned rows; /** The number of rows in the matrix. */
    unsigned cols; /** The number of columns in the matrix. */
    std::vector< std::vector<T> > matrix; /** A vector of vectors storing the data */
public:
    explicit Matrix (const std::vector< std::vector<T> > input_vector) {
        /**
         * function Matrix
         * Matrix class constructor with only 1 input.
         *
         * Parameters:
         *   std::vector< std::vector<T> > input_vector: A vector of vectors of type T used as the data for the matrix.
         *
         * Returns:
         *   Matrix <T>: A Matrix object containing the data.
         *
         * Errors:
         *   Throws an error if any of the vectors in input_vector is a different size to the first vector.
         */


        unsigned column_no = input_vector[0].size();

        // Data validation.
        try {
            // Loop through each vector in input_vector.
            for (const std::vector<T> & row : input_vector){
                // Check that the current vector is the same size as the first vector.
                if (row.size() != column_no) {
                    // Throw an error.
                    throw 1;
                }
            }
        }
        catch (int e) {
            // Print an error statement.
            std::cout << "Matrix constructor: Inconsistent number of columns" << std::endl;
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise the class properties.
        rows = input_vector.size();
        cols = column_no;
        matrix = input_vector;
    }

    Matrix (const unsigned x_size, const unsigned y_size) {
        /**
         * function Matrix
         * Matrix constructor for an empty matrix.
         *
         * Parameters:
         *   const unsigned x_size: The number of columns of the new matrix.
         *   const unsigned y_size: The number of rows of the new matrix.
         *
         * Returns:
         *   Matrix <T>: An empty matrix with the desired number of rows and columsn
         */

        // Create an empty matrix.
        matrix = std::vector< std::vector<T> > (y_size, std::vector<T>(x_size));

        // Initialise the other class properties.
        rows = y_size;
        cols = x_size;
    }

    Matrix () {
        /**
         * function Matrix
         * Matrix constructor for an empty matrix of undetermined size.
         *
         * Returns:
         *   Matrix<T>: A empty matrix with no size.
         */

        // Create an empty matrix with no size.
        matrix = std::vector< std::vector<T> > (0, std::vector<T>(0));

        // Initialise the other class properties.
        rows = 0;
        cols = 0;
    };

    std::vector<T>& operator[] (const int index) {
        /**
         * function operator[]
         * Returns a row of the matrix given an index.
         *
         * Overrides the [] operator so that the matrix vector of vectors does not need to be accessed directly. A
         * second [] operator then be used to access the desired matrix entry. This can be used to change values in the
         * matrix.
         *
         * Parameters:
         *   const int index: The desired row index.
         *
         * Returns:
         *   std::vector<T>&: A reference to the desired vector.
         */
        return matrix[index];
    }

    const std::vector<T>& operator[] (const int index) const {
        /**
         * function operator[]
         * Returns a row of a constant matrix given an index.
         *
         * Overrides the [] operator for const Matrix objects so that the matrix vector of vectors does not need to be
         * accessed directly. A second [] operator then be used to access the desired matrix entry. This CANNOT be used
         * to change values in the matrix.
         *
         * Parameters:
         *   const int index: The desired row index.
         *
         * Returns:
         *   const std::vector<T>&: A reference to the desired vector.
         */
        return matrix[index];
    }

    unsigned get_rows(){
        /**
         * function get_rows
         * Returns the number of rows in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.rows.
         */
        return rows;
    }

    unsigned get_cols(){
        /**
         * function get_cols
         * Returns the number of columns in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.cols.
         */

        return cols;
    }

    unsigned get_rows() const {
        /**
         * function get_rows
         * Returns the number of rows in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.rows.
         */
        return rows;
    }

    unsigned get_cols() const {
        /**
         * function get_cols
         * Returns the number of columns in the matrix.
         *
         * Returns:
         *   unsigned: The value of matrix.cols.
         */

        return cols;
    }

    // Declaring functions that are defined below.

    void print();
    Matrix<T> operator+ (const Matrix<T> & matrix2);
    Matrix<T>& operator+= (const Matrix<T> & matrix2);
    Matrix<T> operator- (const Matrix<T> & matrix2);
    Matrix<T>& operator-= (const Matrix<T> & matrix2);
    Matrix<T> operator* (const Matrix<T> & matrix2);
    Matrix<T>& operator*= (const Matrix<T> & matrix2);
    Matrix<T> operator* (T value);
    Matrix<T>& operator*= (T value);
    Matrix<T> operator/ (T value);
    Matrix<T>& operator/= (T value);

};

template <typename T>
void Matrix<T>::print () {
    /**
     * function print
     * Displays the matrix in the console. Member function of Matrix.
     */

    // Print a '[' and a newline character to start the output.
    std::cout << '[' << '\n';

    // Loop through each row of the matrix.
    for (unsigned i = 0; i < rows; ++i){
        // Print a '[' to start the row.
        std::cout << '[';
        // Loop through each element in the row.
        for (unsigned j = 0; j < cols; ++j){
            // Print the current element.
            std::cout << ' ' << matrix[i][j] << ' ';
        }
        // Print a ']' and a newline character to end the row.
        std::cout << ']' << '\n';
    }

    // Print a ']' to end the output.
    std::cout << ']' << std::endl;
}


template <typename T>
Matrix<T> Matrix<T>::operator+ (const Matrix<T> & matrix2){
    /**
     * function operator+
     * Adds two matrices.
     *
     * Overrides the + operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *   const Matrix<T> matrix2: A second matrix to add to the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>: The sum of the two matrices.
     *
     * Errors:
     *   Throws an error if the matrices are different shapes.
     */

    // Data validation.
    try {
        // Check that the matrices are the same shape.
        if (!(rows == matrix2.rows && cols == matrix2.cols)) {
            // Throw an error.
            throw 1;
        }
    }
    catch (int e) {
        // Print an error statenent.
        std::cout << "Matrix addition: matrices different shapes" << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < rows; ++i){
        // Loop through each item in the row.
        for (unsigned j = 0; j < cols; ++j){
            // Add the values of the matrices at the current index.
            output[i][j] = matrix[i][j] + matrix2[i][j];
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+= (const Matrix<T> & matrix2) {
    /**
     * function operator+=
     * Uses Matrix<T>::operator+ to add to a matrix.
     *
     * Overrides the += operator for a Matrix object.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix to add to the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
     */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this + matrix2;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator- (const Matrix<T> & matrix2){
    /**
     * function operator-
     * Subtracts two matrices.
     *
     * Overrides the - operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *   const Matrix<T> matrix2: A second matrix to subtract from the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>: The difference between the two matrices.
     *
     * Errors:
     *   Throws an error if the matrices are different shapes.
     */

    // Data validation.
    try {
        // Check that the matrices are the same shape.
        if (!(rows == matrix2.rows && cols == matrix2.cols)) {
            // Throw an error.
            throw 1;
        }
    }
    catch (int e) {
        // Print an error statement.
        std::cout << "Matrix subtraction: matrices different shapes." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the matrix.
    for (unsigned i = 0; i < rows; ++i){
        // Loop through the items in each row.
        for (unsigned j = 0; j < cols; ++j){
            // Find the difference between the values of the matrices at the current index.
            output[i][j] = matrix[i][j] - matrix2[i][j];
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-= (const Matrix<T> & matrix2) {
    /**
     * function operator-=
     * Uses Matrix<T>::operator- to add to a matrix.
     *
     * Overrides the -= operator for a Matrix object.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix to subtract from the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
     */

     // Use the overridden matrix addition operator to subtract the matrices.
     // The 'this' keyword is a pointer that must be unpacked before subtraction.
    *this = *this - matrix2;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator* (const Matrix<T> & matrix2) {
    /**
     * function operator *
     * Multiplies two matrices.
     *
     * Overrides the * operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix of the same type T to multiply with the first.
     *
     * Returns:
     *   Matrix<T>: The product of the two matrices.
     *
     * Errors:
     *   Throws an error if the sizes are such that multiplication cannot occur.
    */

    // Data validation.
    try {
        // Check that the number of rows in matrix2 is the same as the number of
        // columns in the original matrix.
        if (cols != matrix2.rows) {
            throw 1;
        }
    }
    catch (int e) {
        // Print an error statenent.
        std::cout << "Matrix multiplication: matrices inconsistent sizes." << std::endl;
        // Exit the program and report the failure.
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(matrix2.cols, rows);

    // Loop through the rows of the output matrix.
    for (unsigned i = 0; i < output.rows; ++i){
      // Loop through each item in the row.
        for (unsigned j = 0; j < output.cols; ++j){
            // Find the value of the output matrix using the definition of
            // matrix multiplication.
            T sum = 0;
            for (unsigned k = 0; k < cols; ++k){
                sum += matrix[i][k] * matrix2[k][j];
            }
            output[i][j] = sum;
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*= (const Matrix<T> & matrix2) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator* to multiply and update a matrix.
     *
     * Overrides the *= operator for a matrix object.
     *
     * Parameters:
     *   Matrix<T> matrix2: A second matrix to multiply with the first, of the same type T.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
    */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this * matrix2;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator* (T value) {
    /**
     * function operator *
     * Multiplies a matrix by some scalar of type T.
     *
     * Overrides the * operator for a Matrix objects of type T and a literal of the same type T.
     *
     * Parameters:
     *   T value: A value of type T to multiply with the matrix.
     *
     * Returns:
     *   Matrix<T>: The scalar product of the matrix and the scalar.
    */

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the output matrix.
    for (unsigned i = 0; i < output.rows; ++i) {
        // Loop through each item in the row.
        for (unsigned j = 0; j < output.cols; ++j) {
            // Multiply the matrix at the current index by the input value.
            output[i][j] = matrix[i][j] * value;
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*= (T value) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator* to multiply and update a matrix.
     *
     * Overrides the *= operator for a matrix object for scalar multiplication.
     *
     * Parameters:
     *   T value: A value of type T to multiply with the matrix.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
    */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this * value;

    // Return the matrix.
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator/ (T value) {
    /**
     * function operator /
     * Divides a matrix by some scalar of type T.
     *
     * Overrides the / operator for a Matrix objects of type T and a literal of the same type T.
     *
     * Parameters:
     *   T value: A value of type T by which to divide the matrix.
     *
     * Returns:
     *   Matrix<T>: The scalar product of the matrix and the reciprocal of the scalar.
    */

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Loop through the rows of the output matrix.
    for (unsigned i = 0; i < output.rows; ++i) {
        // Loop through each item in the row.
        for (unsigned j = 0; j < output.cols; ++j) {
            // Multiply the matrix at the current index by the input value.
            output[i][j] = matrix[i][j] / value;
        }
    }

    // Return the output matrix.
    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/= (T value) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator/ to divide and update a matrix.
     *
     * Overrides the /= operator for a matrix object for scalar division.
     *
     * Parameters:
     *   T value: A value of type T to divide the matrix by.
     *
     * Returns:
     *   Matrix<T>&: A reference to an updated version of the matrix.
    */

    // Use the overridden matrix addition operator to add the matrices.
    // The 'this' keyword is a pointer that must be unpacked before addition.
    *this = *this / value;

    // Return the matrix.
    return *this;
}

// The rest of this files involves LU decomposition functions.

// These functions are designed to be used with any user input, and therefore
// have data validation to ensure, for example, that the input matrices have
// the correct dimensions.

namespace matrix_LU {

    template <typename T>
    struct LU_output{
        /**
         * struct LU_output
         * Stores the output from LU decomposition.
         *
         * Using a structure allows functions in C++ to return more than one object.
         *
         * Members:
         *   matrix_L: The lower triangular matrix from LU decomposition.
         *   matrix_U: The upper triangular matrix from LU decomposition.
         */
        Matrix<T> matrix_L;
        Matrix<T> matrix_U;
    };

    template <typename T>
    LU_output<T> crout (const Matrix<T> & matrix_A) {
        /**
         * function crout
         * Uses Crout's method to carry out LU decomposition of an N*N matrix.
         *
         * This algorithm won't work as expected for a matrix of ints due to integer division.
         *
         * Parameters:
         *   Matrix<T> matrix_A: An N*N matrix to decompose. Works better if is type Matrix<double> or Matrix<float>.
         *
         * Returns:
         *   Matrix<T>: An N*N matrix containing all the elements of U and the non-diagonal elements of L.
         *
         * Errors:
         *   Throws an error if the matrix is not an N*N matrix.
         */

        unsigned rows = matrix_A.get_rows();
        unsigned cols = matrix_A.get_cols();

        // Data validation.
        try {
            // Check that matrix_A is square.
            if (rows != cols) {
                // Throw an error.
                throw 1;
            }
        } catch (int e) {
            // Print an error statenent.
            std::cout << "crout: The input matrix must be square." << std::endl;
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise the L and U matrices.
        Matrix<T> matrix_U(rows, rows);
        Matrix<T> matrix_L(rows, rows);

        // Set the diagonal of L to 1.
        for (unsigned i = 0; i < rows; ++i) {
            matrix_L[i][i] = 1;
        }

        // Uses the algorithm from Computational Physics, Lecture 3 - Matrix
        // Methods, Slide 25 [P. Scott, 2017]

        // Iterate over each column in the matrix.
        for (unsigned j = 0; j < cols; ++j) {
            // Starting at the top, iterate over each entry in the column until
            // reaching the diagonal.
            for (unsigned i = 0; i <= j; ++i) {
                T total = 0;
                for (unsigned k = 0; k < i; ++k) {
                    total += matrix_L[i][k] * matrix_U[k][j];
                }
                // Set the U matrix.
                matrix_U[i][j] = matrix_A[i][j] - total;
            }
            // Starting at the diagonal, iterate over each entry in the column until
            // reaching the bottom.
            for (unsigned i = j; i < rows; ++i) {
                T total = 0;
                for (unsigned k = 0; k < j; ++k) {
                    total += matrix_L[i][k] * matrix_U[k][j];
                }
                // Set the L matrix.
                matrix_L[i][j] = (matrix_A[i][j] - total) / matrix_U[j][j];
            }
        }

        // Initialise the output structure.
        LU_output<T> output;

        // Set the values in the output structure.
        output.matrix_L = matrix_L;
        output.matrix_U = matrix_U;

        // Return the structure.
        return output;
    }

    template <typename T>
    Matrix<T> combine_LU_output (const LU_output<T> & LU_structure) {
        /**
         * function combine_LU_output
         * Takes the output of LU decomposition and combines into a single matrix containing all the elements of U, and the
         * non-diagonal elements of L.
         *
         * Parameters:
         *   LU_output LU_structure<T>: The output structure from and LU decomposition algorithm.
         *
         * Returns:
         *   Matrix<T>: The combined matrix.
         *
         * Errors:
         *   Throws an error if LU_structure.matrix_L and output.matrix_U are different sizes.
         *   Throws an error if LU_structure.matrix_L is not square.
         */

        unsigned rows = LU_structure.matrix_L.get_rows();
        unsigned cols = LU_structure.matrix_L.get_cols();

        // Data validation.
        try {
            // Check that the matrices are the same size.
            if ((LU_structure.matrix_U.get_rows() != rows) && (LU_structure.matrix_U.get_cols() != cols)) {
                throw 1;
            }
            // Check the matrices are square.
            if (rows != cols) {
                throw 2;
            }
        } catch (int e) {
            // Print the relevant error statenent.
            if (e == 1) {
                std::cout << "combine_LU_output: The input matrices must be the same size." << std::endl;
            }
            if (e == 2) {
                std::cout << "combine_LU_output: The matrices must be square." << std::endl;
            }
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise output matrix to contain the same entries as the U matrix.
        Matrix<T> output = LU_structure.matrix_U;

        // Iterate over each row in the matrix.
        for (unsigned i = 0; i < rows; ++i) {
            // Starting at the edge, iterate over each entry in the row until
            // reaching the diagonal.
            for (unsigned j = 0; j < i; ++j) {
                // Copy the entry of the L matrix at the current index into the
                // output matrix.
                output[i][j] = LU_structure.matrix_L[i][j];
            }
        }

        // Return the combined matrix.
        return output;

    }

    template <typename T>
    bool check_lower_triangular(const Matrix<T> & matrix) {
        /**
         * function check_lower_triangular
         * Checks if a matrix is lower triangular.
         *
         * Parameters:
         *   Matrix<T> matrix: A Matrix object.
         *
         * Returns:
         *   bool: True iff matrix is lower triangular.
         */

        // Assume the matrix is lower triangular.
        bool lower_triangular = true;

        // Loop through the rows of the matrix.
        for (unsigned i = 0; i < matrix.get_rows(); ++i) {
            // Loop through items in the current row to the right of the diagonal.
            for (unsigned j = i + 1; j < matrix.get_cols(); ++j) {
                // If any of the matrix elements are non-zero, the matrix is not
                // lower triangular.
                if (matrix[i][j] != 0) {
                    lower_triangular = false;
                }
            }
        }

        // Return the boolean.
        return lower_triangular;
    }

    template <typename T>
    bool check_upper_triangular(const Matrix<T> & matrix) {
        /**
         * function check_upper_triangular
         * Checks if a matrix is upper triangular.
         *
         * Parameters:
         *   Matrix<T> matrix: A Matrix object.
         *
         * Returns:
         *   bool: True iff matrix is upper triangular.
         */

        // Assume the matrix is lower triangular.
        bool upper_triangular = true;

        // Loop through the rows of the matrix.
        for (unsigned i = 0; i < matrix.get_rows(); ++i) {
            // Loop through items in the current row to the right of the diagonal.
            for (unsigned j = 0; j < i; ++j) {
              // If any of the matrix elements are non-zero, the matrix is not
              // upper triangular.
                if (matrix[i][j] != 0) {
                    upper_triangular = false;
                }
            }
        }

        // Return the boolean.
        return upper_triangular;
    }

    template <typename T>
    bool check_triangular(Matrix<T> matrix) {
        /**
         * function check_triangular
         * Checks if a matrix is triangular.
         *
         * Parameters:
         *   Matrix<T> matrix: A Matrix object.
         *
         * Returns:
         *   bool: True iff matrix is triangular.
         */

        unsigned rows = matrix.get_rows();
        unsigned cols = matrix.get_cols();

        // Check that the matrix is square.
        if (cols != rows) {
            return false;
        }

        // Check that the matrix is triangular.
        bool lower_triangular = check_lower_triangular(matrix);
        bool upper_triangular = check_upper_triangular(matrix);

        // Return true if the matrix is either lower or upper triangular.
        return (lower_triangular || upper_triangular);
    }

    template <typename T>
    T triangular_determinant(const Matrix<T> & triangular){
        /**
         * function triangular_determinant
         * Calculates the determinant of a triangular matrix.
         *
         * Uses the result that the determinant of a triangular matrix is the product of the diagonal entries.
         *
         * Parameters:
         *   Matrix<T> triangular: A triangular matrix with entries of type T.
         *
         * Returns:
         *   T determinant: The determinant of the matrix. Returned as the same type T as the matrix entries.
         *
         * Errors:
         *   Throws an error if the matrix is not triangular.
         */

        unsigned rows = triangular.get_rows();
        unsigned cols = triangular.get_cols();

        // Data validation.
        try {
            // Check that the matrix is square.
            if (cols != rows) {
                throw 1;
            }
            // Check that the matrix is triangular.
            if (!check_triangular(triangular)) {
                throw 2;
            }
        } catch (int e) {
            // Print the relevant error statenent.
            if (e == 1) {
                std::cout << "triangular_determinant: The input matrix is not square, so cannot be triangular." << std::endl;
            }
            if (e == 2) {
                std::cout << "triangular_determinant: The input matrix must be triangular." << std::endl;
            }
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise the determinant to 1.
        T determinant = 1;

        // Loop through the rows of the matrix.
        for (unsigned i = 0; i < rows; ++i) {
            // Multiply the determinant by the element of the row on the diagonal.
            determinant *= triangular[i][i];
        }

        // Return the value of the determinant.
        return determinant;
    }

    template <typename T>
    Matrix<T> linsolve_LU(const Matrix<T> & matrix_L, const Matrix<T> & matrix_U, const Matrix<T> & vector_b){
        /**
         * function linsolve_LU
         * Solves the equation Ax = b using the LU decomposition of matrix A.
         *
         * Parameters:
         *   Matrix<T> matrix_L: A lower triangular matrix with entries of type T.
         *   Matrix<T> matrix_U: An upper triangular matrix with entries of the same type T.
         *   Matrix<T> vector_B: A matrix with only one column, with entries of the same type T.
         *
         * Returns:
         *   Matrix<T>: A matrix x of type T that solves the equqtion LUx = b.
         *
         * Errors:
         *   Throws an error if the matrices or vector are inconsistent sizes.
         *   Throws an error if the matrices are not upper or lower triangular.
         */

        unsigned rows = matrix_L.get_rows();
        unsigned cols = matrix_L.get_cols();

        // Data validation.
        try {
            // Check that the matrices are the same size.
            if ((matrix_U.get_rows() != rows) && (matrix_U.get_cols() != cols)) {
                throw 1;
            }
            // Check the matrices are square.
            if (rows != cols) {
                throw 2;
            }
            // Check that matrix_L is lower triangular.
            if (!check_lower_triangular(matrix_L)) {
                throw 3;
            }
            // Check that matrix_U is upper triangular.
            if (!check_upper_triangular(matrix_U)) {
                throw 4;
            }
            // Check that vector_b has only one column.
            if (vector_b.get_cols() != 1) {
                throw 5;
            }
            // Check that vector_b has the correct number of rows.
            if (vector_b.get_rows() != cols) {
                throw 6;
            }
        } catch (int e) {
            // Print the relevant error statenent.
            switch(e) {
                case 1: std::cout << "linsolve_LU: matrix_L and matrix_U must be the same size." << std::endl;
                        break;
                case 2: std::cout << "linsolve_LU: matrix_L and matrix_U must be square." << std::endl;
                        break;
                case 3: std::cout << "linsolve_LU: matrix_L must be lower triangular." << std::endl;
                        break;
                case 4: std::cout << "linsolve_LU: matrix_U must be upper triangular" << std::endl;
                        break;
                case 5: std::cout << "linsolve_LU: vector_b must be a have only 1 column." << std::endl;
                        break;
                case 6: std::cout << "linsolve_LU: vector_b must have the same number of rows as matrix_L and matrix_U." << std::endl;
                        break;
            }
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise a vector y, to solve Ly = b.
        // A std::vector is used because vector_y cannot be accessed outside this
        // function.
        std::vector<T> vector_y(rows);

        // Calculate y_0.
        vector_y[0] = vector_b[0][0]/matrix_L[0][0];

        // Loop through the other entries in y and calculate their values.
        for (unsigned i = 1; i < rows; ++i) {
            // Implement the forward substitution calculation from Computational
            // Physics, Lecture 3 - Matrix Methods, Slide 23 [P. Scott, 2017]
            T total = 0;
            for (unsigned j = 0; j < i; ++j) {
                total += matrix_L[i][j]* vector_y[j];
            }
            // Store the value to vector_y.
            vector_y[i] = (vector_b[i][0] - total)/matrix_L[i][i];
        }

        // Initialise the vector x, to solve LUx = b.
        Matrix<T> vector_x(1, rows);

        // Calculate the last entry in vector_x.
        vector_x[rows - 1][0] = vector_y[rows - 1] / matrix_U[rows - 1][rows - 1];

        // Loop through the other entries in x and calculate their values.
        // Using signed ints despite comparison with a std::vector size (rows) since
        // decrementing unsigned ints can cause overflow errors.
        for (int i = rows - 2; i >= 0; --i){
            // Implement the backward substitution algorithm from Computational
            // Physics, Lecture 3 - Matrix Methods, Slide 23 [P. Scott, 2017]
            T total = 0;
            for (int j = i + 1; j < rows; ++j) {
                total += matrix_U[i][j] * vector_x[j][0];
            }
            // Store the value to vector_x.
            vector_x[i][0] = (vector_y[i] - total)/matrix_U[i][i];
        }

        return vector_x;
    }


    template<typename T>
    Matrix<T> find_inverse (const Matrix<T> & matrix_L, const Matrix<T> & matrix_U) {
        /**
         * function find_inverse
         * Find the inverse of a matrix from its LU decomposition.
         *
         * Parameters:
         *   Matrix<T> matrix_L: A lower triangular matrix with entries of type T.
         *   Matrix<T> matrix_U: An upper triangular matrix with entries of the same type T.
         *
         * Returns:
         *   Matrix<T>: The inverse of the matrix given by L*U.
         */

        unsigned rows = matrix_L.get_rows();
        unsigned cols = matrix_L.get_cols();

        // Data validation.
        try {
            // Check that the matrices are the same size.
            if ((matrix_U.get_rows() != rows) && (matrix_U.get_cols() != cols)) {
                throw 1;
            }
            // Check the matrices are square.
            if (rows != cols) {
                throw 2;
            }
            // Check that matrix_L is lower triangular.
            if (!check_lower_triangular(matrix_L)) {
                throw 3;
            }
            // Check that matrix_U is upper triangular.
            if (!check_upper_triangular(matrix_U)) {
                throw 4;
            }
        } catch (int e) {
            // Print the relevant error statenent.
            switch(e) {
                case 1: std::cout << "linsolve_LU: matrix_L and matrix_U must be the same size." << std::endl;
                        break;
                case 2: std::cout << "linsolve_LU: matrix_L and matrix_U must be square." << std::endl;
                        break;
                case 3: std::cout << "linsolve_LU: matrix_L must be lower triangular." << std::endl;
                        break;
                case 4: std::cout << "linsolve_LU: matrix_U must be upper triangular" << std::endl;
                        break;
            }
            // Exit the program and report the failure.
            exit(EXIT_FAILURE);
        }

        // Initialise the inverse matrix.
        Matrix<T> inverse(rows, rows);

        // Loop through the rows of the inverse matrix.
        for (unsigned i = 0; i < rows; ++i) {

            // Find the ith cartesian basis vector.
            Matrix<T> unit_vector(1, rows);
            unit_vector[i][0] = 1;
            // Use the basis vector to solve for a column of the inverse matrix.
            Matrix<T> vector_x = linsolve_LU(matrix_L, matrix_U, unit_vector);
            // Loop through the solution vector.
            for (unsigned j = 0; j < rows; ++j) {
                // Store the values from the solution to the inverse matrix.
                inverse[j][i] = vector_x[j][0];
           }
        }

        return inverse;

    }
}

#endif //MATRIX_LU_H
