#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>

template <typename T> class Matrix {
    /**
     * class Matrix
     * A class that implements matrix operations.
     *
     * The class is designed to work with any type T. To initialise the class, use Matrix <T>, e.g. Matrix <double>.
     *
     * Properties:
     *  Private:
     *      rows: The number of rows in the matrix.
     *      cols: The number of columns in the matrix.
     *      matrix: A vector of vectors storing the data.
     *
     * Methods:
     *  Public:
     *      Matrix: Class constructor.
     *      get_rows: Returns the number of rows in the matrix.
     *      get_cols: Returns the number of columns in the matrix
     *      print: Displays the matrix in the console.
     *      operator[]: Returns a row of the matrix given an index.
     *      Matrix<T> operator+ (Matrix<T> matrix2);
     *      Matrix<T>& operator+= (Matrix<T> matrix2);
     *      Matrix<T> operator- (Matrix<T> matrix2);
     *      Matrix<T>& operator-= (Matrix<T> matrix2);
     *      Matrix<T> operator* (Matrix<T> matrix2);
     *      Matrix<T>& operator*= (Matrix<T> matrix2);
     *      Matrix<T> operator* (T value);
     *      Matrix<T>& operator*= (T value);
     *      Matrix<T> operator/ (T value);
     *      Matrix<T>& operator/= (T value);
     */
private:
    unsigned int rows; /** The number of rows in the matrix. */
    unsigned int cols; /** The number of columns in the matrix. */
    std::vector <std::vector <T> > matrix; /** A vector of vectors storing the data */
public:

    explicit Matrix (std::vector<std::vector<T>> input_vector) {
        /**
         * function Matrix
         * Matrix class constructor with only 1 input.
         *
         * Paramters:
         *  std::vector<std::vector<T>> input_vector: A vector of vectors of type T used as the data for the matrix.
         *
         * Returns:
         *  Matrix <T>: A Matrix object containing the data.
         *
         * Errors:
         *  Throws an error if any of the vectors in input_vector is a different size to the first vector.
         */

        // Data validation.
        unsigned int column_no = input_vector[0].size();
        try {
            for (const std::vector<T> & row : input_vector){
                if (row.size() != column_no) {
                    throw 24;
                }
            }
        }
        catch (int e) {
            std::cout << "Matrix constructor: Inconsistent number of columns. Error number " << e << "." << std::endl;
            exit(EXIT_FAILURE);
        }


        // Initialise the class properties.
        rows = input_vector.size();
        cols = column_no;
        matrix = input_vector;
    }

    Matrix (const unsigned int x_size, const unsigned int y_size) {
        /**
         * function Matrix
         * Matrix constructor for an empty matrix.
         *
         * Parameters:
         *  const unsigned int x_size: The number of columns of the new matrix.
         *  const unsigned int y_size: The number of rows of the new matrix.
         *
         * Returns:
         *  Matrix <T>: An empty matrix with the desired number of rows and columsn
         */
        matrix = std::vector<std::vector<T>> (y_size, std::vector<T>(x_size));
        rows = y_size;
        cols = x_size;
    }

    Matrix () {
        /**
         * function Matrix
         * Matrix constructor for an empty matrix of undetermined size.
         *
         * Returns:
         *  Matrix<T>: A empty matrix with no size.
         */
        matrix = std::vector<std::vector<T>> (0, std::vector<T>(0));
        rows = 0;
        cols = 0;
    };

    std::vector<T>& operator[] (const int index) {
        /** function operator[]
         * Returns a row of the matrix given an index.
         *
         * Overrides the [] operator so that the matrix vector of vectors does not need to be accessed directly. A
         * second [] operator then be used to access the desired matrix entry. This can be used to change values in the
         * matrix.
         *
         * Parameters:
         *  const int index: The desired row index.
         *
         * Returns:
         *  std::vector<T>&: A reference to the desired vector.
         */
        return matrix[index];
    }

    const std::vector<T>& operator[] (const int index) const {
        /** function operator[]
         * Returns a row of a constant matrix given an index.
         *
         * Overrides the [] operator for const Matrix objects so that the matrix vector of vectors does not need to be
         * accessed directly. A second [] operator then be used to access the desired matrix entry. This CANNOT be used
         * to change values in the matrix.
         *
         * Parameters:
         *  const int index: The desired row index.
         *
         * Returns:
         *  const std::vector<T>&: A reference to the desired vector.
         */
        return matrix[index];
    }

    unsigned int get_rows(){
        /**
         * function get_rows
         * Returns the number of rows in the matrix.
         *
         * Returns:
         *  unsigned int: The value of matrix.rows.
         */
        return rows;
    }

    unsigned int get_cols(){
        /**
         * function get_cols
         * Returns the number of columns in the matrix.
         *
         * Returns:
         *  unsigned int: The value of matrix.cols.
         */

        return cols;
    }

    // Declaring functions that are defined later.

    void print();
    Matrix<T> operator+ (Matrix<T> matrix2);
    Matrix<T>& operator+= (Matrix<T> matrix2);
    Matrix<T> operator- (Matrix<T> matrix2);
    Matrix<T>& operator-= (Matrix<T> matrix2);
    Matrix<T> operator* (Matrix<T> matrix2);
    Matrix<T>& operator*= (Matrix<T> matrix2);
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
    std::cout << '[' << '\n';

    for (unsigned int i = 0; i < rows; ++i){
        std::cout << '[';
        for (unsigned int j = 0; j < cols; ++j){
            std::cout << ' ' << matrix[i][j] << ' ';
        }
        std::cout << ']' << '\n';
    }

    std::cout << ']' << std::endl;
}


template <typename T>
Matrix<T> Matrix<T>::operator+ (const Matrix<T> matrix2){
    /**
     * function operator+
     * Adds two matrices.
     *
     * Overrides the + operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *  const Matrix<T> matrix2: A second matrix to add to the first, of the same type T.
     *
     * Returns:
     *  Matrix<T>: The sum of the two matrices.
     *
     * Errors:
     *  Throws an error if the matrices are different shapes.
     */

    // Data validation.
    try {
        if (!(rows == matrix2.rows && cols == matrix2.cols)) {
            throw 24;
        }
    }
    catch (int e) {
        std::cout << "Matrix addition: matrices different shapes. Error number " << e << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Find the sum at each point.
    for (unsigned int i = 0; i < rows; ++i){
        for (unsigned int j = 0; j < cols; ++j){
            output[i][j] = matrix[i][j] + matrix2[i][j];
        }
    }

    return output;

}

template <typename T>
Matrix<T>& Matrix<T>::operator+= (Matrix<T> matrix2) {
    /**
     * function operator+=
     * Uses Matrix<T>::operator+ to add to a matrix.
     *
     * Overrides the += operator for a Matrix object.
     *
     * Parameters:
     *  Matrix<T> matrix2: A second matrix to add to the first, of the same type T.
     *
     * Returns:
     *  Matrix<T>&: A reference to an updated version of the matrix.
     */
    *this = *this + matrix2;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator- (const Matrix<T> matrix2){
    /**
     * function operator-
     * Subtracts two matrices.
     *
     * Overrides the - operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *  const Matrix<T> matrix2: A second matrix to subtract from the first, of the same type T.
     *
     * Returns:
     *  Matrix<T>: The difference between the two matrices.
     *
     * Errors:
     *  Throws an error if the matrices are different shapes.
     */

    // Data validation.
    try {
        if (!(rows == matrix2.rows && cols == matrix2.cols)) {
            throw 24;
        }
    }
    catch (int e) {
        std::cout << "Matrix addition: matrices different shapes. Error number " << e << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(cols, rows);

    // Find the sum at each point.
    for (unsigned int i = 0; i < rows; ++i){
        for (unsigned int j = 0; j < cols; ++j){
            output[i][j] = matrix[i][j] - matrix2[i][j];
        }
    }

    return output;

}

template <typename T>
Matrix<T>& Matrix<T>::operator-= (Matrix<T> matrix2) {
    /**
     * function operator-=
     * Uses Matrix<T>::operator- to add to a matrix.
     *
     * Overrides the -= operator for a Matrix object.
     *
     * Parameters:
     *  Matrix<T> matrix2: A second matrix to subtract from the first, of the same type T.
     *
     * Returns:
     *  Matrix<T>&: A reference to an updated version of the matrix.
     */
    *this = *this - matrix2;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator* (Matrix<T> matrix2) {
    /**
     * function operator *
     * Multiplies two matrices.
     *
     * Overrides the * operator for two Matrix objects of the same type T.
     *
     * Parameters:
     *  Matrix<T> matrix2: A second matrix of the same type T to multiply with the first.
     *
     * Returns:
     *  Matrix<T>: The product of the two matrices.
     *
     * Errors:
     *  Throws an error if the sizes are such that multiplication cannot occur.
    */

    // Data validation.
    try {
        if (cols != matrix2.rows) {
            throw 24;
        }
    }
    catch (int e) {
        std::cout << "Matrix multiplication: matrices inconsistent sizes. Error number " << e << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create output matrix.
    Matrix<T> output(matrix2.cols, rows);


    // Find the value at each point.
    for (unsigned int i = 0; i < output.rows; ++i){
        for (unsigned int j = 0; j < output.cols; ++j){
            T sum = 0;
            for (unsigned int k = 0; k < cols; ++k){
                sum += matrix[i][k] * matrix2[k][j];
            }
            output[i][j] = sum;
        }
    }

    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*= (Matrix<T> matrix2) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator* to multiply and update a matrix.
     *
     * Overrides the *= operator for a matrix object.
     *
     * Parameters:
     *  Matrix<T> matrix2: A second matrix to multiply with the first, of the same type T.
     *
     * Returns:
     *  Matrix<T>&: A reference to an updated version of the matrix.
    */
    *this = *this * matrix2;
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
     *  T value: A value of type T to multiply with the matrix.
     *
     * Returns:
     *  Matrix<T>: The scalar product of the matrix and the scalar.
    */

    Matrix<T> output(cols, rows);

    for (unsigned int i = 0; i < output.rows; ++i) {
        for (unsigned int j = 0; j < output.cols; ++j) {
            output[i][j] *= value;
        }
    }

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
     *  T value: A value of type T to multiply with the matrix.
     *
     * Returns:
     *  Matrix<T>&: A reference to an updated version of the matrix.
    */

    *this = *this * value;
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
     *  T value: A value of type T by which to divide the matrix.
     *
     * Returns:
     *  Matrix<T>: The scalar product of the matrix and the reciprocal of scalar.
    */

    Matrix<T> output(cols, rows);

    for (unsigned int i = 0; i < output.rows; ++i) {
        for (unsigned int j = 0; j < output.cols; ++j) {
            output[i][j] /= value;
        }
    }

    return output;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/= (T value) {
    /**
     * function operator*=
     * Uses Matrix<T>::operator/* to divide and update a matrix.
     *
     * Overrides the /= operator for a matrix object for scalar division.
     *
     * Parameters:
     *  T value: A value of type T to divide the matrix by.
     *
     * Returns:
     *  Matrix<T>&: A reference to an updated version of the matrix.
    */

    *this = *this / value;
    return *this;
}

#endif //MATRIX_H
