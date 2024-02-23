#ifndef UTILS_H
#define UTILS_H

// Define UTILS_H at compile time to ensure the module is not loaded twice.
// Only load the module if UTILS_H has not previously been defined.

// Functions outside are wrapped in the namespace utils to prevent conflicts
// when included in another file

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

namespace utils {
    // Using a template ensures the function can be used with a vector of any type.
    // Passes the vector by const reference to reduce memory use and save copying time.
    template <typename T>
    void vector_to_file (std::ofstream &stream, const std::vector<T> & vector) {
        /**
         * function vector_to_file
         * Outputs a comma separated vector to a file, followed by a '\n' character.
         *
         * Parameters:
         *   std::ofstream stream: The stream to which the vector should be output.
         *   std::vector vector: The vector to output.
         */

        // Loop through the vector.
        for (unsigned i = 0; i < vector.size(); ++i){
            // Output the vector to the stream.
            // Ensure the precision of any float output is high so that a valid
            // comparison with scipy can be made.
            stream << std::setprecision(19) << vector[i];
            if (i != vector.size() - 1){
                // Output a comma after each element.
                stream << ',';
            }
        }
        // Output a newline.
        stream << '\n';
    }

    std::vector<double> linspace(double start, double stop, int number){
        /**
         * function linspace
         * Generates a linearly spaced vector of doubles for each velue.
         *
         * Parameters:
         *   double start: A value at which to start the array.
         *   double stop: A value at which to stop the array.
         *   int number: The number of points in the vector
         *
         * Returns:
         *   std::vector<double>: A vector containing the requested points.
         */

        // Calculate the spacing between elements.
        double spacing = (stop - start)/(number - 1);

        // Initialise a counter of values to input.
        double value = start;

        // Initialise a vector of doubles.
        std::vector<double> output(number);

        // Loop through the vector.
        for (int i = 0; i < number; ++i) {
            output[i] = value;
            // Calculate the next value.
            value += spacing;
        }

        // Return the vector.
        return output;
    }

}

#endif //UTILS_H
