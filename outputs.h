#ifndef OUTPUTS_H
#define OUTPUTS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

template <typename T>
void vector_to_file (std::ofstream &stream, std::vector<T> vector) {
    /** function vector_to_file
     * Outputs a vector to a file, followed by a '\n' character.
     *
     * Parameters:
     *  std::ofstream stream: The stream to which the vector should be output.
     *  std::vector vector: The vector to output.
     */

    for (int i = 0; i < vector.size(); ++i){
        stream << vector[i];
        if (i != vector.size() - 1){
            stream << ',';
        }
    }

    stream << '\n';
}

std::vector<double> linspace(double start, double stop, int number){
  /**
  * function linspace
  *
  * Generates a linearlly spaced vector.
  *
  * Parameters:
  *   double start: A value at which to start the array.
  *   double stop: A value at which to stop the array.
  *   int number: The number of points in the vector
  *
  * Returns:
  *   std::vector<double>: A vector containing the requested points.
  */

  double spacing = (stop - start)/(number - 1);
  double value = start;

  std::vector<double> output(number);

  for (int i = 0; i < number; ++i) {
    output[i] = value;
    value += spacing;
  }

  return output;
}

#endif //OUTPUTS_H
