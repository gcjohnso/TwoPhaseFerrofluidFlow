/*
 * AppliedMagnetizingField.h
 *
 * This file contains a class representing the Applied Magnetizing Field, h_a, which for this project is
 * defined as the weighted sum of gradients of the potential of a point dipole:
 *
 * theta_s(x) = d * (x_s - x) / |x_s-x|^n
 *
 * where x_s is the location of the dipole, d is the orientation of the dipole, and n is the dimension.
 * The weights correspond to an intensity, which currently is the same for each dipole. The intensities
 * can increase linearly from an initial value to a final intensity over a given time range (outside of
 * the specified time range the intensity will remain constant at the final intensity).
 *
 *  Created on: Nov 28, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#ifndef INCLUDE_APPLIEDMAGNETIZINGFIELD_H_
#define INCLUDE_APPLIEDMAGNETIZINGFIELD_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace dealii;

/**
 * Class encapsulating the Applied Magnetic Field h_a.
 */
template <int dim>
class AppliedMagnetizingField{
	public:
		AppliedMagnetizingField(const char filename[]);
		void value(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values, double time);

	private:
		void parseFile(const char filename[]);

		std::vector<Point<dim>> dipoleLocs;
		std::vector<Point<dim>> dipoleDirs;
		double startIntensity;
		double endIntensity;
		double startTime;
		double endTime;
};

#endif /* INCLUDE_APPLIEDMAGNETIZINGFIELD_H_ */
