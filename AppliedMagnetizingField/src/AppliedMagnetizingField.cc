/*
 * AppliedMagnetizingField.cc
 *
 *  Created on: Nov 28, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "../include/AppliedMagnetizingField.h"

/**
 * Default constructor.
 * @param filename The relative filename path for the dipole configuration file.
 */
template <int dim>
AppliedMagnetizingField<dim>::AppliedMagnetizingField(const char filename[]){
	parseFile(filename);
}

/**
 * Computes the value of the applied magnetizing field at given points in the triangulation.
 * @param points Points in the triangulation.
 * @param values Location where the value of the applied magnetizing field will be stored.
 * @param time The current time.
 */
template <int dim>
void AppliedMagnetizingField<dim>::value(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values, double time){
	Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));

	double currentIntensity;
	if(time > endTime){
		currentIntensity = endIntensity;
	}else if(time < startTime){
		currentIntensity = 0;
	}else{
		currentIntensity = (time - startTime)*(endIntensity - startIntensity)/(endTime - startTime);
	}

	if(dim == 2){
		for(unsigned int point_n = 0; point_n < points.size(); ++point_n){
			double grad_x = 0;
			double grad_y = 0;
			for(unsigned int dipole_n = 0; dipole_n < dipoleLocs.size(); ++dipole_n){
				double normSquared = (dipoleLocs[dipole_n] - points[point_n]).norm_square();
				grad_x += -1*dipoleDirs[dipole_n][0]/normSquared;
				grad_x += 2*(dipoleLocs[dipole_n][0] - points[point_n][0])*(dipoleDirs[dipole_n]*(dipoleLocs[dipole_n] - points[point_n]))/pow(normSquared, 2);

				grad_y += -1*dipoleDirs[dipole_n][1]/normSquared;
				grad_y += 2*(dipoleLocs[dipole_n][1] - points[point_n][1])*(dipoleDirs[dipole_n]*(dipoleLocs[dipole_n] - points[point_n]))/pow(normSquared, 2);
			}
			values[point_n][0] = grad_x*currentIntensity;
			values[point_n][1] = grad_y*currentIntensity;
		}
	}else if(dim == 3){
		for(unsigned int point_n = 0; point_n < points.size(); ++point_n){
			double grad_x = 0;
			double grad_y = 0;
			double grad_z = 0;
			for(unsigned int dipole_n = 0; dipole_n < dipoleLocs.size(); ++dipole_n){
				double normCubed = std::pow((dipoleLocs[dipole_n] - points[point_n]).norm_square(), 3./2.);
				grad_x += -1*dipoleDirs[dipole_n][0]/normCubed;
				grad_x += 3*(dipoleLocs[dipole_n][0] - points[point_n][0])*(dipoleDirs[dipole_n]*(dipoleLocs[dipole_n] - points[point_n]))/(normCubed*(dipoleLocs[dipole_n] - points[point_n]).norm_square());

				grad_y += -1*dipoleDirs[dipole_n][1]/normCubed;
				grad_y += 3*(dipoleLocs[dipole_n][1] - points[point_n][1])*(dipoleDirs[dipole_n]*(dipoleLocs[dipole_n] - points[point_n]))/(normCubed*(dipoleLocs[dipole_n] - points[point_n]).norm_square());

				grad_z += -1*dipoleDirs[dipole_n][2]/normCubed;
				grad_z += 3*(dipoleLocs[dipole_n][2] - points[point_n][2])*(dipoleDirs[dipole_n]*(dipoleLocs[dipole_n] - points[point_n]))/(normCubed*(dipoleLocs[dipole_n] - points[point_n]).norm_square());
			}
			values[point_n][0] = grad_x*currentIntensity;
			values[point_n][1] = grad_y*currentIntensity;
			values[point_n][2] = grad_z*currentIntensity;
		}
	}
}

/**
 * Parses the specified file describing the configuration of the dipoles. The first line describes the intensity configuration
 * and the remaining lines describe each dipole location and orientation.
 * @param filename The relative filename path.
 */
template <int dim>
void AppliedMagnetizingField<dim>::parseFile(const char filename[]){
	std::ifstream inputFile;
	inputFile.open(filename);

	std::string currentString;
	std::getline(inputFile, currentString);
	std::istringstream ss1(currentString);
	//First line of the file is startValue, startTime, endValue, endTime
	std::getline(ss1, currentString, ',');
	startIntensity = std::stod(currentString);
	std::getline(ss1, currentString, ',');
	startTime = std::stod(currentString);
	std::getline(ss1, currentString, ',');
	endIntensity = std::stod(currentString);
	std::getline(ss1, currentString, ',');
	endTime = std::stod(currentString);

	//Then process each dipole
	while(std::getline(inputFile, currentString)){
		std::istringstream ss2(currentString);
		//Location
		Point<dim> currentDipole;
		for(unsigned int i = 0; i < dim; ++i){
			std::getline(ss2, currentString, ',');
			currentDipole[i] = std::stod(currentString);
		}
		dipoleLocs.push_back(currentDipole);

		//Direction
		Point<dim> currentDirection;
		for(unsigned int i = 0; i < dim; ++i){
			std::getline(ss2, currentString, ',');
			currentDirection[i] = std::stod(currentString);
		}
		dipoleDirs.push_back(currentDirection);
	}

	inputFile.close();
}

/**
 * Unit tests for the Applied Magnetizing Field.
 * @return 0.
 */
int main(){
	//Single dipole
	const char filename1[] = "../config/testDipoles1.txt";

	AppliedMagnetizingField<2> field1(filename1);

	double tolerance = 0.00001;

	Point<2> point1, point2;
	point1[0] = 1;
	point1[1] = 1;
	point2[0] = 2;
	point2[1] = 1;

	std::vector<Point<2>> points;
	points.push_back(point1);
	points.push_back(point2);

	std::vector<Tensor<1, 2>> values(points.size());

	//Check at t=0 (should be 0)
	field1.value(points, values, 0);
	if(abs(values[0][0] - 0) > tolerance){
		std::cout << "Failing test case1" << std::endl;
		std::cout << "  Comp val: " << values[0][0] << " Actual val: " << 0 << std::endl;
	}
	if(abs(values[0][1] - 0) > tolerance){
		std::cout << "Failing test case2" << std::endl;
		std::cout << "  Comp val: " << values[0][1] << " Actual val: " << 0 << std::endl;
	}
	if(abs(values[1][0] - 0) > tolerance){
			std::cout << "Failing test case3" << std::endl;
			std::cout << "  Comp val: " << values[1][0] << " Actual val: " << 0 << std::endl;
		}
	if(abs(values[1][1] - 0) > tolerance){
		std::cout << "Failing test case4" << std::endl;
		std::cout << "  Comp val: " << values[1][1] << " Actual val: " << 0 << std::endl;
	}

	//Check at t=1
	field1.value(points, values, 1);
	if(abs(values[0][0] - (2*(-2)*(-2)/64.0)*(1-0)*(60/3.5)) > tolerance){
		std::cout << "Failing test case5" << std::endl;
		std::cout << "  Comp val: " << values[0][0] << " Actual val: " << (2*(-2)*(-2)/64.0)*(1-0)*(60/3.5) << std::endl;
	}
	if(abs(values[0][1] - (-1/8. + (2*(-2)*(-2)/64.0))*(1-0)*(60/3.5)) > tolerance){
		std::cout << "Failing test case6" << std::endl;
		std::cout << "  Comp val: " << values[0][1] << " Actual val: " << (-1/8. + (2*(-2)*(-2)/64.0))*(1-0)*(60/3.5) << std::endl;
	}
	if(abs(values[1][0] - (2*(-3)*(-2)/169.0)*(60/3.5)) > tolerance){
			std::cout << "Failing test case7" << std::endl;
			std::cout << "  Comp val: " << values[1][0] << " Actual val: " << (2*(-2)*(-3)/169.0)*(60/3.5) << std::endl;
		}
	if(abs(values[1][1] - (-1/13.0 + 2*(-2)*(-2)/169.0)*(60/3.5)) > tolerance){
		std::cout << "Failing test case8" << std::endl;
		std::cout << "  Comp val: " << values[1][1] << " Actual val: " << (-1/13.0 + 2*(-2)*(-2)/169.0)*(60/3.5) << std::endl;
	}

	//Check at t=5
	field1.value(points, values, 5.2);
	if(abs(values[0][0] - (2*(-2)*(-2)/64.0)*(1-0)*60) > tolerance){
		std::cout << "Failing test case9" << std::endl;
		std::cout << "  Comp val: " << values[0][0] << " Actual val: " << (2*(-2)*(-2)/64.0)*(1-0)*60 << std::endl;
	}
	if(abs(values[0][1] - (-1/8. + (2*(-2)*(-2)/64.0))*(1-0)*60) > tolerance){
		std::cout << "Failing test case10" << std::endl;
		std::cout << "  Comp val: " << values[0][1] << " Actual val: " << (-1/8. + (2*(-2)*(-2)/64.0))*(1-0)*60 << std::endl;
	}
	if(abs(values[1][0] - (2*(-3)*(-2)/169.0)*60) > tolerance){
			std::cout << "Failing test case11" << std::endl;
			std::cout << "  Comp val: " << values[1][0] << " Actual val: " << (2*(-2)*(-3)/169.0)*60 << std::endl;
		}
	if(abs(values[1][1] - (-1/13.0 + 2*(-2)*(-2)/169.0)*60) > tolerance){
		std::cout << "Failing test case12" << std::endl;
		std::cout << "  Comp val: " << values[1][1] << " Actual val: " << (-1/13.0 + 2*(-2)*(-2)/169.0)*60 << std::endl;
	}

	//Double dipole
	const char filename2[] = "../config/testDipoles2.txt";

	AppliedMagnetizingField<2> field2(filename2);

	//Check at t=.999 (should be 0)
	field2.value(points, values, .999);
	if(abs(values[0][0] - 0) > tolerance){
		std::cout << "Failing test case13" << std::endl;
		std::cout << "  Comp val: " << values[0][0] << " Actual val: " << 0 << std::endl;
	}
	if(abs(values[0][1] - 0) > tolerance){
		std::cout << "Failing test case14" << std::endl;
		std::cout << "  Comp val: " << values[0][1] << " Actual val: " << 0 << std::endl;
	}

	//Check at t=2
	field2.value(points, values, 2);
	if(abs(values[0][0] - (2*(-1)*(-1)/4.0 + 0)*(2-1)*60/(3.5-1)) > tolerance){
		std::cout << "Failing test case15" << std::endl;
		std::cout << "  Comp val: " << values[0][0] << " Actual val: " << (2*(-1)*(-1)/4.0 + 0)*(2-1)*60/(3.5-1) << std::endl;
	}
	if(abs(values[0][1] - (-.5 + 2*(-1)*(-1)/4.0- 1)*(2-1)*60/(3.5-1)) > tolerance){
		std::cout << "Failing test case16" << std::endl;
		std::cout << "  Comp val: " << values[0][1] << " Actual val: " << (-.5 + 2*(-1)*(-1)/4.0- 1)*(2-1)*60/(3.5-1) << std::endl;
	}

	return 0;
}


