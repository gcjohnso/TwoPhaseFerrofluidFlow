/*
 * Mesh.cc
 *
 *  Created on: Dec 1, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "../include/Mesh.h"

/**
 * See description in header file.
 */
template <int dim>
void make_subdivided_rectangle(Triangulation<dim> &tria, std::vector<unsigned int> &repetitions, Point<dim> lowerInner, Point<dim> upperOuter){
	GridGenerator::subdivided_hyper_rectangle(tria, repetitions, lowerInner, upperOuter, false);
}

/**
 * See description in header file.
 */
template <int dim>
void output_mesh(Triangulation<dim> &tria, const char filename[]){
	std::ofstream out(filename);
	GridOut grid_out;
	grid_out.write_eps(tria, out);
}

/**
 * Unit test for mesh generation and printing.
 * @return 0
 */
int main(){
	//Make the grid for experiments 1:
	//Coordinates: (0,0), (0,0.6), (1,0.6), (1,0)
	//10 elements in x-dir, 6 elements in y-dir
	Point<2> lowerInner1(0.0, 0.0);
	Point<2> upperOuter1(1.0, 0.6);

	std::vector<unsigned int> repetitions1;
	repetitions1.push_back(10);
	repetitions1.push_back(6);

	Triangulation<2> triangulation1;

	//Create and then output the mesh for verification.
	make_subdivided_rectangle(triangulation1, repetitions1, lowerInner1, upperOuter1);
	output_mesh(triangulation1, "rectangle_grid1.eps");

	//Make the grid for experiments 1:
	//Coordinates: (0,0), (0,0.6), (1,0.6), (1,0)
	//15 elements in x-dir, 9 elements in y-dir
	Point<2> lowerInner2(0.0, 0.0);
	Point<2> upperOuter2(1.0, 0.6);

	std::vector<unsigned int> repetitions2;
	repetitions2.push_back(15);
	repetitions2.push_back(9);

	Triangulation<2> triangulation2;

	//Create and then output the mesh for verification.
	make_subdivided_rectangle(triangulation2, repetitions2, lowerInner2, upperOuter2);
	output_mesh(triangulation2, "rectangle_grid2.eps");

	return 0;
}
