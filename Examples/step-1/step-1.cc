/*
 * step-1.cc
 *
 *  Created on: Oct 19, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 * This program creates a square and circular mesh, both with
 * 5 levels of refinement. These meshes are exported to the files
 * grid-1.eps and grid-2.eps.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

#define DIM 2

using namespace dealii;

/**
 * Method creates a square grid and then globally refines it a fixed number of times.
 * It then exports the grid to a file.
 */
void first_grid(){
	Triangulation<DIM> triangulation;

	GridGenerator::hyper_cube(triangulation);
	triangulation.refine_global(5);

	std::ofstream out ("grid-1.eps");
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "Grid written to grid-1.eps" << std::endl;
}

/**
 * Method creates a circular grid and then refines the inner most ring a fixed number
 * of times. It then exports the grid to a file.
 */
void second_grid(){
	Triangulation<DIM> triangulation;

	const Point<DIM> center(1,0);
	const double inner_radius = 0.5,
				 outer_radius = 1.0;
	GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

	for(unsigned int step = 0; step < 5; ++step){
		for(auto cell : triangulation.active_cell_iterators()){
			for(unsigned int v = 0; v < GeometryInfo<DIM>::vertices_per_cell; ++v){
				const double distance_from_center = center.distance(cell->vertex(v));
				if(std::fabs(distance_from_center - inner_radius) < 1e-10){
					cell->set_refine_flag();
					break;
				}
			}
		}
		triangulation.execute_coarsening_and_refinement();
	}

	std::ofstream out ("grid-2.eps");
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "Grid written to grid-2.eps" << std::endl;
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
	first_grid();
	if(DIM == 2){
		second_grid();
	}
}
