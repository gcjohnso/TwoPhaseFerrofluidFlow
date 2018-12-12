/*
 * step-2.cc
 *
 *  Created on: Oct 19, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 * This program creates a circular mesh, then distributes DOFs for Q_2 shape functions
 * on the mesh. After the initial distribution, it then renumbers the DOFs using the
 * Cuthill Mckee front marching algorithm in order for the DOFs to lie closer to the
 * diagonal. A comparison of the sparsity patterns can be seen by examining the
 * files sparsity_pattern_1 & sparsity_pattern_2 in gnuplot.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>

#define DIM 2

using namespace dealii;

/**
 * Creates a circular mesh.
 * @param triangulation The circular mesh.
 */
void make_grid(Triangulation<DIM> &triangulation){
	const Point<2> center(1,0);
	const double inner_radius = 0.5, outer_radius = 1.0;
	GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

	for(unsigned int step = 0; step < 3; ++step){
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
}

/**
 * Distributes degrees of freedom (For Q_2 shape functions) onto the mesh.
 * It then prints out the default sparsity pattern.
 * @param dof_handler Object containing info on how the DOFs are distributed.
 */
void distribute_dofs(DoFHandler<DIM> &dof_handler){
	static const FE_Q<2> finite_element(2);
	dof_handler.distribute_dofs(finite_element);

	DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);

	SparsityPattern sparsity_pattern;
	sparsity_pattern.copy_from(dynamic_sparsity_pattern);

	std::ofstream out("sparsity_pattern_1");
	sparsity_pattern.print_gnuplot(out);
}

/**
 *	Renumber the DoFs so that they lie closer to the diagonal, uses the Cuthill Mckee front marching algorithm
 *	to perform the renumbering. It then prints out the new sparsity pattern.
 * @param dof_handler Object containing how the DOFs are distributed.
 */
void renumber_dofs(DoFHandler<DIM> &dof_handler){
	DoFRenumbering::Cuthill_McKee(dof_handler);

	DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);

	SparsityPattern sparsity_pattern;
	sparsity_pattern.copy_from(dynamic_sparsity_pattern);

	std::ofstream out("sparsity_pattern_2");
	sparsity_pattern.print_gnuplot(out);
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
	Triangulation<DIM> triangulation;
	make_grid(triangulation);

	DoFHandler<DIM> dof_handler(triangulation);

	distribute_dofs(dof_handler);
	renumber_dofs(dof_handler);
}
