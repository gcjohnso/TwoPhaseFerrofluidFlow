/*
 * step-3.cc
 *
 *  Created on: Oct 21, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "step-3.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

/**
 * Default constructor: Associates the DoF object with the mesh and instructs
 * to use Q_2 polynomials.
 */
Step3::Step3()
	:
	fe(2), dof_handler(triangulation)
{}

/**
 * Creates a mesh of the unit square with variable level of refinement.
 * @param refinement_level Number of times to globally refine the mesh.
 */
void Step3::make_grid(int refinement_level){
	GridGenerator::hyper_cube(triangulation, -1, 1);
	triangulation.refine_global(refinement_level);

	std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

/**
 * Distributes DoFs and setups the matrices/vectors of the system AU = F.
 */
void Step3::setup_system(){
	dof_handler.distribute_dofs(fe);
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

	DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
	sparsity_pattern.copy_from(dynamic_sparsity_pattern);

	system_matrix.reinit(sparsity_pattern);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

/**
 * Computes the bilinear form and rhs for each cell in the triangulation. Also applies the homogeneous dirichlet
 * boundary conditions.
 */
void Step3::assemble_system(){
	QGauss<DIM> quadrature_formula(3);

	FEValues<DIM> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();
	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	for(const auto &cell : dof_handler.active_cell_iterators()){
		fe_values.reinit(cell);
		cell_matrix = 0;
		cell_rhs = 0;
		//Compute local contributions
		for(unsigned int q_index = 0; q_index < n_q_points; ++q_index){
			//Matrix A
			for(unsigned int i = 0; i < dofs_per_cell; ++i)
				for(unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i,j) += fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j,q_index) * fe_values.JxW(q_index);

			//Vector F
			for(unsigned int i = 0; i < dofs_per_cell; ++i)
				cell_rhs(i) += fe_values.shape_value(i, q_index) * 1 * fe_values.JxW(q_index);
		}
		//Transfer to global entries
		cell->get_dof_indices(local_dof_indices);
		for(unsigned int i = 0; i < dofs_per_cell; ++i)
			for(unsigned int j = 0; j < dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
		for(unsigned int i = 0; i < dofs_per_cell; ++i)
			system_rhs(local_dof_indices[i]) += cell_rhs(i);

	}

	//Incorporate boundary condition
	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<DIM>(), boundary_values);
	MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

/**
 * Solves the system using Conjugate Gradient with no preconditioner.
 */
void Step3::solve(){
	//Specifies a maximum of 1000 iterations or a tolerance of 1e-12.
	SolverControl solver_control(1000, 1e-12);
	SolverCG<> cg_solver(solver_control);
	cg_solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

/**
 * Generates a plot of the solution.
 */
void Step3::output_results() const{
	DataOut<DIM> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");
	data_out.build_patches();

//	std::ofstream output("solution.gpl");
//	data_out.write_gnuplot(output);

	std::ofstream output("solution.vtk");
	data_out.write_vtk(output);
}

/**
 * Runner method for the Step3 class.
 */
void Step3::run(){
	make_grid(5);
	setup_system();
	assemble_system();
	solve();
	output_results();
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
  deallog.depth_console(2);
  Step3 laplace_problem;
  laplace_problem.run();
  return 0;
}
