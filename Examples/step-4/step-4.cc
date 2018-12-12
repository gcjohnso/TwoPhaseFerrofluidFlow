/*
 * step-4.cc
 *
 *  Created on: Oct 23, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "step-4.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/logstream.h>

#include <fstream>
#include <iostream>

/**
 * Default constructor: Associates the DoF object with the mesh and instructs
 * to use Q_2 polynomials.
 */
template<int dim>
Step4<dim>::Step4()
	:
	fe(2), dof_handler(triangulation)
{}

/**
 * Creates a mesh of the unit square with four levels of global refinement.
 */
template <int dim>
void Step4<dim>::make_grid(){
	GridGenerator::hyper_cube(triangulation, -1, 1);
	triangulation.refine_global(4);

	std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
	std::cout << "Total number of cells: " << triangulation.n_cells() << std::endl;
}

/**
 * Distributes DoFs and setups the matrices/vectors of the system AU = F.
 */
template <int dim>
void Step4<dim>::setup_system(){
	dof_handler.distribute_dofs(fe);

	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);

	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

/**
 * Computes the bilinear form and rhs for each cell in the triangulation. Also applies the dirichlet
 * boundary conditions.
 */
template <int dim>
void Step4<dim>::assemble_system(){
	QGauss<dim> quadrature_formula(3);

	const RightHandSide<dim> right_hand_side;

	FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

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
			for(unsigned int i = 0; i < dofs_per_cell; ++i){
				//Matrix A
				for(unsigned int j = 0; j < dofs_per_cell; ++j){
					cell_matrix(i,j) += fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j,q_index) * fe_values.JxW(q_index);
				}
				//Vector F
				cell_rhs(i) += fe_values.shape_value(i,q_index) * right_hand_side.value(fe_values.quadrature_point(q_index)) * fe_values.JxW(q_index);
			}
		}

		//Transfer to global entries
		cell->get_dof_indices(local_dof_indices);
		for(unsigned int i = 0; i < dofs_per_cell; ++i){
			for(unsigned int j = 0; j < dofs_per_cell; ++j){
				system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
			}

			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}

	//Incorporate boundary condition
	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), boundary_values);
	MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

/**
 * Solves the system using Conjugate Gradient with no preconditioner.
 */
template <int dim>
void Step4<dim>::solve(){
	//Specifies a maximum of 1000 iterations or a tolerance of 1e-12.
	SolverControl solver_control(1000, 1e-12);
	SolverCG<> cg_solver(solver_control);
	cg_solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

	std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}

/**
 * Exports the solution to a .vtk file.
 */
template <int dim>
void Step4<dim>::output_results() const{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");

	data_out.build_patches();

	std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
	data_out.write_vtk(output);
}

/**
 * Runner method for the Step4 class.
 */
template <int dim>
void Step4<dim>::run(){
	std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

	make_grid();
	setup_system();
	assemble_system();
	solve();
	output_results();
}

/**
 * Returns the value of the forcing function at a point on the mesh.
 * @param p A point on the mesh
 * @param int Unused.
 * @return The value of the forcing function at the given point on the mesh.
 */
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const{
	double return_value = 0.0;
		for(unsigned int i=0; i<dim; ++i){
			return_value += 4.0 * std::pow(p(i), 4.0);
		}
	return return_value;
}

/**
 * Returns the value of the boundary condition at a point on the boundary of the mesh.
 * @param p A point on the boundary of the mesh.
 * @param int Unused.
 * @return The value of the boundary condition at the given point on the boundary of the mesh.
 */
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const{
	return p.square();
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
	deallog.depth_console(0);
	//Solve the problem in 2d
	{
		Step4<2> laplace_problem_2d;
		laplace_problem_2d.run();
	}
	//Solve the problem in 3d
	{
		Step4<3> laplace_problem_3d;
		laplace_problem_3d.run();
	}
}


