/*
 * step-5.cc
 *
 *  Created on: Oct 31, 2018
 *      Author: gcjohnso@math.umd.edu
 */
#include "step-5.h"

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
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
 * Method to return the value of the coefficient a(x) at a point on the mesh.
 * @param p A point on the mesh.
 * @return The value of the coefficient a(x) at the given point on the mesh.
 */
template <int dim>
double coefficient(const Point<dim> &p){
	if (p.square() < 0.5*0.5)
		return 20;
	else
		return 1;
}

/**
 * Default constructor: Associates the DoF object with the mesh and instructs
 * to use Q_2 polynomials.
 */
template <int dim>
Step5<dim>::Step5()
	:
	fe(2), dof_handler(triangulation)
{}

/**
 * Distributes DoFs and setups the matrices/vectors of the system AU = F.
 */
template <int dim>
void Step5<dim>::setup_system(){
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
 * Computes the bilinear form and rhs for each cell in the triangulation. Also applies the
 * homogeneous dirichlet boundary conditions.
 */
template <int dim>
void Step5<dim>::assemble_system(){
	QGauss<dim> quadrature_formula(3);

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
			const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_index));
			for(unsigned int i = 0; i < dofs_per_cell; ++i){
				//Matrix A
				for(unsigned int j = 0; j < dofs_per_cell; ++j){
					cell_matrix(i,j) += current_coefficient * fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j,q_index) * fe_values.JxW(q_index);
				}
				//Vector F
				cell_rhs(i) += fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index);
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
	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);
	MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
}

/**
 * Solves the system using Conjugate Gradient preconditioned with SSOR.
 */
template <int dim>
void Step5<dim>::solve(){
	//Specifies a maximum of 1000 iterations or a tolerance of 1e-12.
	SolverControl solver_control(1000, 1e-12);
	SolverCG<> cg_solver(solver_control);

	//Use SSOR preconditioner with a relaxation factor of 1.2
	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);

	cg_solver.solve(system_matrix, solution, system_rhs, preconditioner);

	std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}

/**
 * Exports the solution at a given refinement level gnuplot file.
 * @param cycle Current level of refinement.
 */
template <int dim>
void Step5<dim>::output_results(const unsigned int cycle) const{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");

	data_out.build_patches();

//	DataOutBase::EpsFlags eps_flags;
//	eps_flags.z_scaling = 4;
//	eps_flags.azimut_angle = 40;
//	eps_flags.turn_angle = 10;
//	data_out.set_flags(eps_flags);

	std::ofstream output("solution-" + std::to_string(cycle) + ".gpl");
	data_out.write_gnuplot(output);
}

/**
 * Runner method for the Step5 class. It first reads in a given mesh file and then
 * solves the problem on successively refined meshes.
 */
template <int dim>
void Step5<dim>::run(){
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input_file("circle-grid.inp");
	Assert(dim==2, ExcInternalError());

	grid_in.read_ucd(input_file);

	//Apply and label the boundary of the mesh.
	const SphericalManifold<dim> boundary;
	triangulation.set_all_manifold_ids_on_boundary(0);
	triangulation.set_manifold(0, boundary);

	for(unsigned int cycle = 0; cycle < 6; ++cycle){
		std::cout << "Cycle " << cycle << ":" << std::endl;

		if(cycle != 0)
			triangulation.refine_global(1);

		std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
		std::cout << "Total number of cells: " << triangulation.n_cells() << std::endl;

		setup_system();
		assemble_system();
		solve();
		output_results(cycle);
	}
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
	Step5<2>laplace_problem_2d;
	laplace_problem_2d.run();
	return 0;
}
