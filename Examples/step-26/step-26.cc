/*
 * step-26.cc
 *
 *  Created on: Nov 22, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "step-26.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <math.h>

using namespace Step26;

/**
 * Constructor for the forcing function.
 */
template <int dim>
RightHandSide<dim>::RightHandSide()
	:
	Function<dim>()
{}

/**
 * Return the value of the forcing function at a point in the mesh.
 * @param p A point in the mesh.
 * @param component
 * @return The value of the forcing function at the point.
 */
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int component) const{
	(void) component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));
	Assert(dim == 2, ExcNotImplemented());

	const double t = this->get_time();

	double retVal = 0;
	retVal += -2*sin(3*PI*p[0])*exp(-p[1]-2*t);
	retVal -= (1-9*PI*PI)*sin(3*PI*p[0])*exp(-p[1]-2*t);

	return retVal;
}

/**
 * Constructor for the HeatEquation.
 * @param degree Desired degree of the element.
 * @param theta Value for the theta-scheme. 1 for BE, 1/2 for CN, and 0 for FE.
 */
template <int dim>
HeatEquation<dim>::HeatEquation(int degree, int theta)
  :
  fe(degree), dof_handler(triangulation), time(0.0), time_step(1. / 500), timestep_number(0), theta(theta)
{}

/**
 * Return the Dirichlet BC u=sin(3pix)e^(-y-2t) at a given point.
 * @param p A point on the boundary of the mesh.
 * @param component
 * @return The value of the BC at the point.
 */
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & p, const unsigned int component) const{
	(void) component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));

	const double t = this->get_time();

	return sin(3*PI*p[0])*exp(-p[1]-2*t);
}

/**
 * Distributes the DoF, applies the handing node constraints, and creates the mass and stiffness
 * matrices.
 */
template <int dim>
void HeatEquation<dim>::setup_system(){
	dof_handler.distribute_dofs(fe);

	std::cout << std::endl;
	std::cout << "===========================================" << std::endl;
	std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	std::cout << std::endl;

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs = */ true);
	sparsity_pattern.copy_from(dsp);

	mass_matrix.reinit(sparsity_pattern);
	laplace_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);

	//Use library methods to compute the mass and stiffness matrices for the system
	MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree+1), mass_matrix);
	MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree+1), laplace_matrix);

	solution.reinit(dof_handler.n_dofs());
	old_solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

/**
 * Solves the system for the next time increment using CG with a SSOR preconditioner.
 * It then computes the value of the solution at the hanging nodes.
 */
template <int dim>
void HeatEquation<dim>::solve_time_step(){
	SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
	SolverCG<> cg(solver_control);

	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.0);

	cg.solve(system_matrix, solution, system_rhs, preconditioner);

	constraints.distribute(solution);

	std::cout << "     " << solver_control.last_step() << " CG iterations." << std::endl;
}

/**
 * Method to output the solution to a .vtk file.
 */
template <int dim>
void HeatEquation<dim>::output_results() const{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "U");

	data_out.build_patches();

	const std::string filename = "solutionData/solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
	std::ofstream output(filename.c_str());
	data_out.write_vtk(output);
}

/**
 * Refines the mesh by coarsening 40% of the mesh and refining 60% of the mesh, using the Kelly error estimator.
 * In addition, it enforces that no element if coarsened below a minimum level of refinement and refined above a
 * maximum level of refinement.
 * @param min_grid_level Minimum level of refinement for elements.
 * @param max_grid_level Maximum level of refinement for elements.
 */
template <int dim>
void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level, const unsigned int max_grid_level){
	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

	KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim-1>(fe.degree+1), typename FunctionMap<dim>::type(), solution, estimated_error_per_cell);

	//Marks elements for refinement
	GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.6, 0.4);

	//Enforces that we don't refine past our max level of refinement
	if(triangulation.n_levels() > max_grid_level){
		for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(max_grid_level); cell != triangulation.end(); ++cell){
			cell->clear_refine_flag();
		}
	}
	//Enforces that we don't coarsen below our min level of refinement
	for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(min_grid_level); cell != triangulation.end_active(min_grid_level); ++cell){
		cell->clear_coarsen_flag();
	}

	//Object that handles the transfer of the solution
	SolutionTransfer<dim> solution_trans(dof_handler);

	Vector<double> previous_solution;
	previous_solution = solution;
	triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

	triangulation.execute_coarsening_and_refinement();
	setup_system();

	solution_trans.interpolate(previous_solution, solution);
	constraints.distribute(solution);
}

/**
 * Runner method for the Step26 class. It initializes the system and then
 * computes the PDE until T=1 using a theta-scheme. This method handles
 * aggregation of the system and calls methods to refine the mesh and
 * to solve the system at a given time.
 */
template <int dim>
void HeatEquation<dim>::run(){
	const unsigned int initial_global_refinement = 2;
	const unsigned int n_adaptive_pre_refinement_steps = 6;

	GridGenerator::hyper_cube(triangulation, 0, 1);
	triangulation.refine_global(initial_global_refinement);

	setup_system();

	unsigned int pre_refinement_step = 0;

	Vector<double> tmp;
	Vector<double> forcing_terms;

start_time_iteration:

	tmp.reinit(solution.size());
	forcing_terms.reinit(solution.size());

	//Interpolate the IC onto the mesh
	BoundaryValues<dim> initialCondition;
	initialCondition.set_time(time);
	VectorTools::interpolate(dof_handler, initialCondition, old_solution);
	solution = old_solution;

	output_results();

	while(time <= 1){
		time += time_step;
		++timestep_number;

		std::cout << "Time step " << timestep_number << " at t=" << time << std::endl;

		//Compute the RHS
		//First compute AU^{n-1} - (1-theta)kAU^{n-1}
		mass_matrix.vmult(system_rhs, old_solution);

		laplace_matrix.vmult(tmp, old_solution);
		system_rhs.add(-(1 - theta) * time_step, tmp);

		//Then compute k[(1-theta)F^{n-1} + thetaF^{n}]
		RightHandSide<dim> rhs_function;
		rhs_function.set_time(time);
		VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree+1), rhs_function, tmp);
		forcing_terms = tmp;
		forcing_terms *= time_step * theta;

		rhs_function.set_time(time - time_step);
		VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree+1), rhs_function, tmp);
		forcing_terms.add(time_step * (1 - theta), tmp);

		system_rhs += forcing_terms;

		system_matrix.copy_from(mass_matrix);
		system_matrix.add(theta * time_step, laplace_matrix);

		//Eliminate handing nodes
		constraints.condense(system_matrix, system_rhs);

		//Apply BC
		{
			BoundaryValues<dim> boundary_values_function;
			boundary_values_function.set_time(time);

			std::map<types::global_dof_index, double> boundary_values;
			VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_values_function, boundary_values);
			MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
		}

		solve_time_step();

		output_results();

		if((timestep_number == 1) && (pre_refinement_step < n_adaptive_pre_refinement_steps)){
			refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
			++pre_refinement_step;

			tmp.reinit(solution.size());
			forcing_terms.reinit(solution.size());

			std::cout << std::endl;

			goto start_time_iteration;
		}else if((timestep_number > 0) && (timestep_number % 5 == 0)){
			refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
			tmp.reinit(solution.size());
			forcing_terms.reinit(solution.size());
		}
		old_solution = solution;
	}
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
	try{
		using namespace dealii;
		using namespace Step26;
		//Use bilinear elements and BE
		HeatEquation<2> heat_equation_solver(1, 1);
		heat_equation_solver.run();
	}catch(std::exception &exc){
		std::cerr << std::endl << std::endl;
		std::cerr << "----------------------------------------------------" << std::endl;
		std::cerr << "Exception on processing: " << std::endl;
		std::cerr << exc.what() << std::endl;
		std::cerr << "Aborting!" << std::endl;
		std::cerr << "----------------------------------------------------" << std::endl;
		return 1;
	}catch(...){
		std::cerr << std::endl << std::endl;
		std::cerr << "----------------------------------------------------" << std::endl;
		std::cerr << "Unknown exception!" << std::endl;
		std::cerr << "Aborting!" << std::endl;
		std::cerr << "----------------------------------------------------" << std::endl;
		return 1;
	}

	return 0;
}

