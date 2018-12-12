/*
 * step-8.cc
 *
 *  Created on: Nov 11, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "step-8.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>

using namespace Step8;

/**
 * Computes the forcing function at given point in the mesh.
 * @param points Points in the mesh.
 * @param values Data structure to hold the value of the forcing function at each of the points.
 */
template <int dim>
void right_hand_side(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values){
	Assert(values.size() == points.size(), ExcDimensionMismatch (values.size(), points.size()));
	Assert(dim >= 2, ExcNotImplemented());

	Point<dim> point_1, point_2;
	point_1(0) = 0.5;
	point_2(0) = -0.5;

	for(unsigned int point_n = 0; point_n < points.size(); ++point_n){
		//Set x-dir force if the point is in a neighborhood of either points.
		if(((points[point_n]-point_1).norm_square() < 0.2*0.2) || ((points[point_n]-point_2).norm_square() < 0.2*0.2)){
			values[point_n][0] = 1.0;
		}else{
			values[point_n][0] = 0.0;
		}
		//Set y-dir force if the point is near the origin.
		if (points[point_n].norm_square() < 0.2*0.2){
			values[point_n][1] = 1.0;
		}else{
			values[point_n][1] = 0.0;
		}
	}
}

/**
 * Constructor for ElasticProblem, uses Q2 elements in each dimension.
 */
template <int dim>
ElasticProblem<dim>::ElasticProblem()
	:
	dof_handler(triangulation), fe(FE_Q<dim>(2), dim)
{}

/**
 * Destructor for ElasticProblem.
 */
template <int dim>
ElasticProblem<dim>::~ElasticProblem(){
  dof_handler.clear();
}

/**
 * Distributes DoFs and handle the hanging node constraints.
 */
template <int dim>
void ElasticProblem<dim>::setup_system(){
	//Distribute DOFs and handle hanging node constraints
	dof_handler.distribute_dofs(fe);
	hanging_node_constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
	hanging_node_constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, hanging_node_constraints, /* keep_constrained_dofs = */ true);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);

	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

/**
 * Computes the bilinear form and rhs for each cell in the triangulation. Additionally applies the zero Dirichlet boundary condition.
 */
template <int dim>
void ElasticProblem<dim>::assemble_system(){
	QGauss<dim> quadrature_formula(3);

	FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	std::vector<double> lambda_values(n_q_points);
	std::vector<double> mu_values(n_q_points);

	Functions::ConstantFunction<dim> lambda(1.), mu(1.);

	std::vector<Tensor<1, dim>> rhs_values(n_q_points);

	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
	for(; cell!=endc; ++cell){
		cell_matrix = 0;
		cell_rhs = 0;

		fe_values.reinit(cell);

		//Get value of the coefficients and forcing function at each quadrature point
		lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
		mu.value_list(fe_values.get_quadrature_points(), mu_values);
		right_hand_side(fe_values.get_quadrature_points(), rhs_values);

		//Contribution of Bilinear form for the current cell
		for(unsigned int i=0; i<dofs_per_cell; ++i){
			//Which component of the basis vector is non-zero
			const unsigned int component_i = fe.system_to_component_index(i).first;
			for(unsigned int j=0; j<dofs_per_cell; ++j){
				//Which component of the basis vector is non-zero
				const unsigned int component_j = fe.system_to_component_index(j).first;
				for(unsigned int q_point=0; q_point<n_q_points; ++q_point) {
					cell_matrix(i,j) += fe_values.shape_grad(i,q_point)[component_i] * fe_values.shape_grad(j,q_point)[component_j]
									 * lambda_values[q_point] * fe_values.JxW(q_point);
					cell_matrix(i,j) += fe_values.shape_grad(i,q_point)[component_j] * fe_values.shape_grad(j,q_point)[component_i]
									 * mu_values[q_point] * fe_values.JxW(q_point);
					cell_matrix(i,j) += ((component_i == component_j) ?
							(fe_values.shape_grad(i,q_point) * fe_values.shape_grad(j,q_point) * mu_values[q_point]) : 0) * fe_values.JxW(q_point);
				}
			}
		}
		//Contribution of forcing function for the current cell
		for(unsigned int i=0; i<dofs_per_cell; ++i){
			const unsigned int component_i = fe.system_to_component_index(i).first;
			for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
				cell_rhs(i) += fe_values.shape_value(i,q_point) * rhs_values[q_point][component_i] * fe_values.JxW(q_point);
			}
		}
		//Transfer local to global
		cell->get_dof_indices(local_dof_indices);
		for(unsigned int i=0; i<dofs_per_cell; ++i){
			for (unsigned int j=0; j<dofs_per_cell; ++j){
				system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
			}
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
	//Handle hanging nodes constraints
	hanging_node_constraints.condense(system_matrix);
	hanging_node_constraints.condense(system_rhs);

	//Apply homogeneous BC
	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(dim), boundary_values);
	MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

/**
 * Solves the system using Conjugate Gradient with the SSOR preconditioner, then computes the constrained nodes using the unconstrained nodes.
 */
template <int dim>
void ElasticProblem<dim>::solve(){
	//Specifies a maximum of 1000 iterations or a tolerance of 1e-12.
	SolverControl solver_control(1000, 1e-12);
	SolverCG<> cg(solver_control);

	//Use SSOR preconditioner with a relaxation factor of 1.2
	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);

	cg.solve(system_matrix, solution, system_rhs, preconditioner);
	//Compute the solution at the constrained nodes.
	hanging_node_constraints.distribute(solution);
}

/**
 * Refine/coarses the mesh. Uses the Kelly error estimator to coarsen 3% of the cells and refine 30% of the cells.
 */
template <int dim>
void ElasticProblem<dim>::refine_grid(){
	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
	//Computes the error estimator on each element
	KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim-1>(3), typename FunctionMap<dim>::type(), solution, estimated_error_per_cell);

	//Marks the elements for refinement/coarsening
	GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);
	triangulation.execute_coarsening_and_refinement();
}

/**
 * Exports each component of the solution.
 * @param cycle The current level of refinement.
 */
template <int dim>
void ElasticProblem<dim>::output_results(const unsigned int cycle) const{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);

	std::vector<std::string> solution_names;
	switch(dim){
		case 1:
			solution_names.emplace_back("displacement");
			break;
		case 2:
			solution_names.emplace_back("x_displacement");
			solution_names.emplace_back("y_displacement");
			break;
		case 3:
			solution_names.emplace_back("x_displacement");
			solution_names.emplace_back("y_displacement");
			solution_names.emplace_back("z_displacement");
			break;
		default:
			Assert(false, ExcNotImplemented());
	}
	data_out.add_data_vector (solution, solution_names);
	data_out.build_patches ();

	std::ofstream output ("solution-" + std::to_string(cycle) + ".vtk");
	data_out.write_vtk (output);
}

/**
 * Runner method for the Step7 class. It solves the problem on a sequence
 * of adaptively refined meshes.
 */
template <int dim>
void ElasticProblem<dim>::run(){
	for(unsigned int cycle=0; cycle<8; ++cycle){
		std::cout << "Cycle " << cycle << ':' << std::endl;
		if(cycle == 0){
			GridGenerator::hyper_cube(triangulation, -1, 1);
			triangulation.refine_global(3);
		}else{
			refine_grid();
		}

		std::cout << "   Number of active cells: " << triangulation.n_active_cells() << std::endl;

		setup_system();

		std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

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
	try{
		Step8::ElasticProblem<2> elastic_problem_2d;
		elastic_problem_2d.run();
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
