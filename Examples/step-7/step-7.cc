/*
 * step-7.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: gcjohnso@math.umd.edu
 */

#include "step-7.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace Step7;

/**
 * Computes the value of the exact solution at a given point.
 * @param p A point in the mesh.
 * @param int Unused.
 * @return The solution value at the given point.
 */
template <int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const{
	double return_value = 0;
	for(unsigned int i=0; i<this->n_source_centers; ++i){
		const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
		return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
	}
	return return_value;
}

/**
 * Computes the gradient of the exact solution at a given point.
 * @param p A point in the mesh.
 * @param int Unused.
 * @return The gradient of the solution at the given point.
 */
template <int dim>
Tensor<1,dim> Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const{
	Tensor<1,dim> return_value;
	for(unsigned int i=0; i<this->n_source_centers; ++i){
		const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
		return_value += (-2 / (this->width * this->width) * std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
	}
	return return_value;
}

/**
 * Computes the value of the forcing function at a given point.
 * @param p A point in the mesh.
 * @param int Unused.
 * @return The forcing function value at the given point.
 */
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const{
	double return_value = 0;
	for(unsigned int i=0; i<this->n_source_centers; ++i){
		const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
		//Negative Laplacian
		return_value += ((2*dim - 4*x_minus_xi.norm_square()/ (this->width * this->width)) / (this->width * this->width) * std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
		//Solution
		return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
	}
	return return_value;
}

/**
 * Constructor for the HelmholtzProblem object.
 * @param fe Finite Element to be used.
 * @param refinement_mode Adaptive or Global refinement.
 */
template <int dim>
HelmholtzProblem<dim>::HelmholtzProblem(const FiniteElement<dim> &fe, const RefinementMode refinement_mode)
	:
	dof_handler(triangulation), fe(&fe), refinement_mode(refinement_mode)
{}

/**
 * Deconstructor for the HelmholtzProblem object.
 */
template <int dim>
HelmholtzProblem<dim>::~HelmholtzProblem(){
	dof_handler.clear();
}

/**
 * Distributes DoFs and handle the hanging node constraints.
 */
template <int dim>
void HelmholtzProblem<dim>::setup_system(){
	//Distribute and renumber the DOFs, done to increase performance of the SSOR preconditioner.
	dof_handler.distribute_dofs(*fe);
	DoFRenumbering::Cuthill_McKee(dof_handler);

	//Clear the previous hanging node constraints and create new handing node constraints
	hanging_node_constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
	hanging_node_constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	hanging_node_constraints.condense(dsp);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

/**
 * Computes the bilinear form and rhs for each cell in the triangulation. Additionally computes the boundary integral due to the Neumann condition.
 */
template <int dim>
void HelmholtzProblem<dim>::assemble_system(){
	QGauss<dim> quadrature_formula(fe->degree+1);
	QGauss<dim-1> face_quadrature_formula(fe->degree+1);

	const unsigned int n_q_points = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	const unsigned int dofs_per_cell = fe->dofs_per_cell;

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	FEValues<dim>  fe_values(*fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values(*fe, face_quadrature_formula, update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);

	const RightHandSide<dim> right_hand_side;
	std::vector<double> rhs_values(n_q_points);
	const Solution<dim> exact_solution;

	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
	for(; cell!=endc; ++cell){
		cell_matrix = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);

		right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);
		for(unsigned int q_point=0; q_point<n_q_points; ++q_point){
			for(unsigned int i=0; i<dofs_per_cell; ++i){
				for(unsigned int j=0; j<dofs_per_cell; ++j){
					//bilinear form
					cell_matrix(i,j) += fe_values.shape_grad(i,q_point) * fe_values.shape_grad(j,q_point) * fe_values.JxW(q_point);
					cell_matrix(i,j) += fe_values.shape_value(i,q_point) * fe_values.shape_value(j,q_point) * fe_values.JxW(q_point);
				}
				//Forcing function
				cell_rhs(i) += (fe_values.shape_value(i,q_point) * rhs_values [q_point] * fe_values.JxW(q_point));
			}
			//Neumann condition
			for(unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number){
				//Neumann boundary has id=1
				if(cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 1)){
					fe_face_values.reinit(cell, face_number);
					for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
						const double neumann_value = (exact_solution.gradient(fe_face_values.quadrature_point(q_point)) * fe_face_values.normal_vector(q_point));
						for(unsigned int i=0; i<dofs_per_cell; ++i){
							cell_rhs(i) += (neumann_value * fe_face_values.shape_value(i,q_point) * fe_face_values.JxW(q_point));
						}
					}
				}
			}
			//Transfer local to global
			cell->get_dof_indices(local_dof_indices);
			for(unsigned int i=0; i<dofs_per_cell; ++i){
				for(unsigned int j=0; j<dofs_per_cell; ++j){
					system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
				}
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
		}
	}
	hanging_node_constraints.condense(system_matrix);
	hanging_node_constraints.condense(system_rhs);

	//Apply the Dirichlet BC
	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler, 0, Solution<dim>(), boundary_values);
	MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

/**
 * Solves the system using Conjugate Gradient with the SSOR preconditioner, then computes the constrained nodes using the unconstrained nodes.
 */
template <int dim>
void HelmholtzProblem<dim>::solve(){
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
 * Refine/coarses the mesh. Options are global or adaptive refinement. If adaptive, it uses the Kelly error estimator to coarsen 3%
 * of the cells and refine 30% of the cells.
 */
template <int dim>
void HelmholtzProblem<dim>::refine_grid(){
	switch(refinement_mode){
		case global_refinement:{
			triangulation.refine_global(1);
			break;
		}
		case adaptive_refinement:{
			Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
			//Computes the error estimator on each element
			KellyErrorEstimator<dim>::estimate (dof_handler, QGauss<dim-1>(3), typename FunctionMap<dim>::type(), solution, estimated_error_per_cell);

			//Marks the elements for refinement/coarsening
			GridRefinement::refine_and_coarsen_fixed_number (triangulation, estimated_error_per_cell, 0.3, 0.03);
			triangulation.execute_coarsening_and_refinement ();
			break;
		}
		default:{
			Assert (false, ExcNotImplemented());
		}
	}
}

/**
 * Computes the L2, H1-semi, and LInf error, then computes the convergence rate for the global.
 * @param cycle Current level of refinement.
 */
template <int dim>
void HelmholtzProblem<dim>::process_solution(const unsigned int cycle){
	//Compute the L2 norm
	Vector<float> difference_per_cell(triangulation.n_active_cells());
	VectorTools::integrate_difference(dof_handler, solution, Solution<dim>(), difference_per_cell, QGauss<dim>(fe->degree+1), VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

	//Compute H1 semi-norm
	VectorTools::integrate_difference(dof_handler, solution, Solution<dim>(), difference_per_cell, QGauss<dim>(fe->degree+1), VectorTools::H1_seminorm);
	const double H1_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_seminorm);

	//Compute LInf norm using a different quadrature rule
	const QTrapez<1> q_trapez;
	const QIterated<dim> q_iterated(q_trapez, 5);
	VectorTools::integrate_difference(dof_handler, solution, Solution<dim>(), difference_per_cell, q_iterated, VectorTools::Linfty_norm);
	const double Linfty_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

	//Add the errors to the convergence table
	const unsigned int n_active_cells=triangulation.n_active_cells();
	const unsigned int n_dofs=dof_handler.n_dofs();

	std::cout << "Cycle " << cycle << ':' << std::endl;
	std::cout << "  Number of active cells: " << n_active_cells << std::endl;
	std::cout << "  Number of degrees of freedom: " << n_dofs << std::endl;

	convergence_table.add_value("cycle", cycle);
	convergence_table.add_value("cells", n_active_cells);
	convergence_table.add_value("dofs", n_dofs);
	convergence_table.add_value("L2", L2_error);
	convergence_table.add_value("H1", H1_error);
	convergence_table.add_value("Linfty", Linfty_error);
}

/**
 * Runner method for the Step7 class. It first solves the problem using a given
 * element and refinement type on a sequence of meshes. It then computes the
 * error in various norms.
 */
template <int dim>
void HelmholtzProblem<dim>::run(){
	const unsigned int n_cycles = (refinement_mode==global_refinement)?5:9;
	for(unsigned int cycle=0; cycle<n_cycles; ++cycle){
		if (cycle == 0){
			GridGenerator::hyper_cube(triangulation, -1, 1);
			triangulation.refine_global(3);

			//Mark Neumann boundary
			typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
			for(; cell!=endc; ++cell){
				for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number){
					if((std::fabs(cell->face(face_number)->center()(0) - (-1)) < 1e-12) || (std::fabs(cell->face(face_number)->center()(1) - (-1)) < 1e-12)){
						cell->face(face_number)->set_boundary_id(1);
					}
				}
			}
		}else{
			refine_grid();
		}
		setup_system();
		assemble_system();
		solve();
		process_solution(cycle);
	}

	//Export solution data
	std::string vtk_filename;
	switch(refinement_mode){
		case global_refinement:
			vtk_filename = "solution-global";
			break;
		case adaptive_refinement:
			vtk_filename = "solution-adaptive";
			break;
		default:
			Assert(false, ExcNotImplemented());
	}
	switch(fe->degree){
		case 1:
			vtk_filename += "-q1";
			break;
		case 2:
			vtk_filename += "-q2";
			break;
		default:
			Assert(false, ExcNotImplemented());
	}
	vtk_filename += ".vtk";
	std::ofstream output (vtk_filename.c_str());

	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");

	data_out.build_patches(fe->degree);
	data_out.write_vtk(output);

	//Ouput convergence tables
	convergence_table.set_precision("L2", 3);
	convergence_table.set_precision("H1", 3);
	convergence_table.set_precision("Linfty", 3);

	convergence_table.set_scientific("L2", true);
	convergence_table.set_scientific("H1", true);
	convergence_table.set_scientific("Linfty", true);

	convergence_table.set_tex_caption("cells", "\\# cells");
	convergence_table.set_tex_caption("dofs", "\\# dofs");
	convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
	convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
	convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");

	convergence_table.set_tex_format("cells", "r");
	convergence_table.set_tex_format("dofs", "r");

	std::cout << std::endl;
	convergence_table.write_text(std::cout);

	std::string error_filename = "error";
	switch(refinement_mode){
		case global_refinement:
			error_filename += "-global";
			break;
		case adaptive_refinement:
			error_filename += "-adaptive";
			break;
		default:
			Assert(false, ExcNotImplemented());
	}
	switch(fe->degree){
		case 1:
			error_filename += "-q1";
			break;
		case 2:
			error_filename += "-q2";
			break;
		default:
			Assert(false, ExcNotImplemented());
	}
	error_filename += ".tex";
	std::ofstream error_table_file(error_filename.c_str());

	convergence_table.write_tex(error_table_file);

	//Compute convergence table only for global refinement.
	if(refinement_mode==global_refinement){
		convergence_table.add_column_to_supercolumn("cycle", "n cells");
		convergence_table.add_column_to_supercolumn("cells", "n cells");

		std::vector<std::string> new_order;
		new_order.emplace_back("n cells");
		new_order.emplace_back("H1");
		new_order.emplace_back("L2");
		convergence_table.set_column_order (new_order);

		convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
		convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
		convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
		convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

		std::cout << std::endl;
		convergence_table.write_text(std::cout);

		std::string conv_filename = "convergence";
		switch(refinement_mode){
			case global_refinement:
				conv_filename += "-global";
				break;
			case adaptive_refinement:
				conv_filename += "-adaptive";
				break;
			default:
				Assert(false, ExcNotImplemented());
		}
		switch(fe->degree){
			case 1:
				conv_filename += "-q1";
				break;
			case 2:
				conv_filename += "-q2";
				break;
			default:
				Assert(false, ExcNotImplemented());
		}
		conv_filename += ".tex";

		std::ofstream table_file(conv_filename.c_str());
		convergence_table.write_tex(table_file);
	}
}

/**
 * Runner method. Solves the PDE using combinations of Q_1/Q_2 and global/adaptive refinement.
 * @return Unused.
 */
int main(){
	const unsigned int dim = 2;
	try{
		using namespace dealii;
		using namespace Step7;
		{
			std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl;
			std::cout << "=============================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(1);
			HelmholtzProblem<dim>
			helmholtz_problem_2d(fe, HelmholtzProblem<dim>::adaptive_refinement);
			helmholtz_problem_2d.run();
			std::cout << std::endl;
		}
		{
			std::cout << "Solving with Q1 elements, global refinement" << std::endl;
			std::cout << "===========================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(1);
			HelmholtzProblem<dim>
			helmholtz_problem_2d(fe, HelmholtzProblem<dim>::global_refinement);
			helmholtz_problem_2d.run();
			std::cout << std::endl;
		}
		{
			std::cout << "Solving with Q2 elements, global refinement" << std::endl;
			std::cout << "===========================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(2);
			HelmholtzProblem<dim>
			helmholtz_problem_2d(fe, HelmholtzProblem<dim>::global_refinement);
			helmholtz_problem_2d.run();
			std::cout << std::endl;
		}
		{
			std::cout << "Solving with Q2 elements, adaptive refinement" << std::endl;
			std::cout << "===========================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(2);
			HelmholtzProblem<dim>
			helmholtz_problem_2d(fe, HelmholtzProblem<dim>::adaptive_refinement);
			helmholtz_problem_2d.run();
			std::cout << std::endl;
		}
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

