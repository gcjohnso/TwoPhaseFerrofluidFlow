/*
 * step-12.cc
 *
 *  Created on: Nov 15, 2018
 *      Author: gcjohnso@math.umd.edu
 */
#include "step-12.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <iostream>
#include <fstream>

using namespace Step12;

/**
 * Returns the boundary values for the inflow boundary.
 * @param points A vector of points on the inflow boundary of the mesh.
 * @param values Vector where the boundary values are stored.
 * @param int Unused.
 */
template <int dim>
void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int) const{
	Assert(values.size()==points.size(), ExcDimensionMismatch(values.size(),points.size()));

	for(unsigned int i=0; i<values.size(); ++i){
		if (points[i](0)<0.5){
			values[i]=1.;
		}else{
			values[i]=0.;
		}
	}
}

/**
 * Returns the value of the coefficient b at a given point in the mesh, where b is a circular counterclockwise flow field.
 * @param p A point in the mesh.
 * @return The value of the coefficient b at the point.
 */
template <int dim>
Tensor<1,dim> beta(const Point<dim> &p){
	Assert(dim >= 2, ExcNotImplemented());

	Point<dim> wind_field;
	wind_field(0) = -p(1);
	wind_field(1) = p(0);
	wind_field /= wind_field.norm();

	return wind_field;
}


/**
 * Default constructor
 */
template <int dim>
AdvectionProblem<dim>::AdvectionProblem()
  :
  mapping(), fe(1), dof_handler(triangulation)
{}

/**
 * Distributes DoFs and setups the matrices/vectors of the system AU = F.
 */
template <int dim>
void AdvectionProblem<dim>::setup_system(){
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  right_hand_side.reinit(dof_handler.n_dofs());
}

/**
 * Assemble the bilinear and RHS using the MeshWorker framework.
 */
template <int dim>
void AdvectionProblem<dim>::assemble_system(){
	MeshWorker::IntegrationInfoBox<dim> info_box;

	//Use a quadrature scheme one higher than the degree of the element.
	const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
	info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points, n_gauss_points);

	//Flags we need (based on the bilinear form) to be used on cells, boundary and interior faces, and interior neighbor faces.
	info_box.initialize_update_flags();
	UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
	info_box.add_update_flags(update_flags, true, true, true, true);

	//Initialize the FEValues object.
	info_box.initialize(fe, mapping);

	MeshWorker::DoFInfo<dim> dof_info(dof_handler);

	MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;
	assembler.initialize(system_matrix, right_hand_side);

	//Tell the MeshWorker framework which functions to use to integrate each cell, boundary, and interior face term.
	MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim>>(dof_handler.begin_active(), dof_handler.end(),
			dof_info, info_box,  &AdvectionProblem<dim>::integrate_cell_term, &AdvectionProblem<dim>::integrate_boundary_term,
			&AdvectionProblem<dim>::integrate_face_term, assembler);
}

/**
 * Computes the contribution of the bilinear form at a given cell.
 * @param dinfo Object containing info about the FE and DOF.
 * @param info Object containing information about the cell.
 */
template <int dim>
void AdvectionProblem<dim>::integrate_cell_term(DoFInfo &dinfo, CellInfo &info){
	const FEValuesBase<dim> &fe_values = info.fe_values();
	FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
	const std::vector<double> &JxW = fe_values.get_JxW_values();

	for(unsigned int point=0; point<fe_values.n_quadrature_points; ++point){
	    const Tensor<1,dim> beta_at_q_point = beta(fe_values.quadrature_point(point));
	    for(unsigned int i=0; i<fe_values.dofs_per_cell; ++i){
			for (unsigned int j=0; j<fe_values.dofs_per_cell; ++j){
				local_matrix(i,j) += -beta_at_q_point * fe_values.shape_grad(i,point) * fe_values.shape_value(j,point) * JxW[point];
			}
	    }
	}
}

/**
 * Computes the contribution of the boundary terms to the system.
 * @param dinfo Object containing info about where to store the results.
 * @param info Object containing info about the about the FE and DOF on the cell.
 */
template <int dim>
void AdvectionProblem<dim>::integrate_boundary_term (DoFInfo &dinfo, CellInfo &info){
	const FEValuesBase<dim> &fe_face_values = info.fe_values();
	FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
	Vector<double> &local_vector = dinfo.vector(0).block(0);

	const std::vector<double> &JxW = fe_face_values.get_JxW_values();
	const std::vector<Tensor<1,dim>> &normals = fe_face_values.get_normal_vectors();

	//Vector to hold the values of the inflow condition at each quadrature point
	std::vector<double> g(fe_face_values.n_quadrature_points);

	static BoundaryValues<dim> boundary_function;
	boundary_function.value_list(fe_face_values.get_quadrature_points(), g);

	for(unsigned int point=0; point<fe_face_values.n_quadrature_points; ++point){
		const double beta_dot_n = beta(fe_face_values.quadrature_point(point)) * normals[point];
		if(beta_dot_n>0){
			//Non inflow boundary (this term comes from upwinding)
			for(unsigned int i=0; i<fe_face_values.dofs_per_cell; ++i){
				for(unsigned int j=0; j<fe_face_values.dofs_per_cell; ++j){
					local_matrix(i,j) += beta_dot_n * fe_face_values.shape_value(j,point) * fe_face_values.shape_value(i,point) * JxW[point];
				}
			}
		}else{
			//Inflow boundary
			for(unsigned int i=0; i<fe_face_values.dofs_per_cell; ++i){
				local_vector(i) += -beta_dot_n * g[point] * fe_face_values.shape_value(i,point) * JxW[point];
			}
		}
	}
}

/**
 * Computes the contribution of the interior face terms to the system.
 * @param dinfo1 Object containing info about where to store the results for matrix 1.
 * @param dinfo2 Object containing info about where to store the results for matrix 2.
 * @param info1 Object containing info about the about the FE and DOF on cell 1.
 * @param info2 Object containing info about the about the FE and DOF on cell 2.
 */
template <int dim>
void AdvectionProblem<dim>::integrate_face_term (DoFInfo &dinfo1, DoFInfo &dinfo2, CellInfo &info1, CellInfo &info2){
	const FEValuesBase<dim> &fe_face_values = info1.fe_values();
	const FEValuesBase<dim> &fe_face_values_neighbor = info2.fe_values();

	//Local matrices, subscripts correspond to which element the shape function lies. If the subscript matches,
	//then both lie on the same element, otherwise each shape function is from a different element.
	FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0,false).matrix;
	FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0,true).matrix;
	FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0,true).matrix;
	FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0,false).matrix;

	const std::vector<double> &JxW = fe_face_values.get_JxW_values();
	const std::vector<Tensor<1,dim> > &normals = fe_face_values.get_normal_vectors();

	for(unsigned int point=0; point<fe_face_values.n_quadrature_points; ++point){
		const double beta_dot_n = beta(fe_face_values.quadrature_point(point)) * normals[point];
		//Switch depending on the direction of the flow
	    if(beta_dot_n>0){
			for(unsigned int i=0; i<fe_face_values.dofs_per_cell; ++i){
				for(unsigned int j=0; j<fe_face_values.dofs_per_cell; ++j){
					u1_v1_matrix(i,j) += beta_dot_n * fe_face_values.shape_value(j,point) * fe_face_values.shape_value(i,point) * JxW[point];
				}
			}
			for(unsigned int k=0; k<fe_face_values_neighbor.dofs_per_cell; ++k){
				for(unsigned int j=0; j<fe_face_values.dofs_per_cell; ++j){
					u1_v2_matrix(k,j) += -beta_dot_n * fe_face_values.shape_value(j,point) * fe_face_values_neighbor.shape_value(k,point) * JxW[point];
				}
			}
	    }else{
			for(unsigned int i=0; i<fe_face_values.dofs_per_cell; ++i){
				for(unsigned int l=0; l<fe_face_values_neighbor.dofs_per_cell; ++l){
					u2_v1_matrix(i,l) += beta_dot_n * fe_face_values_neighbor.shape_value(l,point) * fe_face_values.shape_value(i,point) * JxW[point];
				}
			}
			for(unsigned int k=0; k<fe_face_values_neighbor.dofs_per_cell; ++k){
				for(unsigned int l=0; l<fe_face_values_neighbor.dofs_per_cell; ++l){
					u2_v2_matrix(k,l) += -beta_dot_n * fe_face_values_neighbor.shape_value(l,point) * fe_face_values_neighbor.shape_value(k,point) * JxW[point];
				}
			}
	    }
	}
}

/**
 * Solves the system using a Richardson iteration with a BlockSSOR preconditioner.
 * @param solution Data structure to store the solution.
 */
template <int dim>
void AdvectionProblem<dim>::solve(Vector<double> &solution){
	//Specifies a maximum of 1000 iterations or a tolerance of 1e-12
	SolverControl solver_control(1000, 1e-12);
	SolverRichardson<> solver(solver_control);

	//Use SSOR preconditioner
	PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
	preconditioner.initialize(system_matrix, fe.dofs_per_cell);

	solver.solve(system_matrix, solution, right_hand_side, preconditioner);
}

/**
 * Refine the mesh by using approximate gradients of the solution as an error indicator.
 */
template <int dim>
void AdvectionProblem<dim>::refine_grid(){
	Vector<float> gradient_indicator(triangulation.n_active_cells());
	//Approximate the gradient
	DerivativeApproximation::approximate_gradient(mapping, dof_handler, solution, gradient_indicator);

	//Compute the error estimate h^(1+.5)abs(grad u)
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
	for(unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no){
		gradient_indicator(cell_no) *= std::pow(cell->diameter(), 1+1.0*dim/2);
	}

	//Coarsen 10% of the mesh and refine 30% of the mesh
	GridRefinement::refine_and_coarsen_fixed_number(triangulation, gradient_indicator, 0.3, 0.1);
	triangulation.execute_coarsening_and_refinement();
}

/**
 * Output the mesh and solution at a given refinement level.
 * @param cycle The current level of refinement.
 */
template <int dim>
void AdvectionProblem<dim>::output_results(const unsigned int cycle) const{
	//Print the grid.
	{
		const std::string filename = "grid-" + std::to_string(cycle) + ".eps";
		deallog << "Writing grid to <" << filename << ">" << std::endl;
		std::ofstream eps_output(filename.c_str());
		GridOut grid_out;
		grid_out.write_eps(triangulation, eps_output);
	}
	//Print the mesh.
	{
		const std::string filename = "sol-" + std::to_string(cycle) + ".gnuplot";
		deallog << "Writing solution to <" << filename << ">" << std::endl;
		std::ofstream gnuplot_output(filename.c_str());
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(solution, "u");
		data_out.build_patches();
		data_out.write_gnuplot(gnuplot_output);
	}
}

/**
 * Runner method for the Step12 class. It solves the problem on a sequence of
 * adaptively refined meshes.
 */
template <int dim>
void AdvectionProblem<dim>::run(){
	for(unsigned int cycle=0; cycle<6; ++cycle){
		deallog << "Cycle " << cycle << std::endl;
		if(cycle == 0){
			GridGenerator::hyper_cube(triangulation);
			triangulation.refine_global(3);
		}
		else{
			refine_grid();
		}

		deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
		setup_system();
		deallog << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
		assemble_system();
		solve(solution);
		output_results(cycle);
	}
}

/**
 * Runner method.
 * @return Unused.
 */
int main(){
	try{
		Step12::AdvectionProblem<2> dgmethod;
		dgmethod.run();
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

