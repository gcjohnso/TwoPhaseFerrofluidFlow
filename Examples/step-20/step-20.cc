/*
 * step-20.cc
 *
 *  Created on: Nov 18, 2018
 *      Author: gcjohnso@math.umd.edu
 */
#include "step-20.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

using namespace Step20;

/**
 * Method to return the forcing function, which for this example if f=0.
 * @param p A point on the mesh.
 * @param component Component index.
 * @return The value of the forcing function at the point.
 */
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int component) const{
	return 0;
}

/**
 * Method to return the value of the pressure at a point on the boundary of the mesh.
 * @param p A point on the mesh.
 * @param component Component index.
 * @return The value of the pressure at the given point on the boundary of the mesh.
 */
template <int dim>
double PressureBoundaryValues<dim>::value(const Point<dim>  &p, const unsigned int component) const{
	const double alpha = 0.3;
	const double beta = 1;

	return -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
}

/**
 * Returns the vector value of the exact solution at a point on the mesh. Note the first two components are
 * the velocity and the last component is the pressure.
 * @param p A point on the mesh.
 * @param values Data structure to store the solution.
 */
template <int dim>
void ExactSolution<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const{
	Assert(values.size() == dim+1, ExcDimensionMismatch(values.size(), dim+1));
	const double alpha = 0.3;
	const double beta = 1;

	values(0) = alpha*p[1]*p[1]/2 + beta - alpha*p[0]*p[0]/2;
	values(1) = alpha*p[0]*p[1];
	values(2) = -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
}

/**
 * Returns the values of KInverse at points on the mesh
 * @param points A vector of points in the mesh.
 * @param values Data structure to store the values of KInverse at the given points on the mesh.
 */
template <int dim>
void KInverse<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<2,dim>> &values) const{
	Assert(points.size() == values.size(), ExcDimensionMismatch(points.size(), values.size()));

	//Return the identity matrix
	for(unsigned int p=0; p<points.size(); ++p){
		values[p].clear();
		for(unsigned int d=0; d<dim; ++d){
			values[p][d][d] = 1.;
		}
	}
}

/**
 * Default constructor for the InverseMatrix class.
 * @param m Matrix to be used.
 */
template <class MatrixType>
InverseMatrix<MatrixType>::InverseMatrix(const MatrixType &m)
	:
	matrix(&m)
{}

/**
 * Computes the action of the inverse of a matrix on a vector, by solving M dst = src using CG.
 * @param dst Data structure where the action of the inverse on a vector is to be stored.
 * @param src The vector to multiply by the inverse.
 */
template <class MatrixType>
void InverseMatrix<MatrixType>::vmult(Vector<double> &dst, const Vector<double> &src) const{
	SolverControl solver_control(std::max<unsigned int>(src.size(), 200), 1e-8*src.l2_norm());
	SolverCG<> cg(solver_control);

	dst = 0;
	cg.solve(*matrix, dst, src, PreconditionIdentity());
}

/**
 * Default constructor for the SchurComplement class.
 * @param A The matrix to form the Schur complement for.
 * @param Minv Class representing the action of the inverse of the mass matrix.
 */
SchurComplement::SchurComplement(const BlockSparseMatrix<double> &A, const InverseMatrix<SparseMatrix<double>> &Minv)
	:
	system_matrix(&A), m_inverse(&Minv), tmp1(A.block(0,0).m()), tmp2(A.block(0,0).m())
{}

/**
 * Computes the action of the Schur complement on a vector.
 * @param dst Data structure where the action of the Schur complement on a vector is to be stored.
 * @param src The vector to multiply by the Schur complement.
 */
void SchurComplement::vmult(Vector<double> &dst, const Vector<double> &src) const{
	//Upper right
	system_matrix->block(0,1).vmult(tmp1, src);
	//Upper left
	m_inverse->vmult(tmp2, tmp1);
	//Lower left
	system_matrix->block(1,0).vmult(dst, tmp2);
}

/**
 * Default constructor for the ApproximateSchurComplement class.
 * @param A The matrix that the Schur complement was computes for.
 */
ApproximateSchurComplement::ApproximateSchurComplement(const BlockSparseMatrix<double> &A)
	:
	system_matrix(&A), tmp1(A.block(0,0).m()), tmp2(A.block(0,0).m())
{}

/**
 * Computes the action of the the Preconditioned Schur complement on a vector.
 * @param dst Data structure where the action of the Preconditioned Schur complement on a vector is to be stored.
 * @param src The vector to multiply by the Preconditioned Schur complement.
 */
void ApproximateSchurComplement::vmult(Vector<double> &dst, const Vector<double> &src) const{
	//Upper right
	system_matrix->block(0,1).vmult(tmp1, src);
	//diag(M)^-1
	system_matrix->block(0,0).precondition_Jacobi(tmp2, tmp1);
	//Lower left
	system_matrix->block(1,0).vmult(dst, tmp2);
}

/**
 * Constructor for MixedLaplaceProblem, uses Raviart Thomas elements of given degree for the velocity, and Q_degree
 * elements for the pressure.
 * @param degree Given degree of the elements.
 */
template <int dim>
MixedLaplaceProblem<dim>::MixedLaplaceProblem(const unsigned int degree)
	:
	degree(degree), fe(FE_RaviartThomas<dim>(degree), 1, FE_DGQ<dim>(degree), 1), dof_handler(triangulation)
{}

/**
 * Initializes the domain to be the unit square and distributes the DoFs. It then sets up the
 * matrix to block diagonal.
 */
template <int dim>
void MixedLaplaceProblem<dim>::make_grid_and_dofs(){
	GridGenerator::hyper_cube(triangulation, -1, 1);
	triangulation.refine_global(3);

	dof_handler.distribute_dofs(fe);

	//Renumber the dofs so that we have a 2x2 block system
	DoFRenumbering::component_wise(dof_handler);

	std::vector<types::global_dof_index> dofs_per_component(dim+1);
	DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);
	const unsigned int n_u = dofs_per_component[0], n_p = dofs_per_component[dim];

	std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
	std::cout << "Total number of cells: " << triangulation.n_cells() << std::endl;
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " (" << n_u << '+' << n_p << ')' << std::endl;

	//Create and initialize the block sparsity pattern
	BlockDynamicSparsityPattern dsp(2, 2);
	dsp.block(0, 0).reinit(n_u, n_u);
	dsp.block(1, 0).reinit(n_p, n_u);
	dsp.block(0, 1).reinit(n_u, n_p);
	dsp.block(1, 1).reinit(n_p, n_p);
	dsp.collect_sizes();
	DoFTools::make_sparsity_pattern(dof_handler, dsp);

	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit (sparsity_pattern);

	solution.reinit(2);
	solution.block(0).reinit(n_u);
	solution.block(1).reinit(n_p);
	solution.collect_sizes();

	system_rhs.reinit(2);
	system_rhs.block(0).reinit(n_u);
	system_rhs.block(1).reinit(n_p);
	system_rhs.collect_sizes();
}

/**
 * Assembles the BlockMatrix LHS and the vector RHS.
 */
template <int dim>
void MixedLaplaceProblem<dim>::assemble_system(){
	//Use a higher order quadrature scheme than the degree of the elements
	QGauss<dim> quadrature_formula(degree+2);
	QGauss<dim-1> face_quadrature_formula(degree+2);

	FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double> local_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	//Initialize classes to be used
	const RightHandSide<dim> right_hand_side;
	const PressureBoundaryValues<dim> pressure_boundary_values;
	const KInverse<dim> k_inverse;
	//Initialize data structures to be used
	std::vector<double> rhs_values(n_q_points);
	std::vector<double> boundary_values(n_face_q_points);
	std::vector<Tensor<2,dim>> k_inverse_values(n_q_points);
	//Object used to get components of shape functions
	const FEValuesExtractors::Vector velocities(0);
	const FEValuesExtractors::Scalar pressure(dim);

	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
	for(; cell!=endc; ++cell){
		fe_values.reinit(cell);
		local_matrix = 0;
		local_rhs = 0;

		//Get RHS and K^-1 values at each quadrature point for the current element
		right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);
		k_inverse.value_list(fe_values.get_quadrature_points(), k_inverse_values);
		for(unsigned int q=0; q<n_q_points; ++q){
			for(unsigned int i=0; i<dofs_per_cell; ++i){
				//Get the value of the ith shape function (velocity), divergence of the ith space function (velocity),
				//and value of the ith shape function (pressure)
				const Tensor<1,dim> phi_i_u = fe_values[velocities].value(i, q);
				const double div_phi_i_u = fe_values[velocities].divergence(i, q);
				const double phi_i_p = fe_values[pressure].value(i, q);
				//LHS
				for(unsigned int j=0; j<dofs_per_cell; ++j){
					//Same as above but for the jth shape function
					const Tensor<1,dim> phi_j_u = fe_values[velocities].value(j, q);
					const double div_phi_j_u = fe_values[velocities].divergence(j, q);
					const double phi_j_p = fe_values[pressure].value(j, q);
					local_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u - div_phi_i_u * phi_j_p - phi_i_p * div_phi_j_u) * fe_values.JxW(q);
				}
				//RHS
				local_rhs(i) += -phi_i_p * rhs_values[q] * fe_values.JxW(q);
			}
		}
		//Incorporate pressure boundary condition
		for(unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n){
			if(cell->at_boundary(face_n)){
				fe_face_values.reinit(cell, face_n);
				pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values);
				for (unsigned int q=0; q<n_face_q_points; ++q){
					for (unsigned int i=0; i<dofs_per_cell; ++i){
						local_rhs(i) += -(fe_face_values[velocities].value(i, q) * fe_face_values.normal_vector(q) * boundary_values[q] * fe_face_values.JxW(q));
					}
				}
			}
		}

		//Transfer local to global
		cell->get_dof_indices (local_dof_indices);
		for(unsigned int i=0; i<dofs_per_cell; ++i){
			for (unsigned int j=0; j<dofs_per_cell; ++j){
				system_matrix.add(local_dof_indices[i], local_dof_indices[j], local_matrix(i,j));
			}
		}
		for(unsigned int i=0; i<dofs_per_cell; ++i){
			system_rhs(local_dof_indices[i]) += local_rhs(i);
		}
	}
}

/**
 * Solves the saddle point problem using a Schur complement technique.
 */
template <int dim>
void MixedLaplaceProblem<dim>::solve(){
	InverseMatrix<SparseMatrix<double>> inverse_mass(system_matrix.block(0,0));
	Vector<double> tmp(solution.block(0).size());
	//Pressure solve
	{
		//Compute the RHS of the Schur equation
		SchurComplement schur_complement(system_matrix, inverse_mass);
		Vector<double> schur_rhs(solution.block(1).size());
		inverse_mass.vmult(tmp, system_rhs.block(0));
		system_matrix.block(1,0).vmult (schur_rhs, tmp);
		schur_rhs -= system_rhs.block(1);

		SolverControl solver_control(solution.block(1).size(), 1e-12*schur_rhs.l2_norm());
		SolverCG<> cg(solver_control);

		ApproximateSchurComplement approximate_schur (system_matrix);
		InverseMatrix<ApproximateSchurComplement> approximate_inverse(approximate_schur);
		cg.solve(schur_complement, solution.block(1), schur_rhs, approximate_inverse);
		std::cout << solver_control.last_step() << " CG Schur complement iterations to obtain convergence." << std::endl;
	}
	//Velocity solve
	{
		system_matrix.block(0,1).vmult(tmp, solution.block(1));
		tmp *= -1;
		tmp += system_rhs.block(0);
		inverse_mass.vmult(solution.block(0), tmp);
	}
}

/**
 * Compute the L2 error for both the pressure and velocity.
 */
template <int dim>
void MixedLaplaceProblem<dim>::compute_errors() const{
	//Objects used to extract the components of the pressure and velocity respectively
	const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
	const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);

	ExactSolution<dim> exact_solution;
	Vector<double> cellwise_errors(triangulation.n_active_cells());

	QTrapez<1> q_trapez;
	QIterated<dim> quadrature(q_trapez, degree+2);

	//Compute L2 error of the pressure
	VectorTools::integrate_difference(dof_handler, solution, exact_solution, cellwise_errors, quadrature, VectorTools::L2_norm, &pressure_mask);
	const double p_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);

	//Compute L2 error of the velocity
	VectorTools::integrate_difference(dof_handler, solution, exact_solution, cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
	const double u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
	std::cout << "Errors: ||e_p||_L2 = " << p_l2_error << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
}

/**
 * Method to output the solution to a .vtu file.
 */
template <int dim>
void MixedLaplaceProblem<dim>::output_results() const{
	std::vector<std::string> solution_names(dim, "u");
	solution_names.emplace_back("p");
	std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	DataOut<dim> data_out;
	data_out.add_data_vector(dof_handler, solution, solution_names, interpretation);

	data_out.build_patches(degree+1);

	std::ofstream output("solution.vtu");
	data_out.write_vtu(output);
}

/**
 * Runner method for the Step20 class so solve the PDE.
 */
template <int dim>
void MixedLaplaceProblem<dim>::run(){
  make_grid_and_dofs();
  assemble_system();
  solve();
  compute_errors();
  output_results();
}

/**
 * Runner method.
 * @return Unused.
 */
int main (){
	try{
		using namespace dealii;
		using namespace Step20;
		MixedLaplaceProblem<2> mixed_laplace_problem(0);
		mixed_laplace_problem.run();
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


