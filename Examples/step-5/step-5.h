/*
 * step-5.h
 *
 *  Created on: Oct 31, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial to solve the elliptic problem
 *  	-grad * (a(x)u(x)) = 1		in Omega,
 *  	                 u = 0		on Boundary of Omega.
 *
 *  The program reads in a mesh and then solves the problem on a
 *  sequence of globally refined meshes.
 */

#ifndef STEP_5_H_
#define STEP_5_H_

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

using namespace dealii;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class Step5{
	public:
		Step5();
		void run();

	private:
		void setup_system();
		void assemble_system();
		void solve();
		void output_results(const unsigned int cycle) const;

		Triangulation<dim> triangulation;
		FE_Q<dim> fe;
		DoFHandler<dim> dof_handler;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> system_rhs;
};

#endif /* STEP_5_H_ */
