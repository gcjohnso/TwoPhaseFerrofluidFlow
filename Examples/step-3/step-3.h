/*
 * step-3.h
 *
 *  Created on: Oct 21, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *	Tutorial program which solves Poissons equation on the unit square with constant RHS (f = 1)
 *	and homogeneous dirichlet boundary conditions.
 */

#ifndef STEP_3_H_
#define STEP_3_H_

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

#define DIM 2

using namespace dealii;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
class Step3{
	public:
		Step3();
		void run();

	private:
		void make_grid(int refinement_level);
		void setup_system();
		void assemble_system();
		void solve();
		void output_results() const;

		Triangulation<DIM> triangulation;
		FE_Q<DIM> fe;
		DoFHandler<DIM> dof_handler;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> system_rhs;
};

#endif /* STEP_3_H_ */
