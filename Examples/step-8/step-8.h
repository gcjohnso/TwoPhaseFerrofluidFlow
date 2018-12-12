/*
 * step-8.h
 *
 *  Created on: Nov 11, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial problem to solve the elastic equation:
 *  	-Partial_j(c_ijkl Partial_k u_l) = f_i,		i = 1, ..., d,	in Omega,
 *  								   u = 0, 						on Boundary Omega,
 *  where Omega is the unit square. Note that the solution to the PDE is vector valued.
 *
 *  This program solves the above PDE on a sequence of adaptively refined meshes.
 */

#ifndef STEP_8_H_
#define STEP_8_H_

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

namespace Step8{

using namespace dealii;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class ElasticProblem{
	public:
		ElasticProblem();
		~ElasticProblem();
		void run();
	private:
		void setup_system();
		void assemble_system();
		void solve();
		void refine_grid();
		void output_results(const unsigned int cycle) const;

		Triangulation<dim> triangulation;
		DoFHandler<dim> dof_handler;
		FESystem<dim> fe;
		ConstraintMatrix hanging_node_constraints;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> system_rhs;
};

}

#endif /* STEP_8_H_ */
