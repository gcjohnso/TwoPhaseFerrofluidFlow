/*
 * step-4.h
 *
 *  Created on: Oct 23, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial problem which solves Laplaces equation in 2D and 3D (Using dimension-less programming)
 *	with non-constant forcing function and non-homogeneous boundary conditions.
 */

#ifndef STEP_4_H_
#define STEP_4_H_

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/function.h>

using namespace dealii;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class Step4{
	public:
		Step4();
		void run();

	private:
		void make_grid();
		void setup_system();
		void assemble_system();
		void solve();
		void output_results() const;

		Triangulation<dim> triangulation;
		FE_Q<dim> fe;
		DoFHandler<dim> dof_handler;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> system_rhs;
};

/**
 * Class which represents the forcing function of the PDE.
 */
template <int dim>
class RightHandSide : public Function<dim>{
	public:
		RightHandSide () : Function<dim>() {};

		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

/**
 * Class which represents the non-homogeneous boundary conditions of the PDE.
 */
template <int dim>
class BoundaryValues : public Function<dim>{
	public:
		BoundaryValues () : Function<dim>() {};

		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

#endif /* STEP_4_H_ */
