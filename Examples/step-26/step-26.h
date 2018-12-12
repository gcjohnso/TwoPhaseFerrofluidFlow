/*
 * step-26.h
 *
 *  Created on: Nov 22, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial program to solve the heat equation:
 *		Partial_t u - Delta u = f			on Omega*(0,1]
 *					   u(*,0) = u_0(*)		on Omega
 *					   		u = g			on Boundary Omega*(0,1],
 *	where Omega is the unit square.
 *
 *	The program solves the above PDE using a theta-scheme for time
 *	and Q_d degree elements in space. The mesh is adaptively
 *	refined/coarsened every 5 time iterations and the solution is transferred
 *	between the mesh after it is refined/coarsened.
 */

#ifndef STEP_26_H_
#define STEP_26_H_

#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

namespace Step26{

using namespace dealii;

static const double PI = 3.14159265358979323846;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class HeatEquation{
	public:
		HeatEquation(int degree, int theta);
		void run();
	private:
		void setup_system();
		void solve_time_step();
		void output_results() const;
		void refine_mesh(const unsigned int min_grid_level, const unsigned int max_grid_level);

		Triangulation<dim> triangulation;
		FE_Q<dim> fe;
		DoFHandler<dim> dof_handler;
		ConstraintMatrix constraints;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> mass_matrix;
		SparseMatrix<double> laplace_matrix;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> old_solution;
		Vector<double> system_rhs;
		double time;
		double time_step;
		unsigned int timestep_number;
		const double theta;
		ConvergenceTable convergence_table;
};

/**
 * Class used to represent the right hand side of the system.
 */
template <int dim>
class RightHandSide : public Function<dim>{
	public:
		RightHandSide();
		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

/**
 * Class used to represent the Boundary conditions of the system.
 */
template <int dim>
class BoundaryValues : public Function<dim>{
	public:
		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

}

#endif /* STEP_26_H_ */
