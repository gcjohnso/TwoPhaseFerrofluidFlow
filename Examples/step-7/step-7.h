/*
 * step-7.h
 *
 *  Created on: Nov 8, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial problem to solve the Helmholtz equation
 *  	-Delta u + u = f		in Omega,
 *  	           u = g_1		on Gamma_1,
 *  	  n * grad u = g_2		on Gamma_2,
 * 	where Omega is the unit square. The boundary conditions
 * 	and forcing function are chosen such that
 * 	u = Sum_1^3 exp(-abs(x-x_i)^2/sigma^2)
 * 	is a solution.
 *
 * 	The program solves the problem on a sequence of globally
 * 	refined meshes and a sequence of adaptively refined meshes.
 * 	It then computes the error in various norms for both sequences.
 */

#ifndef STEP_7_H_
#define STEP_7_H_

#include <deal.II/grid/tria.h>
#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>

namespace Step7{

using namespace dealii;

/**
 * Base class used to define common quantities of the exact solution.
 */
template <int dim>
class SolutionBase{
	protected:
		static const unsigned int n_source_centers = 3;
		static const Point<dim> source_centers[n_source_centers];
		static const double width;
};

/**
 * Specialization of SolutionBase for d=1.
 */
template <>
const Point<1>
SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] = { Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0) };

/**
 * Specialization of SolutionBase for d=2.
 */
template <>
const Point<2>
SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] = { Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5) };

template <int dim>
const double SolutionBase<dim>::width = 1./8.;

/**
 * Class used to represent the exact solution.
 */
template <int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>{
	public:
		Solution() : Function<dim>() {}
		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
		virtual Tensor<1,dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;
};

/**
 * Class used the represent the forcing function f.
 */
template <int dim>
class RightHandSide : public Function<dim>, protected SolutionBase<dim>{
	public:
		RightHandSide() : Function<dim>() {}
		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class HelmholtzProblem{
	public:
		enum RefinementMode{ global_refinement, adaptive_refinement };
		HelmholtzProblem(const FiniteElement<dim> &fe, const RefinementMode refinement_mode);
		~HelmholtzProblem();
		void run();
	private:
		void setup_system();
		void assemble_system();
		void solve ();
		void refine_grid();
		void process_solution(const unsigned int cycle);

		Triangulation<dim> triangulation;
		DoFHandler<dim> dof_handler;
		SmartPointer<const FiniteElement<dim>> fe;
		ConstraintMatrix hanging_node_constraints;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> system_rhs;
		const RefinementMode refinement_mode;
		ConvergenceTable convergence_table;
};

}


#endif /* STEP_7_H_ */
