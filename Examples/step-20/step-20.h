/*
 * step-20.h
 *
 *  Created on: Nov 18, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial program to solve Stokes eq:
 *  	K^-1 u + grad p = 0		in Omega,
 *  	         -div u = -f    in Omega,
 *  	              p = f		on Boundary Omega,
 * 	where Omega is the unit square, K is takes as the
 * 	identity for simplicity, and f & g are chosen s.t.
 * 	p = -(alpha/2 xy^2 + beta x - alpha/6 x^3),
 * 	u = [alpha/2y^2 + beta - alpha/2x^2 ; alpha xy],
 * 	are the exact solution.
 *
 * 	The program solves the saddle point problem using
 * 	a Schur complement technique.
 */

#ifndef STEP_20_H_
#define STEP_20_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/block_sparse_matrix.h>

namespace Step20{

using namespace dealii;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class MixedLaplaceProblem{
	public:
		MixedLaplaceProblem(const unsigned int degree);
		void run();
	private:
		void make_grid_and_dofs();
		void assemble_system();
		void solve();
		void compute_errors() const;
		void output_results() const;

		const unsigned int degree;
		Triangulation<dim> triangulation;
		FESystem<dim> fe;
		DoFHandler<dim> dof_handler;
		BlockSparsityPattern sparsity_pattern;
		BlockSparseMatrix<double> system_matrix;
		BlockVector<double> solution;
		BlockVector<double> system_rhs;
};

/**
 * Class used to represent the right hand side of the system.
 */
template <int dim>
class RightHandSide : public Function<dim>{
	public:
		RightHandSide() : Function<dim>(1) {}
		virtual double value(const Point<dim> &p, const unsigned int  component = 0) const;
};

/**
 * Class used to represent the boundary condition on the pressure.
 */
template <int dim>
class PressureBoundaryValues : public Function<dim>{
	public:
		PressureBoundaryValues() : Function<dim>(1) {}
		virtual double value(const Point<dim> &p, const unsigned int  component = 0) const;
};

/**
 * Class used to represent the exact solution.
 */
template <int dim>
class ExactSolution : public Function<dim>{
	public:
		ExactSolution() : Function<dim>(dim+1) {}
		virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;
};

/**
 * Class used to represent the inverse of K.
 */
template <int dim>
class KInverse : public TensorFunction<2,dim>{
	public:
		KInverse() : TensorFunction<2,dim>() {}
		virtual void value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<2,dim>> &values) const;
};

/**
 * Class used to represent the action of the inverse of a matrix on a vector.
 */
template <class MatrixType>
class InverseMatrix : public Subscriptor{
	public:
		InverseMatrix(const MatrixType &m);
		void vmult(Vector<double> &dst, const Vector<double> &src) const;
	private:
		const SmartPointer<const MatrixType> matrix;
};

/**
 * Class used to represent the action of the action of the Schur complement on a vector.
 */
class SchurComplement : public Subscriptor{
	public:
		SchurComplement(const BlockSparseMatrix<double> &A, const InverseMatrix<SparseMatrix<double>> &Minv);
		void vmult(Vector<double> &dst, const Vector<double> &src) const;
	private:
		const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
		const SmartPointer<const InverseMatrix<SparseMatrix<double>>> m_inverse;
		mutable Vector<double> tmp1, tmp2;
};

/**
 * Class used to represent the preconditioner for the Schur complement matrix.
 */
class ApproximateSchurComplement : public Subscriptor{
	public:
		ApproximateSchurComplement(const BlockSparseMatrix<double> &A);
		void vmult (Vector<double> &dst, const Vector<double> &src) const;
	private:
		const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
		mutable Vector<double> tmp1, tmp2;
};

}

#endif /* STEP_20_H_ */
