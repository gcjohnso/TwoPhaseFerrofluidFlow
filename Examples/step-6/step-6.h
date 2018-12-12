/*
 * step-6.h
 *
 *  Created on: Nov 5, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial to solve the elliptic problem
 *  	-grad * (a(x)u(x)) = 1		in Omega,
 *  	                 u = 0		on Boundary of Omega,
 *  where Omega is the unit circle.
 *
 *  The program solves the problem on a sequence of adaptively
 *  refined meshes using the Kelly error estimator.
 */

#ifndef STEP_6_H_
#define STEP_6_H_

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>

using namespace dealii;

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class Step6{
    public:
        Step6();
        ~Step6();
        void run();
    private:
        void setup_system();
        void assemble_system();
        void solve();
        void refine_grid();
        void output_results(const unsigned int cycle) const;
        
        Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler; 
        ConstraintMatrix constraints;
        SparseMatrix<double> system_matrix;
        SparsityPattern sparsity_pattern;
        Vector<double> solution;
        Vector<double> system_rhs;
};

#endif /* STEP_6_H_ */
