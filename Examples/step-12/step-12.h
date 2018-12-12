/*
 * step-12.h
 *
 *  Created on: Nov 15, 2018
 *      Author: gcjohnso@math.umd.edu
 *
 *  Tutorial program which solves the linear advection equation
 *  	grad * (bu) = 0 in Omega,
 *  	u = 1 on the inflow of the boundary,
 *  	u = 0 on the remainder of the boundary,
 *  where Omega is [0,1]^2.
 *
 *  The program solves the PDE using using discontinuous elements
 *  and the MeshWorker framework.
 */

#ifndef STEP_12_H_
#define STEP_12_H_


#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>

namespace Step12{

using namespace dealii;

/**
 * Class used to prescribe the inflow boundary conditions.
 */
template <int dim>
class BoundaryValues: public Function<dim>{
	public:
		BoundaryValues () {}
		virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component=0) const;
};

/**
 * Class which encapsulates all methods and data structures required
 * for solving the PDE.
 */
template <int dim>
class AdvectionProblem{
	public:
		AdvectionProblem();
		void run();
	private:
		void setup_system();
		void assemble_system();
		void solve(Vector<double> &solution);
		void refine_grid();
		void output_results(const unsigned int cycle) const;

		Triangulation<dim> triangulation;
		const MappingQ1<dim> mapping;
		FE_DGQ<dim> fe;
		DoFHandler<dim> dof_handler;
		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double> solution;
		Vector<double> right_hand_side;
		typedef MeshWorker::DoFInfo<dim> DoFInfo;
		typedef MeshWorker::IntegrationInfo<dim> CellInfo;

		static void integrate_cell_term(DoFInfo &dinfo, CellInfo &info);
		static void integrate_boundary_term(DoFInfo &dinfo, CellInfo &info);
		static void integrate_face_term(DoFInfo &dinfo1, DoFInfo &dinfo2, CellInfo &info1, CellInfo &info2);
};

}

#endif /* STEP_12_H_ */
