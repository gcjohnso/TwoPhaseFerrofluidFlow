 /*
 * CahnHilliardEquation.h
 *
 *      Author: gcjohnso@math.umd.edu
 *
 *  Program to solve the Navier Stokes system of the ferrofluid model:
 */

#ifndef NAVIER_STOKES_EQUATION_H_
#define NAVIER_STOKES_EQUATION_H_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
//#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/data_out.h>

//#define PI dealii::numbers::PI
#define PI 3.141592653589793238462643383279502884L

using namespace dealii;

/**
 * Class which encapsulates the Boundary Condition for the velocity. Currently the function just returns 0 for each component,
 * since we have implemented no-slip BC. If other dirichlet boundary conditions are to be specified, simple change the value
 * that the function "value" returns.
 */
template<int dim>
class NSBoundaryValues : public Function<dim>{
public:
    NSBoundaryValues(double *time) : Function<dim>(dim+1) {t = time;}
    virtual double value(const Point<dim> &p, const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p, Vector<double> &value) const;
private:
    double *t;
};

/*
 * Class which encapsulates the forcing function for the velocity equation. This method should be modified if a different
 * forcing function is to be used.
 */
template<int dim>
class NSVelcoityForcingFunction{
public:
    NSVelcoityForcingFunction();
    void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values, double t) const;
};

/**
 * Class which encapsulates the forced solution. This classed to form the appropriate forcing function.
 */
template<int dim>
class NSExactSolution : public Function<dim>{
public:
    NSExactSolution(double time) : Function<dim>(dim + 1) {t = time;}
    virtual double value(const Point<dim> &p, const unsigned int component) const;
    virtual void vector_value(const Point<dim> &p, Vector<double> &value) const override;
    virtual Tensor<1,dim> gradient( const Point<dim> &p, const unsigned int component) const;
    Tensor<1,dim> vectorLaplacian(const Point<dim> &p) const;
    Tensor<1,dim> timeDeriv(const Point<dim> &p) const;
    
private:
    double t;
};

/*
 * Class that encapsulates the action of the inverse of a matrix on a vector, i.e.
 * computes y = A^{-1}x by solving Ay=x using CG. The CG solve is preconditioned with an AMG 
 * preconditioner, initialized using the configuration given in Step-31 of the deal.ii tutorial.
 */
class AMGMatrixInverseAction{
public:
    AMGMatrixInverseAction();
    
    void initialize(TrilinosWrappers::SparseMatrix &A);
    
    void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const;    
    
private:
    /* Pointer to matrix to compute the inverse action of. */
    TrilinosWrappers::SparseMatrix *matrix;
    
    mutable SolverControl solverControl;
    
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner;
};

/**
 * Class that encapsulates the LSC preconditioner for the Schur Complement.
 */
class SchurComplementPreconditioner{
public:
    SchurComplementPreconditioner();
    
    void initialize(TrilinosWrappers::BlockSparseMatrix &NSSchurComplementPreconditioner);
    
    void updateVelocityBlock(TrilinosWrappers::SparseMatrix &velocityBlock);
    
    void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const;
    
    void Tvmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const;
    
    AMGMatrixInverseAction inverseDiscreteLaplacian;
    
private:
    
    TrilinosWrappers::BlockSparseMatrix *schurComplementPreconditioner;
    TrilinosWrappers::SparseMatrix pressureLaplacian;
    TrilinosWrappers::SparseMatrix *NSVelocityBlock;
    
    TrilinosWrappers::PreconditionJacobi massDiagonalInverse;
    mutable TrilinosWrappers::MPI::Vector tempV1, tempV2, tempP;
};

/**
 * Class that encapsulates the NS Block Preconditioner used to precondition the GMRES solver. For 
 * specific on the form of the block preconditioner refer to Section 3.3 of the Final.pdf.
 */
class NSBlockPreconditioner{
public:
    NSBlockPreconditioner();
    
    void initialize(TrilinosWrappers::BlockSparseMatrix &matrix,  SchurComplementPreconditioner &schurPre, std::shared_ptr<TrilinosWrappers::PreconditionAMG> velocityBlockPreconditioner);
    
    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const;
    
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> AMGPreconditioner; 
private:
    TrilinosWrappers::BlockSparseMatrix *NSMatrix;
    
    SchurComplementPreconditioner *schurPreconditioner;
    
    mutable TrilinosWrappers::MPI::Vector tempP;
};

/**
 * Class which encapsulates all methods and data structures required
 * for solving the system of PDEs.
 */
template <int dim>
class NavierStokesEquation{
public:
    NavierStokesEquation();
    
    void run();
    
private:
    
    //Parameters for the PDEs
    /*Scaling parameter for the magnetic force.*/
    const double mu;
    /*Scaling parameter for the capillary force*/
    const double lambda;
    /*Interface thickness*/
    const double eps;
    /*Viscosity of the non-magnetic fluid*/
    const double nu_w;
    /*Viscosity of the magnetic fluid*/
    const double nu_f;
    
    //Variables related to time stepping
    /* Total number of time steps. */
    unsigned int numTimeSteps;
    /* Final time. */
    double finalTime;
    /* Time step */
    double dt;
    /* Number of time steps performed. */
    unsigned int currTimeStep;
    /* Current time. */
    double currTime;
    
    //Variables related to the mesh, DOFs, BC, and FE
    /* Triangulation of the mesh */
    Triangulation<dim> triangulation;
    /* DoF object for the Phase and Chemical Potential. */
    DoFHandler<dim> CHDoFHandler;
    /* DoF object for the Velocity and Pressure. */
    DoFHandler<dim> NSDoFHandler;
    /* DoF object for the Magnetization. */
    DoFHandler<dim> MagDoFHandler;
    /* DoF object for the Magnetic Potential. */
    DoFHandler<dim> MagPotDoFHandler;
    /* DoF object for the Applied Magnetic field. */
    DoFHandler<dim> AppliedMagDoFHandler;
    /* Degree of the finite element used. */
    unsigned int degree;
    /* Finite Element object for CH. */
    FESystem<dim> CHFE;
    /* Finite Element object for NS. */
    FESystem<dim> NSFE;
    /* Finite Element object for Magnetization. */
    FESystem<dim> MagFE;
    /* Finite Element object for Magnetic Potential. */
    FESystem<dim> MagPotFE;
    /* Finite Element object for Applied Magnetic field. */
    FESystem<dim> AppliedMagFE;
    /* Constraints object used for adaptive mesh refinement. */
    ConstraintMatrix NSConstraints;
    
    
    /* LHS for NS. */
    TrilinosWrappers::BlockSparseMatrix NSMatrix;
    /* Block Matrix that has the Mass matrix of velocity, B, and B^T of NS */
    TrilinosWrappers::BlockSparseMatrix NSSchurComplementPreconditioner;
    /* LSC Preconditioner for the Schur Complement*/
    SchurComplementPreconditioner schurComplementPreconditioner;
    /* AMG Preconditioner for the velocity block of NS */
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> velocityBlockPreconditioner;
    /* Block Preconditioner for NS */
    NSBlockPreconditioner blockPreconditioner;
    /* RHS for NS. */
    TrilinosWrappers::MPI::BlockVector NSRHS;
    /* Object containing parameters for the NS solver */
    SolverControl blockSolverControl;
    
    //Variables related to the solution variables
    /* NS solution at the current time step. */
    TrilinosWrappers::MPI::BlockVector currNSSolution;
    /* NS solution at the previous time step. */
    TrilinosWrappers::MPI::BlockVector prevNSSolution;
    /* Phase variable at the previous time step. */
    TrilinosWrappers::MPI::Vector prevPhaseSolution;
    /* Chemical Potential at the current time step. */
    TrilinosWrappers::MPI::Vector currChemicalPotentialSolution;
    /* Magnetization at the current time step. */
    TrilinosWrappers::MPI::Vector currMagSolution;
    /* Magnetic Potential at the current time step. */
    TrilinosWrappers::MPI::Vector currMagPotSolution;
    
    ConvergenceTable convergence_table;
    
    /* All method descriptions are given in the cpp file */
    void initializeNSDoF();
    
    void initializeCHDoF();
    
    void initializeMagDoF();
    
    void initializeMagPotDoF();
    
    void initializeMesh();
    
    void assembleSchurComplementPreconditioner();
    
    void assembleNSMatrix();
    
    void initializeAMGPreconditioner();
    
    void solveNS();
    
    void processSolution(const unsigned int cycle);
    
    void outputSolution();
};

/**
 * Method which computes the Heaviside function as
 * H(x) = 1/(1 + e^(-x)), the definition of this function
 * is given in the paper by Ignacio and Ricardo.
 */
double Heaviside(double x);


#endif
