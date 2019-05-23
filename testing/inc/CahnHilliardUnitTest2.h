/*
 * CahnHilliardEquation.h
 *
 *      Author: gcjohnso@math.umd.edu
 *
 *  Program to solve the Cahn Hilliard system of the ferrofluid model:
 *		Partial_t \theta + \div(u \theta) + \gamma \lap \psi = 0,			
 *           \psi - \eps\lap\theta + \frac{1}{\eps}f(\theta) = 0, 
 *  with zero flux boundary conditions
 *          Partial_\eta \theta = Partial_\eta \psi = 0.
 */

#ifndef CAHN_HILLIARD_UNIT_TEST_2_H_
#define CAHN_HILLIARD_UNIT_TEST_2_H_ 

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
#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/data_out.h>

#define PI dealii::numbers::PI

using namespace dealii;

/**
 * Class which encapsulates the Initial Condition for the CH system.
 */
template <int dim>
class PhaseInitialCondition2 : public Function<dim>{
public:
    PhaseInitialCondition2();
    
    virtual double value(const Point<dim> &p, const unsigned int) const;
};

/**
 * Class which encapsulates the forcing function for the phase equation, i.e. the first equation.
 */
template <int dim>
class PhaseForcingFunction2 : public Function<dim>{
public:
    PhaseForcingFunction2() : Function<dim>(){}
    
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

/**
 * Class which encapsulates the forcing function for the chemical potential equation, i.e. the second equation.
 */
template <int dim>
class ChemicalPotentialForcingFunction2 : public Function<dim>{
public:
    ChemicalPotentialForcingFunction2() : Function<dim>(){}
    
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

/*
 * Class that encapsulates the action of the inverse of a matrix on a vector, i.e.
 * computes y = A^{-1}x by solving Ay=x using CG.
 */
class MatrixInverseAction2{
public:
    MatrixInverseAction2();
    
    void initialize(TrilinosWrappers::SparseMatrix &A);
    
    void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src);    
    
private:
    /* Pointer to matrix to compute the inverse action of. */
    TrilinosWrappers::SparseMatrix *matrix;
    
    mutable SolverControl solverControl;
    
    TrilinosWrappers::PreconditionILU preconditioner ;
};

/**
 * Class that encapsulates the action of the CH reduced phase matrix on a vector, i.e.
 * M + \dt \frac{\gamma}{\eta} K + dt\gamma\eps K M^{-1}K.
 * This is used for the GMRES solver.
 */
class CHReducedPhaseMatrixAction2{
public:
    CHReducedPhaseMatrixAction2();
    
    void initialize(TrilinosWrappers::SparseMatrix &MassMatrix, TrilinosWrappers::SparseMatrix &StiffnessMatrix, TrilinosWrappers::SparseMatrix &CHPhaseMatrix, const double dt, const double eps, const double gamma, const double eta);
    
    void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const;
    
private:
    TrilinosWrappers::SparseMatrix *MassMatrix;
    TrilinosWrappers::SparseMatrix *StiffnessMatrix;
    TrilinosWrappers::SparseMatrix *CHPhaseMatrix;
    
    mutable TrilinosWrappers::MPI::Vector scratch1, scratch2;
    
    MatrixInverseAction2 MassInverse;
    
    double dt;
    double eps;
    double gamma;
    double eta;
};

/**
 * Class which encapsulates all methods and data structures required
 * for solving the system of PDEs.
 */
template <int dim>
class CahnHilliardUnitTest2{
public:
    CahnHilliardUnitTest2();
    
    void run();
    
private:
    //Parameters for the PDEs
    /* Interface thickness parameter epsilon. */
    const double eps;
    /* Mobility parameter gamma. */
    const double gamma;
    /* Stability parameter eta. */
    const double eta;
    
    //Variables related to the mesh refinement
    /* Maximum number of refinement steps for the intial data */
    unsigned int maxInitialDataRefinement;
    /* Minimum number of levels of refinement for the mesh. */
    unsigned int minMeshRefinement;
    /* Maximum number of levels of refinement for the mesh. */
    unsigned int maxMeshRefinement;
    /* Vector which stores the value of the estimator in each cell */
    Vector<float> estimatorValues;
    
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
    /* DoF object for the Velocity. */
    DoFHandler<dim> NSDoFHandler;
    /* Degree of the finite element used. */
    unsigned int degree;
    /* Finite Element object for CH. */
    FESystem<dim> CHFE;
    /* Finite Element object for NS. */
    FESystem<dim> NSFE;
    /* Constraints matrix used for adaptive mesh refinement. */
    ConstraintMatrix CHConstraints;
    /* Sparsity pattern */
    SparsityPattern sparsityPattern;
    /* Index set for all indices */
    IndexSet CHAllIndex;
    
    //Variables related to the matrices and RHSs of the discretization
    PhaseInitialCondition2<dim> phaseInitialCondition;
    /* Mass matrix */
    TrilinosWrappers::SparseMatrix CHMassMatrix;
    /* Stiffness matrix */
    TrilinosWrappers::SparseMatrix CHStiffnessMatrix;
    /* The Phase matrix, M + dt\frac{\gamma}{\eta}K. */
    TrilinosWrappers::SparseMatrix CHPhaseMatrix;
    /* RHS for the Phase. */
    TrilinosWrappers::MPI::Vector CHPhaseRHS;
    /* Reduced RHS for the Phase. */
    TrilinosWrappers::MPI::Vector CHReducedPhaseRHS;
    /* RHS for the Chemical Potential. */
    TrilinosWrappers::MPI::Vector CHChemicalPotentialRHS;
    /* Reduced RHS for the Chemical Potential. */
    TrilinosWrappers::MPI::Vector CHReducedChemicalPotentialRHS;
    /* Preconditioner for the phase equation */
    TrilinosWrappers::SparseMatrix CHPhasePreconditioner;
    /* Object to compute the action of the reduced phase matrix. */
    CHReducedPhaseMatrixAction2 CHReducedPhaseMatrix;
    /* Object to compute the action of the inverse of the mass matrix. */
    MatrixInverseAction2 CHMassInverse;
    /* Scratch vector for CH */
    TrilinosWrappers::MPI::Vector CHScratchVector;
    
    //Variables related to the solvers
    /* Object containing parameters for the CH solver */
    SolverControl CHSolverControl;
    /* ILU preconditioner for CHPhasePreconditioner */
    TrilinosWrappers::PreconditionILU CHPhaseILUPreconditioner;
    
    //Variables related to the solution variables
    /* Phase variable at the current time step. */
    TrilinosWrappers::MPI::Vector currPhaseSolution;
    /* Phase variable at the previous time step. */
    TrilinosWrappers::MPI::Vector prevPhaseSolution;
    /* Chemical Potential at the current time step. */
    TrilinosWrappers::MPI::Vector currChemicalPotentialSolution;
    /* Chemical Potential at the current time step. */
    TrilinosWrappers::MPI::Vector prevChemicalPotentialSolution;
    /* NS solution at the previous time step. */
    TrilinosWrappers::MPI::BlockVector prevNSSolution;
    
    /* All method descriptions are given in the cpp file */
    void initializeSystemAndRefineInitialData();
    
    void initializeNSDoFs();
    
    void initializeMesh();
    
    void assembleCHMassAndStiffnessMatrices();
    
    void createCHPhaseMatrixAndPreconditioner();
    
    void assembleCHRHS();
    
    void createCHPhaseReducedRHS();
    
    void solveCHPhase();
    
    void createCHChemicalPotentialReducedRHS();
    
    void outputPhase();
    
    void performAdaptiveRefinement();
    
    void computeEstimator();
    
    void setupNSDoF();
    
    void setupCHDoF();
    
    double truncatedDoubleWellValue(const double value);
};

#endif
