 
#ifndef TWO_PHASE_FLUID_SOLVER_H_
#define TWO_PHASE_FLUID_SOLVER_H_

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

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/data_out.h>

//#define PI dealii::numbers::PI
#define PI 3.141592653589793238462643383279502884L

using namespace dealii;

template <int dim>
class PhaseInitialCondition : public Function<dim>{
public:
    PhaseInitialCondition();
    
    virtual double value(const Point<dim> &p, const unsigned int) const;
    
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int) const;
};

/*
 * Class that encapsulates the action of the inverse of a matrix on a vector, i.e.
 * computes y = A^{-1}x by solving Ay=x using CG.
 */
class MatrixInverseAction{
public:
    MatrixInverseAction();
    
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
class CHReducedPhaseMatrixAction{
public:
    CHReducedPhaseMatrixAction();
    
    void initialize(TrilinosWrappers::SparseMatrix &MassMatrix, TrilinosWrappers::SparseMatrix &StiffnessMatrix, TrilinosWrappers::SparseMatrix &CHPhaseMatrix, const double dt, const double eps, const double gamma, const double eta);
    
    void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const;
    
private:
    TrilinosWrappers::SparseMatrix *MassMatrix;
    TrilinosWrappers::SparseMatrix *StiffnessMatrix;
    TrilinosWrappers::SparseMatrix *CHPhaseMatrix;
    
    mutable TrilinosWrappers::MPI::Vector scratch1, scratch2;
    
    MatrixInverseAction MassInverse;
    
    double dt;
    double eps;
    double gamma;
    double eta;
};

template<int dim>
class NSBoundaryValues : public Function<dim>{
public:
    NSBoundaryValues(double *time) : Function<dim>(dim+1) {t = time;}
    virtual double value(const Point<dim> &p, const unsigned int  component = 0) const;
    virtual void vector_value (const Point<dim> &p, Vector<double> &value) const;
private:
    const double alpha = 0.3;
    const double beta  = 1;
    double *t;
};

template<int dim>
class NSVelcoityForcingFunction{
public:
    NSVelcoityForcingFunction(double eps);
    void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values, double t, std::vector<double> localPrevPhaseValue) const;
    
private:
    const double alpha = 0.3;
    const double beta  = 1;
    double eps;
};

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
    const double alpha = 0.3;
    const double beta  = 1;
    double t;
};

/*
 * Class that encapsulates the action of the inverse of a matrix on a vector, i.e.
 * computes y = A^{-1}x by solving Ay=x using CG.
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



template <int dim>
class TwoPhaseFluidSolver{
public:
    TwoPhaseFluidSolver();
    
    void run();

private:
    
    //Parameters for the PDEs
    /* Interface thickness parameter epsilon. */
    const double eps;
    /* Mobility parameter gamma. */
    const double gamma;
    /* Stability parameter eta. */
    const double eta;
    /* */
    const double mu;
    
    const double lambda;
    
    const double nu_w;
    
    const double nu_f;

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
    /* Constraints object used for adaptive mesh refinement. */
    ConstraintMatrix NSConstraints;
    /* Sparsity pattern */
    SparsityPattern sparsityPattern;
    /* Index set for all indices */
    IndexSet CHAllIndex;
    
    //Variables related to the matrices and RHSs of the discretization
    PhaseInitialCondition<dim> phaseInitialCondition;
    /* Mass matrix */
    TrilinosWrappers::SparseMatrix CHMassMatrix;
    /* Stiffness matrix */
    TrilinosWrappers::SparseMatrix CHStiffnessMatrix;
    /* Constant portion of the Phase matrix, M + dt\frac{\gamma}{\eta}K. */
    TrilinosWrappers::SparseMatrix CHPhaseMatrix;
    /* RHS for the Phase. */
    TrilinosWrappers::MPI::Vector CHPhaseRHS;
    /* Reduced RHS for the Phase. */
    TrilinosWrappers::MPI::Vector CHReducedPhaseRHS;
    /* RHS for the Chemical Potential. */
    TrilinosWrappers::MPI::Vector CHChemicalPotentialRHS;
    /* Reduced RHS for the Chemical Potential. */
    TrilinosWrappers::MPI::Vector CHReducedChemicalPotentialRHS;
    /* Constant portion of the preconditioner for the phase equation */
    TrilinosWrappers::SparseMatrix CHPhasePreconditioner;
    /* Object to compute the action of the reduced phase matrix. */
    CHReducedPhaseMatrixAction CHReducedPhaseMatrix;
    /* Object to compute the action of the inverse of the mass matrix. */
    MatrixInverseAction CHMassInverse;
    /* Scratch vector for CH */
    TrilinosWrappers::MPI::Vector CHScratchVector;
    /* LHS for NS. */
    TrilinosWrappers::BlockSparseMatrix NSMatrix;
    /* Preconditioner for the Schur Complement of NS */
    TrilinosWrappers::BlockSparseMatrix NSSchurComplementPreconditioner;
    /* RHS for NS. */
    TrilinosWrappers::MPI::BlockVector NSRHS;
    
    //Variables related to the solvers
    /* Object containing parameters for the CH solver */
    SolverControl CHSolverControl;
    /* ILU preconditioner for CHPhaseTimeDepPreconditioner */
    TrilinosWrappers::PreconditionILU CHPhaseILUPreconditioner;
    /* */
    SolverControl blockSolverControl;
    /* */
    SchurComplementPreconditioner schurComplementPreconditioner;
    /* AMG Preconditioner for the velocity block of NS */
    std::shared_ptr<TrilinosWrappers::PreconditionAMG> velocityBlockPreconditioner;
    /* Block Preconditioner for NS */
    NSBlockPreconditioner blockPreconditioner;
    
    
    //Variables related to the solution variables
    /* Phase variable at the current time step. */
    TrilinosWrappers::MPI::Vector currPhaseSolution;
    /* Phase variable at the previous time step. */
    TrilinosWrappers::MPI::Vector prevPhaseSolution;
    /* Chemical Potential at the current time step. */
    TrilinosWrappers::MPI::Vector currChemicalPotentialSolution;
    /* Chemical Potential at the current time step. */
    TrilinosWrappers::MPI::Vector prevChemicalPotentialSolution;
    /* NS solution at the current time step. */
    TrilinosWrappers::MPI::BlockVector currNSSolution;
    /* NS solution at the previousPicardIteration. */
    TrilinosWrappers::MPI::BlockVector picardNSSolution;
    /* NS solution at the previous time step. */
    TrilinosWrappers::MPI::BlockVector prevNSSolution;
    
    void initializeSystemAndRefineInitialData();
    
    void initializeNSDoF();
    
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
    
    void processSolution(const unsigned int cycle);
    
    void processNSSolution();
    
    void setupNSDoF();
    
    void setupCHDoF();
    
    double truncatedDoubleWellValue(const double value);
    
    void assembleSchurComplementPreconditioner();
    
    void assembleNSMatrix();
    
    void initializeAMGPreconditioner();
    
    void solveNS();
    
    void outputNSSolution();
};

double Heaviside(double x);

#endif

