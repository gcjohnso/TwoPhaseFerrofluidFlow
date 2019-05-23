#include "../inc/TwoPhaseFluidSolver.h"
using namespace dealii;

template<int dim>
TwoPhaseFluidSolver<dim>::TwoPhaseFluidSolver()
:
mu(1),
lambda(.05),
eps(0.01),
gamma(.0002),
eta(.000001),//.95*eps
nu_w(1.0),
nu_f(2.0),
maxInitialDataRefinement(30),
minMeshRefinement(1),
maxMeshRefinement(6),
degree(2),
CHFE(FE_Q<dim>(degree), 1),
NSFE(FE_Q<dim>(degree), dim, FE_Q<dim>(degree-1), 1),
phaseInitialCondition()
{
}

/**
 * Method which creates the grid, initializes the DoFs, and refines the mesh to resolve the inital data.
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::initializeSystemAndRefineInitialData(){
    CHDoFHandler.clear();
    triangulation.clear();
    
    initializeMesh();

    CHDoFHandler.initialize(triangulation, CHFE);
    //Begin the refinement procedure
    for(unsigned int i = 0; i < maxInitialDataRefinement; i++){
        if(i == 0){
            //Associate the DoFs with the FEs used
            CHDoFHandler.initialize(triangulation, CHFE);
        }else{
            CHDoFHandler.distribute_dofs(CHFE);
        }
        
        unsigned int CHNumOfDoFs = CHDoFHandler.n_dofs();
        {
            CHConstraints.clear();
            DoFTools::make_hanging_node_constraints(CHDoFHandler, CHConstraints);
            CHConstraints.close();
            
            DynamicSparsityPattern dynamicSparsityPattern;
            dynamicSparsityPattern.reinit(CHNumOfDoFs, CHNumOfDoFs);
            DoFTools::make_sparsity_pattern(CHDoFHandler, dynamicSparsityPattern, CHConstraints, false);
        }

        CHAllIndex.clear();
        CHAllIndex.set_size(CHNumOfDoFs);
        CHAllIndex.add_range(0, CHNumOfDoFs);
        currPhaseSolution.reinit(CHAllIndex);
        prevPhaseSolution.reinit(CHAllIndex);
        
        //Interpolate the IC onto the mesh
        VectorTools::interpolate(CHDoFHandler, phaseInitialCondition, currPhaseSolution);
        CHConstraints.distribute(currPhaseSolution);

        //Compute the estimator on the IC and then refine the mesh accordingly
        computeEstimator();
        GridRefinement::refine_and_coarsen_fixed_fraction (triangulation, estimatorValues, 0.55, 0.05);
    
        //Clear the refine flag if the cell has been refined over the maximum number
        if (triangulation.n_levels() > maxMeshRefinement){
            for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(maxMeshRefinement) ; cell != triangulation.end(); ++cell){
                cell->clear_refine_flag();
            }
        }
        
        //Transfer the solution from the old mesh to the new mesh
        TrilinosWrappers::MPI::Vector oldMeshCH = currPhaseSolution;
        SolutionTransfer<dim,TrilinosWrappers::MPI::Vector> CHTransfer(CHDoFHandler);
        triangulation.prepare_coarsening_and_refinement();
        CHTransfer.prepare_for_coarsening_and_refinement(oldMeshCH);
        triangulation.execute_coarsening_and_refinement();
    }
    //Now that the triangulation has been adequately refined, setup the DoFs
    initializeNSDoF();
    setupCHDoF();
    
    //Finally, reinterpolate the IC one last time onto the refined mesh
    VectorTools::interpolate(CHDoFHandler, phaseInitialCondition, currPhaseSolution);
    CHConstraints.distribute(currPhaseSolution);
    
    prevPhaseSolution = currPhaseSolution;
    
    outputPhase();
}

template<int dim>
void TwoPhaseFluidSolver<dim>::initializeNSDoF(){
    //Navier-Stokes
    NSDoFHandler.initialize(triangulation, NSFE);
    DoFRenumbering::Cuthill_McKee(NSDoFHandler);
    DoFRenumbering::component_wise(NSDoFHandler);
    
    std::vector<unsigned int> DoFsPerComponent(dim+1);
    DoFTools::count_dofs_per_component(NSDoFHandler, DoFsPerComponent);  
    unsigned int VelocityNumOfDoFs = dim * DoFsPerComponent[0];
    unsigned int PressureNumOfDoFs = DoFsPerComponent[dim];
    
    NSMatrix.clear();
    NSSchurComplementPreconditioner.clear();
    
    {
        NSConstraints.clear();
        DoFTools::make_hanging_node_constraints(NSDoFHandler, NSConstraints);
        
        FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(NSDoFHandler, 0, NSBoundaryValues<dim>(&currTime), NSConstraints, NSFE.component_mask(velocities));
        NSConstraints.close();
    }
    
    {    
        BlockDynamicSparsityPattern dsp(2,2);
        dsp.block(0,0).reinit(VelocityNumOfDoFs, VelocityNumOfDoFs);
        dsp.block(1,0).reinit(PressureNumOfDoFs, VelocityNumOfDoFs);
        dsp.block(0,1).reinit(VelocityNumOfDoFs, PressureNumOfDoFs);
        dsp.block(1,1).reinit(PressureNumOfDoFs, PressureNumOfDoFs);
        dsp.collect_sizes();
        
        DoFTools::make_sparsity_pattern(NSDoFHandler, dsp, NSConstraints, false);
        NSMatrix.reinit(dsp);
        NSSchurComplementPreconditioner.reinit(dsp);
    }
    
    IndexSet VelocityAllIndex(VelocityNumOfDoFs);
    VelocityAllIndex.add_range(0, VelocityNumOfDoFs);
    IndexSet PressureAllIndex(PressureNumOfDoFs);
    PressureAllIndex.add_range(0, PressureNumOfDoFs);
    currNSSolution.reinit(2);
    currNSSolution.block(0).reinit(VelocityAllIndex);
    currNSSolution.block(1).reinit(PressureAllIndex);
    currNSSolution.collect_sizes();
    picardNSSolution.reinit(2);
    picardNSSolution.block(0).reinit(VelocityAllIndex);
    picardNSSolution.block(1).reinit(PressureAllIndex);
    picardNSSolution.collect_sizes();
    prevNSSolution.reinit(2);
    prevNSSolution.block(0).reinit(VelocityAllIndex);
    prevNSSolution.block(1).reinit(PressureAllIndex);
    prevNSSolution.collect_sizes();
    NSRHS.reinit(2);
    NSRHS.block(0).reinit(VelocityAllIndex);
    NSRHS.block(1).reinit(PressureAllIndex);
    NSRHS.collect_sizes();
}

template<int dim>
void TwoPhaseFluidSolver<dim>::setupCHDoF(){
    CHDoFHandler.distribute_dofs(CHFE);
    DoFRenumbering::Cuthill_McKee(CHDoFHandler);
    
    unsigned int CHNumOfDoFs = CHDoFHandler.n_dofs();

    //Handle hanging nodes and generate sparsity pattern (Done in {} to make the variables go out of scope)
    {
        CHMassMatrix.clear();
        CHStiffnessMatrix.clear();
        CHPhaseMatrix.clear();
        
        CHConstraints.clear();
        DoFTools::make_hanging_node_constraints (CHDoFHandler, CHConstraints);
        CHConstraints.close();
    
        SparsityPattern CHSparsityPattern;
        DynamicSparsityPattern dynamicSparsityPattern;
        dynamicSparsityPattern.reinit(CHNumOfDoFs, CHNumOfDoFs);
        DoFTools::make_sparsity_pattern(CHDoFHandler, dynamicSparsityPattern, CHConstraints, false);
        CHSparsityPattern.copy_from(dynamicSparsityPattern);
        
        CHMassMatrix.reinit(CHSparsityPattern);
        CHStiffnessMatrix.reinit(CHSparsityPattern);
        CHPhaseMatrix.reinit(CHSparsityPattern);
    }
    
    //Initialize all vectors
    CHAllIndex.clear();
    CHAllIndex.set_size(CHNumOfDoFs);
    CHAllIndex.add_range(0, CHNumOfDoFs);
    currPhaseSolution.reinit(CHAllIndex);
    prevPhaseSolution.reinit(CHAllIndex);
    currChemicalPotentialSolution.reinit(CHAllIndex);
    prevChemicalPotentialSolution.reinit(CHAllIndex);
    CHPhaseRHS.reinit(CHAllIndex);
    CHReducedPhaseRHS.reinit(CHAllIndex);
    CHChemicalPotentialRHS.reinit(CHAllIndex);
    CHReducedChemicalPotentialRHS.reinit(CHAllIndex);
    CHScratchVector.reinit(CHAllIndex);
}

template<int dim>
void TwoPhaseFluidSolver<dim>::setupNSDoF(){
    NSDoFHandler.distribute_dofs(NSFE);
    DoFRenumbering::Cuthill_McKee(NSDoFHandler);
    DoFRenumbering::component_wise(NSDoFHandler);
    
    std::vector<unsigned int> DoFsPerComponent(dim+1);
    DoFTools::count_dofs_per_component(NSDoFHandler, DoFsPerComponent);  
    unsigned int VelocityNumOfDoFs = dim * DoFsPerComponent[0];
    unsigned int PressureNumOfDoFs = DoFsPerComponent[dim];
    
    NSMatrix.clear();
    NSSchurComplementPreconditioner.clear();
    
    {
        NSConstraints.clear();
        DoFTools::make_hanging_node_constraints(NSDoFHandler, NSConstraints);
        
        FEValuesExtractors::Vector velocities(0);
        VectorTools::interpolate_boundary_values(NSDoFHandler, 0, NSBoundaryValues<dim>(&currTime), NSConstraints, NSFE.component_mask(velocities));
        NSConstraints.close();
    }
    
    {        
        BlockDynamicSparsityPattern dsp(2,2);
        dsp.block(0,0).reinit(VelocityNumOfDoFs, VelocityNumOfDoFs);
        dsp.block(1,0).reinit(PressureNumOfDoFs, VelocityNumOfDoFs);
        dsp.block(0,1).reinit(VelocityNumOfDoFs, PressureNumOfDoFs);
        dsp.block(1,1).reinit(PressureNumOfDoFs, PressureNumOfDoFs);
        dsp.collect_sizes();
        
        DoFTools::make_sparsity_pattern(NSDoFHandler, dsp, NSConstraints, false);
        NSMatrix.reinit(dsp);
        NSSchurComplementPreconditioner.reinit(dsp);
    }
    
    //Initialize all vectors
    IndexSet VelocityAllIndex(VelocityNumOfDoFs);
    VelocityAllIndex.add_range(0, VelocityNumOfDoFs);
    IndexSet PressureAllIndex(PressureNumOfDoFs);
    PressureAllIndex.add_range(0, PressureNumOfDoFs);
    currNSSolution.reinit(2);
    currNSSolution.block(0).reinit(VelocityAllIndex);
    currNSSolution.block(1).reinit(PressureAllIndex);
    currNSSolution.collect_sizes();
    picardNSSolution.reinit(2);
    picardNSSolution.block(0).reinit(VelocityAllIndex);
    picardNSSolution.block(1).reinit(PressureAllIndex);
    picardNSSolution.collect_sizes();
    prevNSSolution.reinit(2);
    prevNSSolution.block(0).reinit(VelocityAllIndex);
    prevNSSolution.block(1).reinit(PressureAllIndex);
    prevNSSolution.collect_sizes();
    NSRHS.reinit(2);
    NSRHS.block(0).reinit(VelocityAllIndex);
    NSRHS.block(1).reinit(PressureAllIndex);
    NSRHS.collect_sizes();
}

/**
 * Assembles the Mass and Stiffness matrices for the Cahn Hilliard equation.
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::assembleCHMassAndStiffnessMatrices(){
    //Set all entries to be 0
    CHMassMatrix = 0;
    CHStiffnessMatrix = 0;
    
    QGauss<dim> quadratureFormula(degree + 2);
    
    FEValues<dim> CHFEValues(CHFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    const unsigned int CHDoFPerCell = CHFE.dofs_per_cell;
    const unsigned int numQuadraturePoints = quadratureFormula.size();
    
    FullMatrix<double> CHLocalMassMatrix(CHDoFPerCell, CHDoFPerCell);
    FullMatrix<double> CHLocalStiffnessMatrix(CHDoFPerCell, CHDoFPerCell);
    
    std::vector<types::global_dof_index> CHLocalDoFIndices(CHDoFPerCell);
    
    FEValuesExtractors::Scalar phaseExtractor(0);
    std::vector<double> shapeValue(CHDoFPerCell);
    std::vector<Tensor<1, dim>> shapeGradient(CHDoFPerCell);
    
    typename DoFHandler<dim>::active_cell_iterator cell = CHDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator endCell = CHDoFHandler.end();

    for(; cell != endCell; ++cell){
        CHFEValues.reinit(cell);
        CHLocalMassMatrix = 0;
        CHLocalStiffnessMatrix = 0;
        for(unsigned int q = 0; q < numQuadraturePoints; ++q){
            //Preload shape functions gradient and shapeValue
            for(unsigned int i = 0; i < CHDoFPerCell; ++i){
                shapeValue[i] = CHFEValues[phaseExtractor].value(i, q);
                shapeGradient[i] = CHFEValues[phaseExtractor].gradient(i, q);
            }
            //Compute local contributions of mass and stiffness matrices
            for(unsigned int i = 0; i < CHDoFPerCell; ++i){
                for(unsigned int j = 0; j < CHDoFPerCell; ++j){
                    CHLocalMassMatrix(i, j) += shapeValue[i] * shapeValue[j] * CHFEValues.JxW(q);
                    CHLocalStiffnessMatrix(i, j) += shapeGradient[i] * shapeGradient[j] * CHFEValues.JxW(q);
                }
            }
        }
        
        cell->get_dof_indices(CHLocalDoFIndices);
        CHConstraints.distribute_local_to_global(CHLocalMassMatrix, CHLocalDoFIndices, CHMassMatrix);
        CHConstraints.distribute_local_to_global(CHLocalStiffnessMatrix, CHLocalDoFIndices, CHStiffnessMatrix);
    }
}

/*
 * Creates the preconditioner for the phase matrix
 * 
 * M + \dt \frac{\gamma}{\eta} K + dt\gamma\eps K M^{-1}K.
 * 
 * The preconditioner is simply substituting M^{-1} by diag(M)^{-1}.
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::createCHPhaseMatrixAndPreconditioner(){
    CHPhaseMatrix = 0;
    CHPhasePreconditioner = 0;
    
    CHPhaseMatrix.copy_from(CHMassMatrix);
    CHPhaseMatrix.add(dt*gamma/eta, CHStiffnessMatrix);
    
    TrilinosWrappers::MPI::Vector CHMassInverseDiagonal(CHAllIndex);
    for(unsigned int i = 0; i < CHMassMatrix.m(); ++i){
        CHMassInverseDiagonal[i] = 1.0 / CHMassMatrix.diag_element(i);
    }
    
    //Computes Kdiag(M)^{-1}K
    CHStiffnessMatrix.mmult(CHPhasePreconditioner, CHStiffnessMatrix, CHMassInverseDiagonal);
    CHPhasePreconditioner *= dt*gamma*eps;
    CHPhasePreconditioner.add(1.0, CHPhaseMatrix);
    
    CHMassInverse.initialize(CHMassMatrix);
    
    CHReducedPhaseMatrix.initialize(CHMassMatrix, CHStiffnessMatrix, CHPhaseMatrix, dt, eps, gamma, eta);
    
    CHPhaseILUPreconditioner.initialize(CHPhasePreconditioner);
}

/**
 * Assembles the RHS for the Cahn Hilliard equation.
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::assembleCHRHS(){
    CHPhaseRHS = 0;
    CHChemicalPotentialRHS = 0;
    
    QGauss<dim> quadratureFormula(degree + 2);
    
    FEValues<dim> CHFEValues(CHFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> NSFEValues(NSFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    const unsigned int CHDoFPerCell = CHFE.dofs_per_cell;
    const unsigned int numQuadraturePoints = quadratureFormula.size();
    
    Vector<double> CHLocalPhaseRHS(CHDoFPerCell);
    Vector<double> CHLocalChemicalPotentialRHS(CHDoFPerCell);
    
    std::vector<types::global_dof_index> CHLocalDoFIndices(CHDoFPerCell);
    
    FEValuesExtractors::Scalar phaseExtractor(0);
    FEValuesExtractors::Vector velocityExtractor(0);
    
    std::vector<double> shapeValue(CHDoFPerCell);
    std::vector<Tensor<1, dim>> shapeGradient(CHDoFPerCell);
    std::vector<double> localprevPhaseValue(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localprevVelocityValue(numQuadraturePoints);
    
    typename DoFHandler<dim>::active_cell_iterator CHCell = CHDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator CHEndCell = CHDoFHandler.end();
    typename DoFHandler<dim>::active_cell_iterator NSCell = NSDoFHandler.begin_active();
    
    for(; CHCell != CHEndCell; ++CHCell, ++NSCell){        
        CHFEValues.reinit(CHCell);
        NSFEValues.reinit(NSCell);
        
        CHLocalPhaseRHS = 0;
        CHLocalChemicalPotentialRHS = 0;
        
        CHFEValues[phaseExtractor].get_function_values(prevPhaseSolution, localprevPhaseValue);
        NSFEValues[velocityExtractor].get_function_values(prevNSSolution, localprevVelocityValue);
        
        for(unsigned int q = 0; q < numQuadraturePoints; ++q){
            for(unsigned int i = 0; i < CHDoFPerCell; ++i){
                shapeValue[i] = CHFEValues[phaseExtractor].value(i, q);
                shapeGradient[i] = CHFEValues[phaseExtractor].gradient(i, q);
            }
            for(unsigned int i = 0; i < CHDoFPerCell; ++i){                
                CHLocalPhaseRHS(i) += (localprevPhaseValue[q] * shapeValue[i]
                                     + dt * localprevPhaseValue[q] * localprevVelocityValue[q] * shapeGradient[i]
                ) * CHFEValues.JxW(q);
                
                
                CHLocalChemicalPotentialRHS(i) += ( -(1.0/eps) * truncatedDoubleWellValue(localprevPhaseValue[q])
                                                    + (1.0/eta) * localprevPhaseValue[q]) * shapeValue[i] * CHFEValues.JxW(q);
            }
        }
        CHCell->get_dof_indices(CHLocalDoFIndices);
        CHConstraints.distribute_local_to_global(CHLocalPhaseRHS, CHLocalDoFIndices, CHPhaseRHS);
        CHConstraints.distribute_local_to_global(CHLocalChemicalPotentialRHS, CHLocalDoFIndices, CHChemicalPotentialRHS);
    }
}

/*
 * Computes the reduced RHS for the phase equation
 * f + dt\gamma K M^{-1} CHReducedPhaseRHS
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::createCHPhaseReducedRHS(){
    CHReducedPhaseRHS = 0;
    CHScratchVector = 0;
    
    //Computes dt \gamma KM^{-1} CHChemicalPotentialRHS
    CHMassInverse.vmult(CHScratchVector, CHChemicalPotentialRHS);
    CHStiffnessMatrix.vmult(CHReducedPhaseRHS, CHScratchVector);
    CHReducedPhaseRHS *= dt*gamma;
    
    std::cout << "L2-norm of CHReduced before " << CHReducedPhaseRHS.l2_norm() << std::endl;
    std::cout << "L2-norm of CHRHS " << CHPhaseRHS.l2_norm() << std::endl;;
    
    CHReducedPhaseRHS += CHPhaseRHS;
    std::cout << "L2-norm of CHReduced after " << CHReducedPhaseRHS.l2_norm() << std::endl;
}

/*
 * Solve the system using CG with an ILU preconditioner.
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::solveCHPhase(){
    CHSolverControl.set_max_steps(5000);
    CHSolverControl.set_tolerance(1e-6 * CHReducedPhaseRHS.l2_norm());
    
    //Construct the GMRES Solver
    GrowingVectorMemory<TrilinosWrappers::MPI::Vector> vectorMemory ;
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData  GMRESData ;
    GMRESData.max_n_tmp_vectors = 300 ;
    SolverGMRES<TrilinosWrappers::MPI::Vector> GMRES(CHSolverControl, vectorMemory, GMRESData);
    
    GMRES.solve(CHReducedPhaseMatrix, currPhaseSolution, CHReducedPhaseRHS, CHPhaseILUPreconditioner);
    
    CHConstraints.distribute(currPhaseSolution);
}

/*
 * Computes the reduced RHS for the chemical potential equation
 * g - (\eps K + \frac{1}{\eta}M)currPhaseSolution
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::createCHChemicalPotentialReducedRHS(){
    CHReducedChemicalPotentialRHS = CHChemicalPotentialRHS;
    CHScratchVector = 0;
    
    CHMassMatrix.vmult(CHScratchVector, currPhaseSolution);
    CHReducedChemicalPotentialRHS.add(-(1.0/eta), CHScratchVector);
    
    CHStiffnessMatrix.vmult(CHScratchVector, currPhaseSolution);
    CHReducedChemicalPotentialRHS.add(-eps, CHScratchVector);
}


template<int dim>
void TwoPhaseFluidSolver<dim>::assembleSchurComplementPreconditioner(){
    NSSchurComplementPreconditioner = 0;
    
    QGauss<dim> quadratureFormula(degree + 2);
    
    FEValues<dim> NSFEValues(NSFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    const unsigned int NSDoFPerCell = NSFE.dofs_per_cell;
    const unsigned int numQuadraturePoints = quadratureFormula.size();
    
    FullMatrix<double> NSLocalSchurComplementPreconditioner(NSDoFPerCell, NSDoFPerCell);
    
    std::vector<types::global_dof_index> NSLocalDoFIndices(NSDoFPerCell);
    
    FEValuesExtractors::Vector velocityExtractor(0);
    FEValuesExtractors::Scalar pressureExtractor(dim);
    
    std::vector<Tensor<1, dim>> velocityShapeValue(NSDoFPerCell);
    std::vector<double> velocityShapeDivergence(NSDoFPerCell);
    std::vector<double> pressureShapeValue(NSDoFPerCell);
    
    typename DoFHandler<dim>::active_cell_iterator NSCell = NSDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator NSEndCell = NSDoFHandler.end();
    
    for(; NSCell != NSEndCell; ++NSCell){
        NSFEValues.reinit(NSCell);
        
        NSLocalSchurComplementPreconditioner = 0;
        
        for(unsigned int q = 0; q < numQuadraturePoints; ++q){
            for(unsigned int i = 0; i < NSDoFPerCell; ++i){
                velocityShapeValue[i] = NSFEValues[velocityExtractor].value(i, q);
                velocityShapeDivergence[i] = NSFEValues[velocityExtractor].divergence(i, q);
                pressureShapeValue[i] = NSFEValues[pressureExtractor].value(i, q);
            }
            for(unsigned int i = 0; i < NSDoFPerCell; ++i){
                for(unsigned int j = 0; j < NSDoFPerCell; ++j){
                    NSLocalSchurComplementPreconditioner(i, j) += (velocityShapeValue[j] * velocityShapeValue[i]
                                                                 - dt * pressureShapeValue[j] * velocityShapeDivergence[i]
                                                                 - dt * velocityShapeDivergence[j] * pressureShapeValue[i]) * NSFEValues.JxW(q);
                }
            }
        }
        NSCell->get_dof_indices(NSLocalDoFIndices);
        NSConstraints.distribute_local_to_global(NSLocalSchurComplementPreconditioner, NSLocalDoFIndices, NSSchurComplementPreconditioner);
    }
}

template<int dim>
void TwoPhaseFluidSolver<dim>::assembleNSMatrix(){
    NSMatrix = 0;
    NSRHS = 0;
    
    QGauss<dim> quadratureFormula(degree + 2);
    
    FEValues<dim> NSFEValues(NSFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> CHFEValues(CHFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    const unsigned int NSDoFPerCell = NSFE.dofs_per_cell;
    const unsigned int numQuadraturePoints = quadratureFormula.size();
    
    FullMatrix<double> NSLocalMatrix(NSDoFPerCell, NSDoFPerCell);
    Vector<double> NSLocalRHS(NSDoFPerCell);
    
    std::vector<types::global_dof_index> NSLocalDoFIndices(NSDoFPerCell);
    
    FEValuesExtractors::Vector velocityExtractor(0);
    FEValuesExtractors::Scalar pressureExtractor(dim);
    FEValuesExtractors::Scalar phaseExtractor(0); //Works for both phase and chem pot (As they use the same fe basis)
    
    std::vector<Tensor<1, dim>> velocityShapeValue(NSDoFPerCell);
    std::vector<Tensor<2, dim>> velocityShapeGradientValue(NSDoFPerCell);
    std::vector<SymmetricTensor<2, dim>> velocityShapeSymmetricGradientValue(NSDoFPerCell);
    std::vector<double> velocityShapeDivergence(NSDoFPerCell);
    std::vector<double> pressureShapeValue(NSDoFPerCell);
    
    std::vector<Tensor<1, dim>> localPrevVelocityValue(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localPicardVelocityValue(numQuadraturePoints);
    std::vector<double>         localPicardVelocityDivergence(numQuadraturePoints);
    std::vector<double>         localPrevPhaseValue(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localCurrChemPotGradient(numQuadraturePoints);  
    
    const NSVelcoityForcingFunction<dim> NSVelocityForcing(eps);
    std::vector<Tensor<1, dim>> velocityForcingValue(numQuadraturePoints);
    
    typename DoFHandler<dim>::active_cell_iterator NSCell = NSDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator NSEndCell = NSDoFHandler.end();
    typename DoFHandler<dim>::active_cell_iterator CHCell = CHDoFHandler.begin_active();
    
    for(; NSCell != NSEndCell; ++CHCell, ++NSCell){
        NSFEValues.reinit(NSCell);
        CHFEValues.reinit(CHCell);
        
        NSLocalMatrix = 0;
        NSLocalRHS = 0;
        
        NSFEValues[velocityExtractor].get_function_values(prevNSSolution, localPrevVelocityValue);
        NSFEValues[velocityExtractor].get_function_values(picardNSSolution, localPicardVelocityValue);
        NSFEValues[velocityExtractor].get_function_divergences(picardNSSolution, localPicardVelocityDivergence);
        CHFEValues[phaseExtractor].get_function_values(prevPhaseSolution, localPrevPhaseValue);
        CHFEValues[phaseExtractor].get_function_gradients(currChemicalPotentialSolution, localCurrChemPotGradient);
        
        NSVelocityForcing.vector_value_list(NSFEValues.get_quadrature_points(), velocityForcingValue, currTime, localPrevPhaseValue);
        
        for(unsigned int q = 0; q < numQuadraturePoints; ++q){
            
            for(unsigned int i = 0; i < NSDoFPerCell; ++i){
                velocityShapeValue[i] = NSFEValues[velocityExtractor].value(i, q);
                velocityShapeGradientValue[i] = NSFEValues[velocityExtractor].gradient(i, q);
                velocityShapeSymmetricGradientValue[i] = NSFEValues[velocityExtractor].symmetric_gradient(i, q);
                velocityShapeDivergence[i] = NSFEValues[velocityExtractor].divergence(i, q);
                pressureShapeValue[i] = NSFEValues[pressureExtractor].value(i, q);
            }
            
            for(unsigned int i = 0; i < NSDoFPerCell; ++i){
                for(unsigned int j = 0; j < NSDoFPerCell; ++j){
                    NSLocalMatrix(i, j) += (velocityShapeValue[j] * velocityShapeValue[i]
                                    + dt * .5 * (localPicardVelocityValue[q] * velocityShapeGradientValue[j]) * velocityShapeValue[i] //Swapped order of shape grad and prev vel
                                    + dt * .5 * localPicardVelocityDivergence[q] * velocityShapeValue[i] * velocityShapeValue[j] //Swapped order of shape grad and prev vel
                                    + dt * nu_w *(1.0 + (nu_f - nu_w)*Heaviside(localPrevPhaseValue[q]/eps)) *  velocityShapeSymmetricGradientValue[i] * velocityShapeSymmetricGradientValue[j] //swapped i & j index
                                    - dt * pressureShapeValue[j] * velocityShapeDivergence[i]
                                    - dt * velocityShapeDivergence[j] * pressureShapeValue[i]) * NSFEValues.JxW(q);
                }
                
                NSLocalRHS(i) += (localPrevVelocityValue[q] * velocityShapeValue[i]
                             + (lambda / eps) * dt * (localPrevPhaseValue[q] * localCurrChemPotGradient[q]) * velocityShapeValue[i]
                             + dt * velocityShapeValue[i] * velocityForcingValue[q]) * NSFEValues.JxW(q);
            }
        }
        NSCell->get_dof_indices(NSLocalDoFIndices);
        NSConstraints.distribute_local_to_global(NSLocalMatrix, NSLocalRHS, NSLocalDoFIndices, NSMatrix, NSRHS);
    }
}

template<int dim>
void TwoPhaseFluidSolver<dim>::initializeAMGPreconditioner(){
    //Initialize the AMG preconditioner (Just like in Step-31)
    velocityBlockPreconditioner = std::shared_ptr<TrilinosWrappers::PreconditionAMG> (new TrilinosWrappers::PreconditionAMG());
    
    std::vector<std::vector<bool>> constantModes;
    FEValuesExtractors::Vector velocityExtractor(0) ;
    DoFTools::extract_constant_modes(NSDoFHandler, NSFE.component_mask(velocityExtractor), constantModes);
    
    TrilinosWrappers::PreconditionAMG::AdditionalData AMGData;
    AMGData.constant_modes = constantModes;
    AMGData.elliptic = true;
    AMGData.higher_order_elements = true;
    AMGData.smoother_sweeps = 2;
    AMGData.aggregation_threshold = 0.02;
    
    velocityBlockPreconditioner->initialize(NSMatrix.block(0, 0), AMGData);
}

template<int dim>
void TwoPhaseFluidSolver<dim>::solveNS(){
    blockPreconditioner.initialize(NSMatrix, schurComplementPreconditioner, velocityBlockPreconditioner);
    
    blockSolverControl.set_max_steps(1000);
    blockSolverControl.set_tolerance(1e-10*NSRHS.l2_norm());
    
    //Construct the GMRES Solver
    GrowingVectorMemory<TrilinosWrappers::MPI::BlockVector> vectorMemory;
    SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData GMRESData;
    GMRESData.max_n_tmp_vectors = 100;
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> GMRES(blockSolverControl, vectorMemory, GMRESData);
    
    
    for(unsigned int i = 0; i < currNSSolution.size(); ++i){
        if(NSConstraints.is_constrained(i)){
            currNSSolution(i) = 0;
        }
    }
    
    GMRES.solve(NSMatrix, currNSSolution, NSRHS, blockPreconditioner);
    
    NSConstraints.distribute(currNSSolution);
    
    std::cout << "   " << blockSolverControl.last_step() << " GMRES iterations for NS Solve." << std::endl;
}

template<int dim>
void TwoPhaseFluidSolver<dim>::performAdaptiveRefinement(){
    computeEstimator();
    
    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation, estimatorValues, 0.55, 0.05);
    
    //Clear the refine flag if the cell has been refined over the maximum number
    if (triangulation.n_levels() > maxMeshRefinement){
        for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(maxMeshRefinement) ; cell != triangulation.end(); ++cell){
            cell->clear_refine_flag();
        }
    }
    
    TrilinosWrappers::MPI::BlockVector oldMeshNS = currNSSolution;
    std::vector<TrilinosWrappers::MPI::Vector> oldMeshCH(2);
    oldMeshCH[0] = currPhaseSolution;
    oldMeshCH[1] = currChemicalPotentialSolution;
    
    SolutionTransfer<dim,TrilinosWrappers::MPI::BlockVector> NSTransfer(NSDoFHandler);
    SolutionTransfer<dim,TrilinosWrappers::MPI::Vector> CHTransfer(CHDoFHandler);
    
    triangulation.prepare_coarsening_and_refinement();
    
    NSTransfer.prepare_for_coarsening_and_refinement(oldMeshNS);
    CHTransfer.prepare_for_coarsening_and_refinement(oldMeshCH);
    
    triangulation.execute_coarsening_and_refinement();
    
    setupCHDoF();
    setupNSDoF();
    
    NSTransfer.interpolate(oldMeshNS, currNSSolution);
    std::vector<TrilinosWrappers::MPI::Vector> scratch(2);
    scratch[0].reinit(currPhaseSolution);
    scratch[1].reinit(currPhaseSolution);
    CHTransfer.interpolate(oldMeshCH, scratch);
    currPhaseSolution = scratch[0];
    currChemicalPotentialSolution = scratch[1];
    
    NSConstraints.distribute(currNSSolution);
    CHConstraints.distribute(currPhaseSolution);
    CHConstraints.distribute(currChemicalPotentialSolution);
}

template<int dim>
void TwoPhaseFluidSolver<dim>::computeEstimator(){
    estimatorValues.reinit(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(CHDoFHandler, QGauss<dim-1>(degree+1), typename FunctionMap<dim>::type(), currPhaseSolution, estimatorValues);
}

template<int dim>
void TwoPhaseFluidSolver<dim>::outputPhase(){
    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(CHDoFHandler);
    
    dataOut.add_data_vector(currPhaseSolution, "phase");
    
    dataOut.build_patches();
    
    std::ostringstream filename;
    filename << "/home/werdho/src/UMD/AMSC664/TwoPhaseFluidFlow/output/CH-" << Utilities::int_to_string(currTimeStep, 5) << ".vtk";
    
    std::ofstream output(filename.str().c_str());
    dataOut.write_vtk(output);
}

template<int dim>
void TwoPhaseFluidSolver<dim>::processNSSolution(){
    const ComponentSelectFunction<dim> pressureMask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocityMask(std::make_pair(0, dim), dim + 1);
    
    Vector<float> differencePerCell(triangulation.n_active_cells());
    
    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature(q_trapez, degree + 2);
    
    VectorTools::integrate_difference(NSDoFHandler, currNSSolution, NSExactSolution<dim>(currTime), differencePerCell, quadrature, VectorTools::L2_norm, &pressureMask);
    const double pressureL2Error = VectorTools::compute_global_error(triangulation, differencePerCell, VectorTools::L2_norm);
    
    std::cout << "pressureL2Error: " << pressureL2Error << std::endl;
    
    VectorTools::integrate_difference(NSDoFHandler, currNSSolution, NSExactSolution<dim>(currTime), differencePerCell, quadrature, VectorTools::L2_norm, &velocityMask);
    const double velocityL2Error = VectorTools::compute_global_error(triangulation, differencePerCell, VectorTools::L2_norm);
    
    VectorTools::integrate_difference(NSDoFHandler, currNSSolution, NSExactSolution<dim>(currTime), differencePerCell, quadrature, VectorTools::H1_seminorm, &velocityMask);
    const double velocityH1Error = VectorTools::compute_global_error(triangulation, differencePerCell, VectorTools::H1_seminorm);
    
    const double H1Error = sqrt(velocityL2Error*velocityL2Error + velocityH1Error*velocityH1Error);
    
    std::cout << "velocityL2Error: " << velocityL2Error << " , velocityH1Error: " << velocityH1Error << std::endl;
}

template<int dim>
void TwoPhaseFluidSolver<dim>::outputNSSolution(){
    std::vector<std::string> stokes_names(dim, "velocity");
    stokes_names.emplace_back("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>stokes_component_interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i = 0; i < dim; ++i)
        stokes_component_interpretation[i] = DataComponentInterpretation::component_is_part_of_vector;
    DataOut<dim> data_out;
    data_out.add_data_vector(NSDoFHandler,
                            currNSSolution,
                            stokes_names,
                            stokes_component_interpretation);
    data_out.build_patches(degree);
    std::ofstream output("/home/werdho/src/UMD/AMSC664/TwoPhaseFluidFlow/output/NS-" +
                        Utilities::int_to_string(currTimeStep, 4) + ".vtk");
    data_out.write_vtk(output);
}

template<int dim>
void TwoPhaseFluidSolver<dim>::run(){
    numTimeSteps = 1000;
    finalTime = 2.0;
    dt = finalTime/numTimeSteps;
    currTimeStep = 1;
    currTime = 0;
    
    initializeSystemAndRefineInitialData();
    
    processNSSolution();
    currTime = dt;
    
    prevNSSolution = 0;
    prevPhaseSolution = currPhaseSolution;
    prevChemicalPotentialSolution = 0;
    currTimeStep++;
    
    bool recomputeConstantMatrices = true;
    
    while(currTimeStep <= numTimeSteps){
        std::cout << "Curr iter" << currTimeStep << std::endl;;
        if(recomputeConstantMatrices){
            //Build CH constant matrices
            assembleCHMassAndStiffnessMatrices();
            createCHPhaseMatrixAndPreconditioner();
            std::cout << "Finished CH" << std::endl;
            //Build NS constant matrices
            assembleSchurComplementPreconditioner();
            std::cout << "Finished Regular Build" << std::endl;
            schurComplementPreconditioner.initialize(NSSchurComplementPreconditioner);
            std::cout << "Finished NS" << std::endl;
            recomputeConstantMatrices = false;
        }
        std::cout << "Finished Const" << std::endl;
        
        bool picardConverged = false;
        unsigned int picardIter = 1;
        while(!picardConverged){
            //Now solve CH
            assembleCHRHS();
            
            createCHPhaseReducedRHS();
            
            solveCHPhase();
            std::cout << "Finished Solve" << std::endl;
            
            //Solve for the Chemical Potential
            createCHChemicalPotentialReducedRHS();
            CHMassInverse.vmult(currChemicalPotentialSolution, CHReducedChemicalPotentialRHS);
            CHConstraints.distribute(currChemicalPotentialSolution);
            
            //Now solve NS
            std::cout << "Begin assembleNSMatrix()" << std::endl;
            if(picardIter == 1){
                picardNSSolution = prevNSSolution;
                assembleNSMatrix();
            }
            schurComplementPreconditioner.updateVelocityBlock(NSMatrix.block(0,0));
            initializeAMGPreconditioner();
            std::cout << "Begin solveNS()" << std::endl;
            solveNS();
            std::cout << "Finished solveNS()" << std::endl;
            picardNSSolution = currNSSolution;
            
            //Check for convergence of picard iteration
            assembleNSMatrix();
            TrilinosWrappers::MPI::BlockVector residual;
            IndexSet VelocityAllIndex(NSMatrix.block(0,0).m());
            VelocityAllIndex.add_range(0, NSMatrix.block(0,0).m());
            IndexSet PressureAllIndex(NSMatrix.block(1,1).m());
            PressureAllIndex.add_range(0, NSMatrix.block(1,1).m());
            residual.reinit(2);
            residual.block(0).reinit(VelocityAllIndex);
            residual.block(1).reinit(PressureAllIndex);
            residual.collect_sizes();
            NSMatrix.vmult(residual, picardNSSolution);
            residual -= NSRHS;
            if(residual.l2_norm() < NSRHS.l2_norm()){
                picardConverged = true;
            }else{
                std::cout << "Failed to converge at iter " << picardIter << ", " << residual.l2_norm()<< " vs " << 1e-5 * NSRHS.l2_norm() << std::endl;
                picardIter++;
                picardConverged = true;
            }
            processNSSolution();
        }
        std::cout << "Finished Picard" << std::endl;
        
        if(currTimeStep % 5 == 0){
            
            
            performAdaptiveRefinement();
            recomputeConstantMatrices = true;
            std::cout << "Finished refine" << std::endl;
        }
        outputPhase();
        outputNSSolution();
        
        currTimeStep++;
        currTime += dt;
        prevPhaseSolution = currPhaseSolution;
        prevChemicalPotentialSolution = currChemicalPotentialSolution;
        prevNSSolution = currNSSolution;
    }
}

double Heaviside(double x){
    return 0.5;//1.0/(1.0 + exp(-x));
}

/**
 * Method used the create the initial mesh. The mesh is the rectangle with
 * Coordinates: (0,0), (0,0.6), (1,0.6), (1,0) with 10 elements in the 
 * x-direction and 6 elements in y-direction.
 */
template<int dim>
void TwoPhaseFluidSolver<dim>::initializeMesh(){
    Point<dim> lowerInner(0.0, 0.0);
	Point<dim> upperOuter(1.0, 1.0);

	std::vector<unsigned int> repetitions;
	repetitions.push_back(10);
	repetitions.push_back(6);
    
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, lowerInner, upperOuter, false);
}

template<int dim>
double TwoPhaseFluidSolver<dim>::truncatedDoubleWellValue(const double value){
    if(value < -1.0){
        return 2.0*(value + 1.0);
    }else if(value >= -1.0 && value <= 1.0){
        return (value*value - 1.0)*value;
    }else{
        return 2.0*(value - 1.0);
    }
}

//Things not in NavierStokesClass
template class TwoPhaseFluidSolver<2>;

template<int dim>
double NSBoundaryValues<dim>::value(const Point<dim> &p, const unsigned int component) const{
    Assert (component < this->n_components, ExcIndexRange (component, 0, this->n_components));
    /*if (component == 0)
        return 0;*/
    /*if(component == 0)
        return alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
    else if(component == 1)
        return alpha * p[0] * p[1];
    else if(component == 2)
        return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] - alpha * p[0] * p[0] * p[0] / 6);*/
    /*if(component == 0){
        return *t * sin(PI * p[0]) * sin(PI*(p[1] + .5));
    }else if(component == 1)
        return *t * cos(PI * p[0]) * cos(PI*(p[1] + .5));
    else if(component == 2)
        return sin(2.0*PI*(p[0] - p[1]) + *t);
    std::cout << "Reached a bad component" << std::endl;
    exit(1);*/
    return 0;
}

template<int dim>
void NSBoundaryValues<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const{
  for(unsigned int c=0; c<this->n_components; ++c)
    values(c) = value(p, c);
}

template<int dim>
NSVelcoityForcingFunction<dim>::NSVelcoityForcingFunction(double eps)
{
    eps = eps;
}

template<int dim>
void NSVelcoityForcingFunction<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values, double t, std::vector<double> localPrevPhaseValue) const{
    NSExactSolution<dim> exactSolu(t);
    double nu = 1 *(1.0 + Heaviside(0));
    for(unsigned int i = 0; i < points.size(); ++i){
        Point p = points[i];
        /*double velocity1 = (alpha/2.0) * pow(p[1], 2) + beta - (alpha/2.0) * pow(p[0], 2);
        double velocity2 = alpha * p[0] * p[1];
        values[i][0] = - (velocity1) 
                       + (velocity1) * (-alpha * p[0]) 
                       + (velocity2) * (alpha * p[1]);
        values[i][1] = - velocity2
                       + (velocity1) * (alpha * p[1])
                       + (velocity2) * (alpha * p[0]);*/
        /*double velocity1 = t * sin(PI * p[0]) * sin(PI*(p[1] + .5));
        double velocity2 = t * cos(PI * p[0]) * cos(PI*(p[1] + .5));
        values[i][0] = sin(PI * p[0]) * sin(PI*(p[1] + .5))
                     + PI * PI * velocity1
                     + nu * velocity1 * PI * t * cos(PI * p[0]) * sin(PI*(p[1] + .5)) + nu * velocity2 * PI * t * sin(PI * p[0]) * cos(PI*(p[1] + .5))
                     + 2 * PI * cos(2* PI * (p[0] - p[1]) + t);
        values[i][1] = cos(PI * p[0]) * cos(PI*(p[1] + .5))
                     + PI * PI * velocity2
                     - nu * velocity1 * PI * t * sin(PI * p[0]) * cos(PI*(p[1] + .5)) - nu * velocity2 * PI * t * cos(PI * p[0]) * sin(PI*(p[1] + .5))
                     - 2 * PI * cos(2* PI * (p[0] - p[1]) + t);*/
        /*double velocity1 = sin(PI * p[0]) * sin(PI*(p[1] + .5));
        double velocity2 = cos(PI * p[0]) * cos(PI*(p[1] + .5));
        values[i][0] = PI * PI * velocity1
                     + velocity1 * PI * cos(PI * p[0]) * sin(PI*(p[1] + .5)) + velocity2 * PI * sin(PI * p[0]) * cos(PI*(p[1] + .5))
                     + 2 * PI * cos(2* PI * (p[0] - p[1]));
        values[i][1] = PI * PI * velocity2
                     - velocity1 * PI * sin(PI * p[0]) * cos(PI*(p[1] + .5)) - velocity2 * PI * cos(PI * p[0]) * sin(PI*(p[1] + .5))
                     - 2 * PI * cos(2* PI * (p[0] - p[1]));*/
        //std::cout << "p " << p << " tensor " << values[i] << std::endl;
        Tensor<1, dim> U;
        U[0] = exactSolu.value(p, 0);
        U[1] = exactSolu.value(p, 1);
        Tensor<1, dim> Ut = exactSolu.timeDeriv(p);
        Tensor<2, dim> gradU;
        Tensor<1, dim> gradUx = exactSolu.gradient(p, 0);
        Tensor<1, dim> gradUy = exactSolu.gradient(p, 1);
        gradU[0][0] = gradUx[0];
        gradU[0][1] = gradUx[1];
        gradU[1][0] = gradUy[0];
        gradU[1][1] = gradUy[1];
        
        values[i] = Ut - .5 * nu * exactSolu.vectorLaplacian(p) + (U * gradU) + exactSolu.gradient(p, 2);
    }
}


template<int dim>
double NSExactSolution<dim>::value(const Point<dim> &p, const unsigned int component) const{
    const double x = p[0];
    const double y = p[1];
    if(component == 0){
        //return -t * 4 * PI * sin(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * cos(2*PI*y);
        return -t * 2 * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y);
    }else if(component == 1){
        //return t * 4 * PI * cos(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * sin(2*PI*y);
        return t * 2 * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y);
    }else{
        return sin(2.0*PI*(x - y) + t);
    }
}

template<int dim>
void NSExactSolution<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const{
    Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
    const double x = p[0];
    const double y = p[1];
    /*
    values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
    values(1) = alpha * p[0] * p[1];
    values(2) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] - alpha * p[0] * p[0] * p[0] / 6);*/
    /*values(0) = sin(time) * sin(PI * p[0]) * sin(PI*(p[1] + .5));
    values(1) = sin(time) * cos(PI * p[0]) * cos(PI*(p[1] + .5));
    values(2) = sin(2.0*PI*(p[0] - p[1]) + time);*/
    /*values(0) = sin(PI * p[0]) * sin(PI*(p[1] + .5));
    values(1) = cos(PI * p[0]) * cos(PI*(p[1] + .5));
    values(2) = sin(2.0*PI*(p[0] - p[1]));*/
    /*
    values(0) = -t * 4 * PI * sin(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * cos(2*PI*y);
    values(1) = t * 4 * PI * cos(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * sin(2*PI*y);
    */
    values(0) = -t * 2 * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y);
    values(1) = t * 2 * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y);
    values(2) = sin(2.0*PI*(x - y) + t);
}

template<int dim>
Tensor<1,dim> NSExactSolution<dim>::gradient(const Point<dim> &p, const unsigned int component) const{
    Tensor<1,dim> returnValue;
    const double x = p[0];
    const double y = p[1];
    if(component == 0){
        /*
        returnValue[0] = -t * 4 * PI * PI * sin(4*PI*x) * sin(4*PI*y);
        returnValue[1] = -t * 8 * PI * PI * sin(2*PI*x) * sin(2*PI*x) * cos(4*PI*y);
        */
        returnValue[0] = t * - PI * PI * sin(2*PI*x) * sin(2*PI*y);
        returnValue[1] = -t * 2 * PI * PI * sin(PI*x) * sin(PI*x) * cos(2*PI*y);
    }else if(component == 1){
        /*
        returnValue[0] = t * 8 * PI * PI * cos(4*PI*x) * sin(4*PI*y) * sin(4*PI*y);
        returnValue[1] = t * 4 * PI * PI * sin(4*PI*x) * sin(4*PI*y);
        */
        returnValue[0] = t * 2 * PI * PI * cos(2*PI*x) * sin(PI*y) * sin(PI*y);
        returnValue[1] = t * PI * PI * sin(2*PI*x) * sin(2*PI*y);
    }else{
        returnValue[0] = 2 * PI * cos(2.0*PI*(p[0] - p[1]) + t);
        returnValue[1] = -2 * PI * cos(2.0*PI*(p[0] - p[1]) + t);
    }
    return returnValue;
}

template<int dim>
Tensor<1,dim> NSExactSolution<dim>::vectorLaplacian(const Point<dim> &p) const{
    Tensor<1,dim> returnValue;
    const double x = p[0];
    const double y = p[1];
    /*
    returnValue[0] = t * (32 * PI * PI * PI * (sin(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * cos(2*PI*y) - cos(2*PI*x) * cos(2*PI*x) * sin(2*PI*y) * cos(2*PI*y)) + 64 * PI * PI * PI * sin(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * cos(2*PI*y));
    returnValue[1] = t * (32 * PI * PI * PI * (sin(2*PI*x) * cos(2*PI*x) * cos(2*PI*y) * cos(2*PI*y) - sin(2*PI*x) * cos(2*PI*x) * sin(2*PI*y) * sin(2*PI*y)) - 64 * PI * PI * PI * sin(2*PI*x) * cos(2*PI*x) * sin(2*PI*y) * sin(2*PI*y));
    */
    returnValue[0] = t * (4 * PI * PI * PI * sin(PI*y) * cos(PI*y) * (sin(PI*x) * sin(PI*x) - cos(PI*x) * cos(PI*x)) + 8 * PI * PI * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y));
    returnValue[1] = t * (-8 * PI * PI * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y) + 4 * PI * PI * PI * sin(PI*x) * cos(PI*x) * (cos(PI*y) * cos(PI*y) - sin(PI*y) * sin(PI*y)));
    
    return returnValue;
}

template<int dim>
Tensor<1,dim> NSExactSolution<dim>::timeDeriv(const Point<dim> &p) const{
    Tensor<1,dim> returnValue;
    const double x = p[0];
    const double y = p[1];
    returnValue[0] = -2 * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y);
    returnValue[1] = 2 * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y);
    return returnValue;
}

SchurComplementPreconditioner::SchurComplementPreconditioner()
{
}

void SchurComplementPreconditioner::initialize(TrilinosWrappers::BlockSparseMatrix &NSSchurComplementPreconditioner){
    schurComplementPreconditioner = &NSSchurComplementPreconditioner;
    
    IndexSet allVelocityIndex(NSSchurComplementPreconditioner.block(0,0).m());
    allVelocityIndex.add_range(0, NSSchurComplementPreconditioner.block(0,0).m());
    TrilinosWrappers::MPI::Vector diagMassInverse(allVelocityIndex);
    for(unsigned int i = 0; i < NSSchurComplementPreconditioner.block(0,0).m(); ++i){
        diagMassInverse[i] = 1.0 / NSSchurComplementPreconditioner.block(0,0).diag_element(i);
    }
    
    NSSchurComplementPreconditioner.block(1,0).mmult(pressureLaplacian, NSSchurComplementPreconditioner.block(0,1), diagMassInverse);
    
    inverseDiscreteLaplacian.initialize(pressureLaplacian);
    massDiagonalInverse.initialize(schurComplementPreconditioner->block(0,0));
    
    tempV1.reinit(allVelocityIndex);
    tempV2.reinit(allVelocityIndex);
    IndexSet allPressureIndex(NSSchurComplementPreconditioner.block(1,1).m());
    allPressureIndex.add_range(0, NSSchurComplementPreconditioner.block(1,1).m());
    tempP.reinit(allPressureIndex);
    tempV1 = 0;
    tempV2 = 0;
    tempP = 0;
}

void SchurComplementPreconditioner::updateVelocityBlock(TrilinosWrappers::SparseMatrix &velocityBlock){
    NSVelocityBlock = &velocityBlock;
}

void SchurComplementPreconditioner::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    inverseDiscreteLaplacian.vmult(tempP, src);
    
    schurComplementPreconditioner->block(0, 1).vmult(tempV1, tempP);
    massDiagonalInverse.vmult(tempV2, tempV1);
    NSVelocityBlock->vmult(tempV1, tempV2);
    massDiagonalInverse.vmult(tempV2, tempV1);
    schurComplementPreconditioner->block(1, 0).vmult(tempP, tempV2);
    
    inverseDiscreteLaplacian.vmult(dst, tempP);
}

void SchurComplementPreconditioner::Tvmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    std::cout << "Tvmult called" << std::endl;
    inverseDiscreteLaplacian.vmult(tempP, src);
    
    schurComplementPreconditioner->block(0, 1).vmult(tempV1, tempP);
    massDiagonalInverse.vmult(tempV2, tempV1);
    NSVelocityBlock->Tvmult(tempV1, tempV2);
    massDiagonalInverse.vmult(tempV2, tempV1);
    schurComplementPreconditioner->block(1, 0).vmult(tempP, tempV2);
    
    inverseDiscreteLaplacian.vmult(dst, tempP);
}

NSBlockPreconditioner::NSBlockPreconditioner()
{
}

void NSBlockPreconditioner::initialize(TrilinosWrappers::BlockSparseMatrix &matrix,  SchurComplementPreconditioner &schurPre, std::shared_ptr<TrilinosWrappers::PreconditionAMG> velocityBlockPreconditioner){
    NSMatrix = &matrix;
    schurPreconditioner = &schurPre;
    AMGPreconditioner = velocityBlockPreconditioner;
                                    
    IndexSet allIndex(matrix.block(1,1).m());
    allIndex.add_range(0, matrix.block(1,1).m());
    tempP.reinit(allIndex);
    tempP = 0;
}

void NSBlockPreconditioner::vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const{
    AMGPreconditioner->vmult(dst.block(0), src.block(0));
    
    NSMatrix->block(1,0).vmult(tempP, dst.block(0));
    
    tempP -= src.block(1);
    
    schurPreconditioner->vmult(dst.block(1), tempP);
}


AMGMatrixInverseAction::AMGMatrixInverseAction()
{
}

void AMGMatrixInverseAction::initialize(TrilinosWrappers::SparseMatrix &A){
    matrix = &A;

    preconditioner.reset();
    
    //Initialize the AMG preconditioner (Just like in Step-31)
    preconditioner = std::shared_ptr<TrilinosWrappers::PreconditionAMG> (new TrilinosWrappers::PreconditionAMG());
    
    TrilinosWrappers::PreconditionAMG::AdditionalData AMGData;
    AMGData.elliptic = true;
    AMGData.higher_order_elements = true;
    AMGData.smoother_sweeps = 2;
    AMGData.aggregation_threshold = 0.02;
    AMGData.n_cycles = 1;
    
    preconditioner->initialize(*matrix, AMGData);
    
    solverControl.set_max_steps(1500);
}

void AMGMatrixInverseAction::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    solverControl.set_tolerance(1e-6*src.l2_norm());
    
    SolverCG<TrilinosWrappers::MPI::Vector> CGSolver(solverControl);
    
    CGSolver.solve(*matrix, dst, src, *preconditioner);
}

template<int dim>
PhaseInitialCondition<dim>::PhaseInitialCondition() : Function<dim>()
{
}

template<int dim>
double PhaseInitialCondition<dim>::value(const Point<dim> &p, const unsigned int) const{
    if(p[1] > .5)
        return -1.0;
    else
        return 1.0;
    /*double dis = sqrt(pow(p[0] - .5, 2) + pow(p[1] - .3, 2));
    if(dis < .2)
        return -1.0;
    else
        return 1.0;*/
    //return cos(2*PI*p[0])*cos(2*PI*p[1]);
}

template<int dim>
Tensor<1, dim> PhaseInitialCondition<dim>::gradient(const Point<dim> &p, const unsigned int) const{
    Tensor<1, dim> gradient;
    gradient[0] = 0;
    gradient[1] = 0;
    return gradient;
}

MatrixInverseAction::MatrixInverseAction()
{
}

void MatrixInverseAction::initialize(TrilinosWrappers::SparseMatrix &A){
    matrix = &A;
    
    preconditioner.initialize(*matrix);
    
    solverControl.set_max_steps(500);
}

void MatrixInverseAction::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src){
    solverControl.set_tolerance(1e-8*src.l2_norm());
    
    SolverCG<TrilinosWrappers::MPI::Vector> CGSolver(solverControl);
    
    CGSolver.solve(*matrix, dst, src, preconditioner);
}

CHReducedPhaseMatrixAction::CHReducedPhaseMatrixAction()
{
}

void CHReducedPhaseMatrixAction::initialize(TrilinosWrappers::SparseMatrix &MassMatrix, TrilinosWrappers::SparseMatrix &StiffnessMatrix, TrilinosWrappers::SparseMatrix &CHPhaseMatrix, const double dt, const double eps, const double gamma, const double eta){
    this->MassMatrix = &MassMatrix;
    this->StiffnessMatrix = &StiffnessMatrix;
    this->CHPhaseMatrix = &CHPhaseMatrix;
    
    IndexSet allIndex(MassMatrix.m());
    allIndex.add_range(0, MassMatrix.m());
    scratch1.reinit(allIndex);
    scratch2.reinit(allIndex);
    
    MassInverse.initialize(MassMatrix);
    
    this->dt = dt;
    this->eps = eps;
    this->gamma = gamma;
    this->eta = eta;
}

void CHReducedPhaseMatrixAction::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    //Compute KM^{-1}K*src
    StiffnessMatrix->vmult(scratch1, src);
    MassMatrix->vmult(scratch2, scratch1);
    StiffnessMatrix->vmult(dst, scratch2);
    dst *= dt*gamma*eps;
    
    CHPhaseMatrix->vmult(scratch1, src);
    dst += scratch1;
}



