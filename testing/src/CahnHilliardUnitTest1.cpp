#include "../inc/CahnHilliardUnitTest1.h"
using namespace dealii;

/**
 * Constructor which initializes the model parameters, parameters related to mesh refinement, and the finite elements used.
 */
template<int dim>
CahnHilliardUnitTest1<dim>::CahnHilliardUnitTest1()
:
eps(0.001),
gamma(.2),
eta(.000001),
numTimeSteps(1000),
finalTime(2.0),
maxInitialDataRefinement(20),
minMeshRefinement(1),
maxMeshRefinement(5),
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
void CahnHilliardUnitTest1<dim>::initializeSystemAndRefineInitialData(){
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
    //Now that the triangulation has been adequently refined, setup the DoFs
    initializeNSDoFs();
    setupCHDoF();
    
    //Finally, reinterpolate the IC one last time onto the refined mesh
    VectorTools::interpolate(CHDoFHandler, phaseInitialCondition, currPhaseSolution);
    CHConstraints.distribute(currPhaseSolution);
    
    prevPhaseSolution = currPhaseSolution;
    
    outputPhase();
}

/**
 * Method which initializes the DoFs for the NS solution variables. As this is only the solver for CH, this method is essentially just a placeholder
 * and will be eventually replaced by the initializeNSDoFs from the NS solver when everything gets combined.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::initializeNSDoFs(){
    //Navier-Stokes
    NSDoFHandler.initialize(triangulation, NSFE);
    DoFRenumbering::Cuthill_McKee(NSDoFHandler);
    DoFRenumbering::component_wise(NSDoFHandler);
    
    std::vector<unsigned int> DoFsPerComponent(dim+1);
    DoFTools::count_dofs_per_component(NSDoFHandler, DoFsPerComponent);  
    unsigned int VelocityNumOfDoFs = dim * DoFsPerComponent[0];
    unsigned int PressureNumOfDoFs = DoFsPerComponent[dim];
    
    IndexSet VelocityAllIndex(VelocityNumOfDoFs);
    VelocityAllIndex.add_range(0, VelocityNumOfDoFs);
    IndexSet PressureAllIndex(PressureNumOfDoFs);
    PressureAllIndex.add_range(0, PressureNumOfDoFs);
    prevNSSolution.reinit(2);
    prevNSSolution.block(0).reinit(VelocityAllIndex);
    prevNSSolution.block(1).reinit(PressureAllIndex);
    prevNSSolution.collect_sizes();
}

/**
 * Assembles the Mass and Stiffness matrices for the Cahn Hilliard equation.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::assembleCHMassAndStiffnessMatrices(){
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
void CahnHilliardUnitTest1<dim>::createCHPhaseMatrixAndPreconditioner(){
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
    //Then adds M + \dt \frac{\gamma}{\eta} K
    CHPhasePreconditioner.add(1.0, CHPhaseMatrix);
    
    CHMassInverse.initialize(CHMassMatrix);
    
    CHReducedPhaseMatrix.initialize(CHMassMatrix, CHStiffnessMatrix, CHPhaseMatrix, dt, eps, gamma, eta);
    
    CHPhaseILUPreconditioner.initialize(CHPhasePreconditioner);
}

/**
 * Assembles the RHS for the Cahn Hilliard equation.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::assembleCHRHS(){
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
    
    const PhaseForcingFunction1<dim> phaseForcingFunction;
    std::vector<double> phaseForcingValue(numQuadraturePoints);
    const ChemicalPotentialForcingFunction1<dim> chemicalPotentialForcingFunction;
    std::vector<double> chemicalPotentialForcingValue(numQuadraturePoints);
    
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
        
        phaseForcingFunction.value_list(CHFEValues.get_quadrature_points(), phaseForcingValue);
        chemicalPotentialForcingFunction.value_list(CHFEValues.get_quadrature_points(), chemicalPotentialForcingValue);
        
        for(unsigned int q = 0; q < numQuadraturePoints; ++q){
            for(unsigned int i = 0; i < CHDoFPerCell; ++i){
                shapeValue[i] = CHFEValues[phaseExtractor].value(i, q);
                shapeGradient[i] = CHFEValues[phaseExtractor].gradient(i, q);
            }
            for(unsigned int i = 0; i < CHDoFPerCell; ++i){                
                CHLocalPhaseRHS(i) += (localprevPhaseValue[q] * shapeValue[i] 
                                     + localprevVelocityValue[q] * localprevPhaseValue[q] * shapeGradient[i]
                                     + phaseForcingValue[q] * shapeValue[i]) * CHFEValues.JxW(q);
                
                CHLocalChemicalPotentialRHS(i) += ( -(1.0/eps) * truncatedDoubleWellValue(localprevPhaseValue[q])
                                                    + (1.0/eta) * localprevPhaseValue[q]
                                                    + chemicalPotentialForcingValue[q]) * shapeValue[i] * CHFEValues.JxW(q);
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
void CahnHilliardUnitTest1<dim>::createCHPhaseReducedRHS(){
    CHReducedPhaseRHS = 0;
    CHScratchVector = 0;
    
    //Computes dt \gamma KM^{-1} CHChemicalPotentialRHS
    CHMassInverse.vmult(CHScratchVector, CHChemicalPotentialRHS);
    CHStiffnessMatrix.vmult(CHReducedPhaseRHS, CHScratchVector);
    CHReducedPhaseRHS*= dt*gamma;
    
    CHReducedPhaseRHS += CHPhaseRHS;
}

/*
 * Solve the system using GMRES with an ILU preconditioner.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::solveCHPhase(){
    CHSolverControl.set_max_steps(1000);
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
void CahnHilliardUnitTest1<dim>::createCHChemicalPotentialReducedRHS(){
    CHReducedChemicalPotentialRHS = CHChemicalPotentialRHS;
    CHScratchVector = 0;
    
    CHMassMatrix.vmult(CHScratchVector, currPhaseSolution);
    CHReducedChemicalPotentialRHS.add(-(1.0/eta), CHScratchVector);
    
    CHStiffnessMatrix.vmult(CHScratchVector, currPhaseSolution);
    CHReducedChemicalPotentialRHS.add(-eps, CHScratchVector);
}

/*
 * Outputs the solution at the current iteration to a file.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::outputPhase(){
    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(CHDoFHandler);
    
    dataOut.add_data_vector(currPhaseSolution, "phase");
    
    dataOut.build_patches();
    
    std::ostringstream filename;
    filename << "/home/werdho/src/UMD/AMSC664/TwoPhaseFerrofluidFlow/testing/output/CHUnitTest1-" << Utilities::int_to_string(currTimeStep, 5) << ".vtk";
    
    std::ofstream output(filename.str().c_str());
    dataOut.write_vtk(output);
}

/**
 * Performs an iteration of adaptive mesh refinement/coarsening. The current percentage of elements to refine is .55 and the percentage of elements to coarsen is .05, which appear to be
 * generally used values. After elements are marked for coarsening/refinement, each element is checked to ensure that it has not already been refined the maximum allowed times. It then
 * executes the refinement/coarsening and then transfers each solution (both CH and NS) from the old mesh to the new mesh.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::performAdaptiveRefinement(){
    computeEstimator();
    
    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation, estimatorValues, 0.55, 0.05);
    
    //Clear the refine flag if the cell has been refined over the maximum number
    if (triangulation.n_levels() > maxMeshRefinement){
        for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(maxMeshRefinement) ; cell != triangulation.end(); ++cell){
            cell->clear_refine_flag();
        }
    }
    
    TrilinosWrappers::MPI::BlockVector oldMeshNS = prevNSSolution;
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
    
    NSTransfer.interpolate(oldMeshNS, prevNSSolution);
    std::vector<TrilinosWrappers::MPI::Vector> scratch(2);
    scratch[0].reinit(currPhaseSolution);
    scratch[1].reinit(currPhaseSolution);
    CHTransfer.interpolate(oldMeshCH, scratch);
    currPhaseSolution = scratch[0];
    currChemicalPotentialSolution = scratch[1];
    
    CHConstraints.distribute(currPhaseSolution);
    CHConstraints.distribute(currChemicalPotentialSolution);
}

/**
 * This method computes the Kelly error estimator on each element.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::computeEstimator(){
    estimatorValues.reinit(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(CHDoFHandler, QGauss<dim-1>(degree+1), typename FunctionMap<dim>::type(), currPhaseSolution, estimatorValues);
}

/**
 * This method is similar to initializeNSDoFs, as it sets up the DoFs for the NS solution. However, this method is intended to only be called after
 * the mesh has been adaptively refined/coarsened.
 */ 
template<int dim>
void CahnHilliardUnitTest1<dim>::setupNSDoF(){
    NSDoFHandler.distribute_dofs(NSFE);
    
    std::vector<unsigned int> DoFsPerComponent(dim+1);
    DoFTools::count_dofs_per_component(NSDoFHandler, DoFsPerComponent);  
    unsigned int VelocityNumOfDoFs = dim * DoFsPerComponent[0];
    unsigned int PressureNumOfDoFs = DoFsPerComponent[dim];
    
    //Initialize all vectors
    IndexSet VelocityAllIndex(VelocityNumOfDoFs);
    VelocityAllIndex.add_range(0, VelocityNumOfDoFs);
    IndexSet PressureAllIndex(PressureNumOfDoFs);
    PressureAllIndex.add_range(0, PressureNumOfDoFs);
    prevNSSolution.reinit(2);
    prevNSSolution.block(0).reinit(VelocityAllIndex);
    prevNSSolution.block(1).reinit(PressureAllIndex);
    prevNSSolution.collect_sizes();
}

/**
 * This method distributes the DoFs for the CH system and initializes the matrices and vectors used.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::setupCHDoF(){
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

/*
 * Method which controls the solving of the PDE. It iteratively calls to configure the RHS at each time step and then
 * solves for the next value of the solution.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::run(){
    dt = finalTime/numTimeSteps;
    currTimeStep = 0;
    initializeSystemAndRefineInitialData();
    
    prevNSSolution = 0;
    prevPhaseSolution = currPhaseSolution;
    prevChemicalPotentialSolution = 0;
    currTimeStep++;
    
    bool recomputeMatrices = true;
    
    while(currTimeStep <= numTimeSteps){
        std::cout << "Curr iter" << currTimeStep;
        if(recomputeMatrices){
            assembleCHMassAndStiffnessMatrices();
            createCHPhaseMatrixAndPreconditioner();
            recomputeMatrices = false;
        }
        
        assembleCHRHS();
        
        createCHPhaseReducedRHS();
        
        solveCHPhase();
        std::cout << "Finished Solve" << std::endl;
        
        //Solve for the Chemical Potential
        createCHChemicalPotentialReducedRHS();
        CHMassInverse.vmult(currChemicalPotentialSolution, CHReducedChemicalPotentialRHS);
        CHConstraints.distribute(currChemicalPotentialSolution);
        
        if(currTimeStep % 5 == 0){
            outputPhase();
            
            performAdaptiveRefinement();
            recomputeMatrices = true;
            std::cout << "Finished refine" << std::endl;
        }
        
        currTimeStep++;
        currTime += dt;
        prevPhaseSolution = currPhaseSolution;
        prevChemicalPotentialSolution = currChemicalPotentialSolution;
    }
}

/**
 * Method used the create the initial mesh. The mesh is the rectangle with
 * Coordinates: (0,0), (0,0.6), (1,0.6), (1,0) with 10 elements in the 
 * x-direction and 6 elements in y-direction.
 */
template<int dim>
void CahnHilliardUnitTest1<dim>::initializeMesh(){
    Point<dim> lowerInner(0.0, 0.0);
	Point<dim> upperOuter(1.0, 0.6);

	std::vector<unsigned int> repetitions;
	repetitions.push_back(10);
	repetitions.push_back(6);
    
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, lowerInner, upperOuter, false);
}

/**
 * Method which computes the value of the truncated double well at the specified value.
 */
template<int dim>
double CahnHilliardUnitTest1<dim>::truncatedDoubleWellValue(const double value){
    if(value < -1.0){
        return 2.0*(value + 1.0);
    }else if(value >= -1.0 && value <= 1.0){
        return (value*value - 1.0)*value;
    }else{
        return 2.0*(value - 1.0);
    }
}


template class CahnHilliardUnitTest1<2>;
/*
 * Default constructor
 */
template<int dim>
PhaseInitialCondition1<dim>::PhaseInitialCondition1() : Function<dim>()
{
}

/*
 * Function which returns the value of the Phase IC at the specified point p. The second arguement isn't used and doesn't need to be passed when called.
 * This function currently implements the IC to be a flat profile.
 */
template<int dim>
double PhaseInitialCondition1<dim>::value(const Point<dim> &p, const unsigned int) const{
    if(p[1] > 0.06)
        return -1.0;
    else
        return 1.0;
}


/*
 * Function which returns the value of a forcing function at the specified point p for the Phase equation.
 * The second arguement isn't used and doesn't need to be passed when called.
 * This function currently implements the forcing function to be 0.
 */
template<int dim>
double PhaseForcingFunction1<dim>::value(const Point<dim> &p, const unsigned int) const{
    return 0;
}

/*
 * Function which returns the value of a forcing function at the specified point p for the Chemical Potential equation.
 * The second arguement isn't used and doesn't need to be passed when called.
 * This function currently implements the forcing function to be 0.
 */
template<int dim>
double ChemicalPotentialForcingFunction1<dim>::value(const Point<dim> &p, const unsigned int) const{
    return 0;
}

/*
 * Default constructor
 */
MatrixInverseAction1::MatrixInverseAction1()
{
}

/*
 * Function which initializes the MatrixInverseAction object. It creates the preconditioner from the given matrix and sets the maximum number of numTimeSteps
 * that CG will use.
 */
void MatrixInverseAction1::initialize(TrilinosWrappers::SparseMatrix &A){
    matrix = &A;
    
    preconditioner.initialize(*matrix);
    
    solverControl.set_max_steps(500);
}

/*
 * Function which computes the action of the inverse of the matrix on a vector. This is computed by solving Ay=x using CG.
 */
void MatrixInverseAction1::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src){
    solverControl.set_tolerance(1e-8*src.l2_norm());
    
    SolverCG<TrilinosWrappers::MPI::Vector> CGSolver(solverControl);
    
    CGSolver.solve(*matrix, dst, src, preconditioner);
}

/*
 * Default constructor
 */
CHReducedPhaseMatrixAction1::CHReducedPhaseMatrixAction1()
{
}

/**
 * Function which initializes the CHReducedPhaseMatrixAction object. It takes in the Mass, Stiffness, and Phase matrices. It then creates a object to compute the inverse action of the Mass matrix.
 */
void CHReducedPhaseMatrixAction1::initialize(TrilinosWrappers::SparseMatrix &MassMatrix, TrilinosWrappers::SparseMatrix &StiffnessMatrix, TrilinosWrappers::SparseMatrix &CHPhaseMatrix, const double dt, const double eps, const double gamma, const double eta){
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

/**
 * Function which computes the action of the CH reduced phase matrix on a vector. This is done this way so that the matrix never needs to be explicitely computed.
 */
void CHReducedPhaseMatrixAction1::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    //Compute KM^{-1}K*src
    StiffnessMatrix->vmult(scratch1, src);
    MassMatrix->vmult(scratch2, scratch1);
    StiffnessMatrix->vmult(dst, scratch2);
    dst *= dt*gamma*eps;
    
    CHPhaseMatrix->vmult(scratch1, src);
    dst += scratch1;
}

