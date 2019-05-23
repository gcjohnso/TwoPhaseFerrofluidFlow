#include "../inc/NavierStokesEquation.h"
using namespace dealii;

/**
 * Constructor which initializes the model parameters, parameters related to mesh refinement, and the finite elements used.
 */
template<int dim>
NavierStokesEquation<dim>::NavierStokesEquation()
:
mu(1),
lambda(0.05),
eps(0.2),
nu_w(1.0),
nu_f(2.0),
numTimeSteps(1000),
finalTime(1.0),
globalRefinementLevel(3),
degree(2),
CHFE(FE_Q<dim>(degree), 1),
NSFE(FE_Q<dim>(degree), dim, FE_Q<dim>(degree-1), 1),
MagFE(FE_DGQ<dim>(degree), dim),
MagPotFE(FE_Q<dim>(degree), 1),
AppliedMagFE(FE_Q<dim>(1), dim)
{
}

/**
 * Method which initializes the DoFs, applies the dirichlet BC, and initializes the matrices and vectors used.
 */
template<int dim>
void NavierStokesEquation<dim>::initializeNSDoF(){
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
    prevNSSolution.reinit(2);
    prevNSSolution.block(0).reinit(VelocityAllIndex);
    prevNSSolution.block(1).reinit(PressureAllIndex);
    prevNSSolution.collect_sizes();
    NSRHS.reinit(2);
    NSRHS.block(0).reinit(VelocityAllIndex);
    NSRHS.block(1).reinit(PressureAllIndex);
    NSRHS.collect_sizes();
    
     //Interpolate the IC
    VectorTools::interpolate(CHDoFHandler, VelocityInitialCondition<dim>(), prevNSSolution);
    NSConstraints.distribute(prevNSSolution);
}

/**
 * Method which initializes the DoFs for the CH solution variables. As this is only the solver for NS, this method is essentially just a placeholder
 * and will be eventually replaced by the initializeCHDoF from the CH solver when everything gets combined.
 */
template<int dim>
void NavierStokesEquation<dim>::initializeCHDoF(){
    CHDoFHandler.initialize(triangulation, CHFE);
    DoFRenumbering::Cuthill_McKee(CHDoFHandler);
    
    unsigned int CHNumOfDoFs = CHDoFHandler.n_dofs();
    
    //Initialize all vectors
    IndexSet CHAllIndex(CHNumOfDoFs);
    CHAllIndex.add_range(0, CHNumOfDoFs);
    prevPhaseSolution.reinit(CHAllIndex);
    currChemicalPotentialSolution.reinit(CHAllIndex);
}

/**
 * Method which initializes the DoFs for the Magnetization solution variable. As this is only the solver for NS, this method is essentially just a placeholder
 * and will be eventually replaced by the initializeMagDoF from the magnetization solver when everything gets combined.
 */
template<int dim>
void NavierStokesEquation<dim>::initializeMagDoF(){
    MagDoFHandler.initialize(triangulation, MagFE);
    DoFRenumbering::Cuthill_McKee(MagDoFHandler);
    
    unsigned int MagNumOfDoFs = MagDoFHandler.n_dofs();
    
    //Initialize all vectorsCHFEValues
    IndexSet MagAllIndex(MagNumOfDoFs);
    MagAllIndex.add_range(0, MagNumOfDoFs);
    currMagSolution.reinit(MagAllIndex);
}

/**
 * Method which initializes the DoFs for the Magnetic Potential solution variable. As this is only the solver for NS, this method is essentially just a placeholder
 * and will be eventually replaced by the initializeMagPotDoF from the Magnetic Potential solver when everything gets combined.
 */
template<int dim>
void NavierStokesEquation<dim>::initializeMagPotDoF(){
    MagPotDoFHandler.initialize(triangulation, MagPotFE);
    DoFRenumbering::Cuthill_McKee(MagPotDoFHandler);
    
    unsigned int MagPotNumOfDoFs = MagPotDoFHandler.n_dofs();
    
    //Initialize all vectors
    IndexSet MagPotAllIndex(MagPotNumOfDoFs);
    MagPotAllIndex.add_range(0, MagPotNumOfDoFs);
    currMagPotSolution.reinit(MagPotAllIndex);   
}

/*
 * Method which creates the square mesh and globally refines it a set number of time.
 */
template<int dim>
void NavierStokesEquation<dim>::initializeMesh(){
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(globalRefinementLevel);
}

/**
 * Method which computes the Mass matrix for the velocity, and the matrices B and B^T for the terms (q, divU).
 * The matrix will then be used in created the LSC preconditioner.
 */
template<int dim>
void NavierStokesEquation<dim>::assembleSchurComplementPreconditioner(){
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

/**
 * Method which computes the LHS and the RHS of the NS system.
 */
template<int dim>
void NavierStokesEquation<dim>::assembleNSMatrix(){
    NSMatrix = 0;
    NSRHS = 0;
    
    QGauss<dim> quadratureFormula(degree + 2);
    
    FEValues<dim> NSFEValues(NSFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> MagFEValues(MagFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> MagPotFEValues(MagPotFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> CHFEValues(CHFE, quadratureFormula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    const unsigned int NSDoFPerCell = NSFE.dofs_per_cell;
    const unsigned int numQuadraturePoints = quadratureFormula.size();
    
    FullMatrix<double> NSLocalMatrix(NSDoFPerCell, NSDoFPerCell);
    Vector<double> NSLocalRHS(NSDoFPerCell);
    
    std::vector<types::global_dof_index> NSLocalDoFIndices(NSDoFPerCell);
    
    FEValuesExtractors::Vector velocityExtractor(0);
    FEValuesExtractors::Scalar pressureExtractor(dim);
    FEValuesExtractors::Vector magExtractor(0);
    FEValuesExtractors::Scalar magPotExtractor(0);
    FEValuesExtractors::Scalar phaseExtractor(0); //Works for both phase and chem pot (As they use the same fe basis)
    
    std::vector<Tensor<1, dim>> velocityShapeValue(NSDoFPerCell);
    std::vector<Tensor<2, dim>> velocityShapeGradientValue(NSDoFPerCell);
    std::vector<SymmetricTensor<2, dim>> velocityShapeSymmetricGradientValue(NSDoFPerCell);
    std::vector<double> velocityShapeDivergence(NSDoFPerCell);
    std::vector<double> pressureShapeValue(NSDoFPerCell);
    
    std::vector<Tensor<1, dim>> localPrevVelocityValue(numQuadraturePoints);
    std::vector<double>         localPrevVelocityDivergence(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localCurrMagValue(numQuadraturePoints);
    std::vector<Tensor<2, dim>> localCurrMagGradient(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localCurrMagPotGradient(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localTotalMagFieldValue(numQuadraturePoints);
    std::vector<double> localPrevPhaseValue(numQuadraturePoints);
    std::vector<Tensor<1, dim>> localCurrChemPotGradient(numQuadraturePoints);  
    
    const NSVelcoityForcingFunction<dim> NSVelocityForcing;
    std::vector<Tensor<1, dim>> velocityForcingValue(numQuadraturePoints);
    
    typename DoFHandler<dim>::active_cell_iterator NSCell = NSDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator NSEndCell = NSDoFHandler.end();
    typename DoFHandler<dim>::active_cell_iterator MagCell = MagDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator MagPotCell = MagPotDoFHandler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator CHCell = CHDoFHandler.begin_active();
    
    for(; NSCell != NSEndCell; ++MagCell, ++MagPotCell, ++CHCell, ++NSCell){
        NSFEValues.reinit(NSCell);
        MagFEValues.reinit(MagCell);
        MagPotFEValues.reinit(MagPotCell);
        CHFEValues.reinit(CHCell);
        
        NSLocalMatrix = 0;
        NSLocalRHS = 0;
        
        NSFEValues[velocityExtractor].get_function_values(prevNSSolution, localPrevVelocityValue);
        NSFEValues[velocityExtractor].get_function_divergences(prevNSSolution, localPrevVelocityDivergence);
        MagFEValues[magExtractor].get_function_values(currMagSolution, localCurrMagValue);
        MagFEValues[magExtractor].get_function_gradients(currMagSolution, localCurrMagGradient);
        MagPotFEValues[magPotExtractor].get_function_gradients(currMagPotSolution, localCurrMagPotGradient);
        //Here we compute the total mag field
        for(unsigned int q = 0; q < numQuadraturePoints; ++q){
            //Get the value of the applied field
            
            //Then add it to the value of the MagPot gradient
            localTotalMagFieldValue[q] = localCurrMagPotGradient[q];
        }
        CHFEValues[phaseExtractor].get_function_values(prevPhaseSolution, localPrevPhaseValue);
        CHFEValues[phaseExtractor].get_function_gradients(currChemicalPotentialSolution, localCurrChemPotGradient);
        
        NSVelocityForcing.vector_value_list(NSFEValues.get_quadrature_points(), velocityForcingValue, currTime);
        
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
                                    + dt * .5 * (localPrevVelocityValue[q] * velocityShapeGradientValue[j]) * velocityShapeValue[i] //Swapped order of shape grad and prev vel
                                    + dt * .5 * localPrevVelocityDivergence[q] * velocityShapeValue[i] * velocityShapeValue[j] //Swapped order of shape grad and prev vel
                                    + dt * nu_w *(1.0 + (nu_f - nu_w)*Heaviside(localPrevPhaseValue[q]/eps)) *  velocityShapeSymmetricGradientValue[i] * velocityShapeSymmetricGradientValue[j] //swapped i & j index
                                    - dt * pressureShapeValue[j] * velocityShapeDivergence[i]
                                    - dt * velocityShapeDivergence[j] * pressureShapeValue[i]) * NSFEValues.JxW(q);
                }
                
                NSLocalRHS(i) += (localPrevVelocityValue[q] * velocityShapeValue[i]
                             /*- mu * dt * (velocityShapeValue[i] * localCurrMagGradient[q]) * localTotalMagFieldValue[q]
                             - .5 * mu * dt * velocityShapeDivergence[i] * (localCurrMagValue[q] * localTotalMagFieldValue[q])
                             + (lambda / eps) * dt * (localPrevPhaseValue[q] * localCurrChemPotGradient[q]) * velocityShapeValue[i]*/
                             + dt * velocityShapeValue[i] * velocityForcingValue[q]) * NSFEValues.JxW(q);
            }
        }
        NSCell->get_dof_indices(NSLocalDoFIndices);
        NSConstraints.distribute_local_to_global(NSLocalMatrix, NSLocalRHS, NSLocalDoFIndices, NSMatrix, NSRHS);
    }
}

/**
 * Method which initializes the AMG preconditioner used to compute the inverse action of the velocity matrix. Again, the configuration
 * of the preconditioner is taken from the Step-31 deal.II tutorial.
 */
template<int dim>
void NavierStokesEquation<dim>::initializeAMGPreconditioner(){
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

/**
 * Method which solves the NS system using GMRES with the Block Preconditioner detailed in Section 3.3 of the 
 * FinalPaper.pdf
 */
template<int dim>
void NavierStokesEquation<dim>::solveNS(){
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

/**
 * Method which computes the Heaviside function as
 * H(x) = 1/(1 + e^(-x)), the definition of this function
 * is given in the paper by Ignacio and Ricardo.
 */
double Heaviside(double x){
    return 0.5;//1.0/(1.0 + exp(-x));
}

template<int dim>
void NavierStokesEquation<dim>::run(){
    //Initialize time variables
    dt = finalTime/numTimeSteps;
    currTimeStep = 1;
    currTime = dt;
    
    initializeMesh();
    initializeNSDoF();
    initializeCHDoF();
    initializeMagDoF();
    initializeMagPotDoF();
    
    //Initialize other solu variables to 0
    prevPhaseSolution = 0;
    currChemicalPotentialSolution = 0;
    currMagSolution = 0;
    currMagPotSolution = 0;
    
    bool recomputeNSShcurComplement = true;
    
    while(currTimeStep <= numTimeSteps){
        std::cout << "Curr iter" << currTimeStep << " at time " << currTime << std::endl;
        if(recomputeNSShcurComplement){
            assembleSchurComplementPreconditioner();
            schurComplementPreconditioner.initialize(NSSchurComplementPreconditioner);
            recomputeNSShcurComplement = false;
        }
        assembleNSMatrix();
        
        schurComplementPreconditioner.updateVelocityBlock(NSMatrix.block(0,0));
        initializeAMGPreconditioner();
        //std::cout << "Begin solveNS()" << std::endl;
        solveNS();
        
        outputSolution();
        
        prevNSSolution = currNSSolution;
        
        currTimeStep++;
        currTime += dt;
        
    }
}            

/*
 * Outputs the solution at the current iteration to a file.
 */
template<int dim>
void NavierStokesEquation<dim>::outputSolution(){
    if (currTimeStep % 10 != 0)
        return;
    
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
    std::ofstream output("/home/werdho/src/UMD/AMSC664/TwoPhaseFerrofluidFlow/output/NS-" +
                        Utilities::int_to_string(currTimeStep, 4) + ".vtk");
    data_out.write_vtk(output);
}



//Things not in NavierStokesClass
template class NavierStokesEquation<2>;

/*
 * Default constructor
 */
template<int dim>
VelocityInitialCondition<dim>::VelocityInitialCondition() : Function<dim>()
{
}

/*
 * Function which returns the value of the initial condition for the velocity at the point p. This is currently implemeneted as a zero function,
 * as the forced solution is 0 at t=0.
 */
template<int dim>
double VelocityInitialCondition<dim>::value(const Point<dim> &p, const unsigned int component) const{
    if(component == 0){//Velocity x component
        return 0;
    }else if(component == 1){//Velocity y component
        return 0;
    }else{//Pressure value
        return 0;
    }
}

/**
 * Method which return the value of the dirichlet BC at a given point p. The component value refers to 
 * 0 - velocity x value
 * 1 - velocity y value
 * 2 - pressure value
 * Currently the method return 0 as we are using no-slip conditions.
 */
template<int dim>
double NSBoundaryValues<dim>::value(const Point<dim> &p, const unsigned int component) const{
    Assert (component < this->n_components, ExcIndexRange (component, 0, this->n_components));
    return 0;
}

/**
 * Method which returns the vector of values of the BC at a given point p.
 */
template<int dim>
void NSBoundaryValues<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const{
  for(unsigned int c=0; c<this->n_components; ++c)
    values(c) = value(p, c);
}

/*
 * Default constructor
 */
template<int dim>
NSVelcoityForcingFunction<dim>::NSVelcoityForcingFunction()
{
}

/**
 * Method which computes the value of the forcing function at a set of points. Currently the forcing function is implemeted to be the force
 * given the exact solution in NSExactSolution.
 */
template<int dim>
void NSVelcoityForcingFunction<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values, double t) const{
    NSExactSolution<dim> exactSolu(t);
    double nu = 1 *(1.0 + Heaviside(0));
    for(unsigned int i = 0; i < points.size(); ++i){
        Point p = points[i];

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

/**
 * Method which returns the value of the forced solution at a given point p. Again the component value refers to 
 * 0 - velocity x value
 * 1 - velocity y value
 * 2 - pressure value
 */
template<int dim>
double NSExactSolution<dim>::value(const Point<dim> &p, const unsigned int component) const{
    const double x = p[0];
    const double y = p[1];
    if(component == 0){
        //return -t * 4 * PI * sin(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * cos(2*PI*y);
        return -sin(t) * 2 * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y);
    }else if(component == 1){
        //return t * 4 * PI * cos(2*PI*x) * sin(2*PI*x) * sin(2*PI*y) * sin(2*PI*y);
        return sin(t) * 2 * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y);
    }else{
        return sin(2.0*PI*(x - y) + t);
    }
}

/**
 * Method which returns the value as a vector of the forced solution at a given point p.
 */
template<int dim>
void NSExactSolution<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const{
    Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));
    const double x = p[0];
    const double y = p[1];
    values(0) = -sin(t) * 2 * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y);
    values(1) = sin(t) * 2 * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y);
    values(2) = sin(2.0*PI*(x - y) + t);
}

/**
 * Method which returns the gradient of the forced solution at a given point p.
 */
template<int dim>
Tensor<1,dim> NSExactSolution<dim>::gradient(const Point<dim> &p, const unsigned int component) const{
    Tensor<1,dim> returnValue;
    const double x = p[0];
    const double y = p[1];
    if(component == 0){
        returnValue[0] = sin(t) * - PI * PI * sin(2*PI*x) * sin(2*PI*y);
        returnValue[1] = -sin(t) * 2 * PI * PI * sin(PI*x) * sin(PI*x) * cos(2*PI*y);
    }else if(component == 1){
        returnValue[0] = sin(t) * 2 * PI * PI * cos(2*PI*x) * sin(PI*y) * sin(PI*y);
        returnValue[1] = sin(t) * PI * PI * sin(2*PI*x) * sin(2*PI*y);
    }else{
        returnValue[0] = 2 * PI * cos(2.0*PI*(p[0] - p[1]) + t);
        returnValue[1] = -2 * PI * cos(2.0*PI*(p[0] - p[1]) + t);
    }
    return returnValue;
}

/**
 * Method which return the vector laplacian of the forced solution at a given point p.
 */
template<int dim>
Tensor<1,dim> NSExactSolution<dim>::vectorLaplacian(const Point<dim> &p) const{
    Tensor<1,dim> returnValue;
    const double x = p[0];
    const double y = p[1];
    returnValue[0] = sin(t) * (4 * PI * PI * PI * sin(PI*y) * cos(PI*y) * (sin(PI*x) * sin(PI*x) - cos(PI*x) * cos(PI*x)) + 8 * PI * PI * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y));
    returnValue[1] = sin(t) * (-8 * PI * PI * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y) + 4 * PI * PI * PI * sin(PI*x) * cos(PI*x) * (cos(PI*y) * cos(PI*y) - sin(PI*y) * sin(PI*y)));
    
    return returnValue;
}

/**
 * Method which return the time derivative of the forced solution at a given point p.
 */
template<int dim>
Tensor<1,dim> NSExactSolution<dim>::timeDeriv(const Point<dim> &p) const{
    Tensor<1,dim> returnValue;
    const double x = p[0];
    const double y = p[1];
    returnValue[0] = -cos(t) * 2 * PI * sin(PI*x) * sin(PI*x) * sin(PI*y) * cos(PI*y);
    returnValue[1] = cos(t) * 2 * PI * sin(PI*x) * cos(PI*x) * sin(PI*y) * sin(PI*y);
    return returnValue;
}

/*
 * Default constructor
 */
SchurComplementPreconditioner::SchurComplementPreconditioner()
{
}

/**
 * Method which initializes the SchurComplementPreconditioner object. It creates all objects needed to compute the action
 * of the LSC preconditioner on a vector.
 */
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

/**
 * Method for the preconditioner to take in a reference to the current velocity block
 */
void SchurComplementPreconditioner::updateVelocityBlock(TrilinosWrappers::SparseMatrix &velocityBlock){
    NSVelocityBlock = &velocityBlock;
}

/**
 * Method which computes the action of the LSC preconditioner on a vector
 */
void SchurComplementPreconditioner::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    inverseDiscreteLaplacian.vmult(tempP, src);
    
    schurComplementPreconditioner->block(0, 1).vmult(tempV1, tempP);
    massDiagonalInverse.vmult(tempV2, tempV1);
    NSVelocityBlock->vmult(tempV1, tempV2);
    massDiagonalInverse.vmult(tempV2, tempV1);
    schurComplementPreconditioner->block(1, 0).vmult(tempP, tempV2);
    
    inverseDiscreteLaplacian.vmult(dst, tempP);
}

/*
 * Default constructor
 */
NSBlockPreconditioner::NSBlockPreconditioner()
{
}

/**
 * Method which initializes the Block Preconditioner.
 */
void NSBlockPreconditioner::initialize(TrilinosWrappers::BlockSparseMatrix &matrix,  SchurComplementPreconditioner &schurPre, std::shared_ptr<TrilinosWrappers::PreconditionAMG> velocityBlockPreconditioner){
    NSMatrix = &matrix;
    schurPreconditioner = &schurPre;
    AMGPreconditioner = velocityBlockPreconditioner;
                                    
    IndexSet allIndex(matrix.block(1,1).m());
    allIndex.add_range(0, matrix.block(1,1).m());
    tempP.reinit(allIndex);
    tempP = 0;
}

/**
 * Applies the action of P^{-1} to a vector. The algorithm used is described in Section 3.3 in FinalPaper.pdf
 */
void NSBlockPreconditioner::vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const{
    AMGPreconditioner->vmult(dst.block(0), src.block(0));
    
    NSMatrix->block(1,0).vmult(tempP, dst.block(0));
    
    tempP -= src.block(1);
    
    schurPreconditioner->vmult(dst.block(1), tempP);
}

/*
 * Default constructor
 */
AMGMatrixInverseAction::AMGMatrixInverseAction()
{
}

/**
 * Method which initializes the invese action object. Uses the configuration for the AMG preconditioner
 * given in Step-31 of the deal.II tutorial.
 */
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

/**
 * Computes the inverse action of the given matrix by solving Ay=x using CG with the AMG preconditioner.
 */
void AMGMatrixInverseAction::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const{
    solverControl.set_tolerance(1e-6*src.l2_norm());
    
    SolverCG<TrilinosWrappers::MPI::Vector> CGSolver(solverControl);
    
    CGSolver.solve(*matrix, dst, src, *preconditioner);
}

