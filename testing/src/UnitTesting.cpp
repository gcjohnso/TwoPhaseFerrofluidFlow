#include "../inc/CahnHilliardUnitTest1.h"
#include "../inc/CahnHilliardUnitTest2.h"
#include "../inc/CahnHilliardUnitTest3.h"
#include "../inc/NavierStokesEquation.h"


int main(int argc, char *argv[]){
    using namespace dealii;
    
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    {
        //Flat profile, no forcing, adaptive refinement
        CahnHilliardUnitTest1<2> CHUnitTest1;
        CHUnitTest1.run();
    }
    
    {
    //Circular profile, no forcing, adaptive refinement
        CahnHilliardUnitTest2<2> CHUnitTest2;
        CHUnitTest2.run();
    }
    
    {
    //Forced solution, global refinement
        CahnHilliardUnitTest3<2> CHUnitTest3;
        CHUnitTest3.run();
    }
    
    {
        //NavierStokesEquation<2> NSEq;
        //NSEq.run();
    }
}
