#include "../inc/CahnHilliardEquation.h"
#include "../inc/NavierStokesEquation.h"

int main(int argc, char *argv[]){
    using namespace dealii;
    
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    
    //CahnHilliardEquation<2> CHEq;
    //CHEq.run();
    
    NavierStokesEquation<2> NSEq;
    NSEq.run();
    
}
