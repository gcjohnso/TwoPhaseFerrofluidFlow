# TwoPhaseFerrofluidFlow
A two phase ferrofluid flow solver. This code implements the numerical method described in the paper [1]. For a summarized description of the PDE model, numerical method, and implementation detaiils see Final.pdf in the root directory.

This code is built using the deal.II library [2]. Additionally libraries required are Trilinos and MPI. For instructions on how to install the library see https://www.dealii.org/9.0.0/readme.html and for instructions on install Trilinos refer to https://www.dealii.org/9.0.0/external-libs/trilinos.html. All code is compiled using CMake and the process for using it as such. First navigate to the root folder of the component you wish to compile, for example the Navier-Stokes solver in the MainSource folder. Then type the following commands:

cmake .

make

make run

The code currently has the following functionality:

1) A Cahn-Hilliard solver (located in MainSource)
2) A Navier-Stokes solver (located in MainSource)
3) Code which returns the value of the applied magnetic field given a set of dipoles at given locations in space (located in OldCurrentlyUnusedCode)

The above three components have been unit tested to ensure correctness of the numerical implementation. The Cahn-Hilliard and Navier-Stokes unit tests are located in the testing folder, while the applied magnetic field unit tests are currently implemented inside the source file. For additional information on the unit tests please refer to Section 4 of Final.pdf in the root directory.

In order to run the Cahn-Hilliard or Navier-Stokes solvers, you must make an instantiation of the respective class objects and then call the run method. For detailed information on how to modify the solvers to work with your specific initial conditions and forcing function, please refer to the respective wiki entries.

Additional examples from the deal.II tutorials can be found in the Examples folder along with a document describing each tutorial.

References:
[1] R.H. Nochetto, A.J. Salgado, and I. Tomas, A diffuse interface model for two-phase derrofluid flows, Computer Methods in Applied Mechanics and Engineering, 309 (2016), pp. 497-531.
[2] W. Bangerth, T. Heister, and G. Kanschat, Deai.ii differential equations analysis library, technical reference.
