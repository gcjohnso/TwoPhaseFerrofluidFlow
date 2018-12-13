# TwoPhaseFerrofluidFlow
A two phase ferrofluid flow solver. This code implements the numerical method described in the paper [1]. For a summarized description of the PDE model and the numerical method see either Proposal.pdf or Final.pdf in the LatexDocumentation folder.

This code is built using the deal.II library [2]. For instructions on how to install the library see https://www.dealii.org/9.0.0/readme.html. The code is compiled using CMake and the process for using it as such. First navigate to the root folder of the component you wish to compile, for example the Mesh folder. Then type the following commands:

cmake .

make

make run

The code currently has functionality to generate the mesh and to handle the generation of the applied harmonizing field. For a description of these methods refer to the wiki.

Additional examples from the deal.II tutorials can be found in the Examples folder along with a document describing each tutorial.

References:
[1] R.H. Nochetto, A.J. Salgado, and I. Tomas, A diffuse interface model for two-phase derrofluid flows, Computer Methods in Applied Mechanics and Engineering, 309 (2016), pp. 497-531.
[2] W. Bangerth, T. Heister, and G. Kanschat, Deai.ii differential equations analysis library, technical reference.
