/*
 * Mesh.h
 *
 * This file contains all functions related to operations on the mesh.
 *
 *  Created on: Dec 1, 2018
 *      Author: gcjohnso
 */

#ifndef MESH_INCLUDE_MESH_H_
#define MESH_INCLUDE_MESH_H_

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <vector>
#include <fstream>

using namespace dealii;

/**
 * Creates a rectangle (in d=2 or 3) defined by two diagonally opposite corners with a set amount of elements in each direction. Additionally,
 * there is only one labeled boundary (which is enough as our PDE only has one boundary).
 * @param tria Triangulation that will be generated.
 * @param repetitions Vector of size = dim. In direction i, repetitions[i] elements are created.
 * @param lowerInner Lower inner point of the rectangle.
 * @param upperOuter Upper outer point of the rectangle (i.e. diagonally opposite of lowerInner).
 */
template <int dim>
void make_subdivided_rectangle(Triangulation<dim> &tria, std::vector<unsigned int> &repetitions, Point<dim> lowerInner, Point<dim> upperOuter);


/**
 * Outputs the given triangulation to a .eps file
 * @param tria The triangulation to be output.
 * @param filename The filename.
 */
template <int dim>
void output_mesh(Triangulation<dim> &tria, const char filename[]);

#endif /* MESH_INCLUDE_MESH_H_ */
