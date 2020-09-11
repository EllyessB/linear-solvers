// #pragma once
#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H
#include "Matrix.h"

template <class T>
class CSRMatrix : public Matrix<T>
{
public:
   // constructor where we want to preallocate ourselves
   CSRMatrix(int rows, int cols, int nnzs, bool preallocate);
   // constructor where we already have allocated memory outside
   CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index);
   // destructor
   ~CSRMatrix();

   // generate pseudo-random values, col_index and row_positon data
   void genValues(int *irow, int *jcol, T *val);
   // generate pseudo-random values, col_index and row_positon data -- Symmetric Positive Definite
   void genValuesSPD(int *irow, int *jcol, T *val);
   // generate values, row_position and col_index data for a banded matrix
   void genValuesBand(int *irow, int *jcol, T *val, int bandwidth, T *stencil);
   // set values, col_index and row_position data
   void setValues(int *irow, int *jcol, T *val);

   // Print out the matrix in CSR format
   virtual void printMatrix();
   // Sparse to dense converter (for testing purposes)
   void sparse2dense(Matrix<T> &dense_mat);

   // Perform some operations with our matrix
   void matVecMult(double *input_vec, double *output_vec);
   void matMatMult(CSRMatrix<T> &mat_right, CSRMatrix<T> &output);
   int matMatMult_SYM(CSRMatrix<T> &mat_right);

   // Linear solver routines
   int SOR(T *RHS_vec, T *sol_vec, int it_max, double tol);
   void MGV(T *RHS_vec, T *sol_vec);

   // CSR data structures
   int *row_position = nullptr;
   int *col_index = nullptr;
   // number of non-zero entries the matrix
   int nnzs = -1;

private:
};

#endif