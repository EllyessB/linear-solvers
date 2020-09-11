#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>
#include <vector>
#include <memory>
// #include <cblas.h>

#include "CSRMatrix.h"

using namespace std;

/*
Constructor for the CSRMatrix object with memory preallocation
*/
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) : Matrix<T>(rows, cols, false), nnzs(nnzs)
{
   // If we don't pass false in the initialisation list base constructor, it would allocate values to be of size
   // rows * cols in our base matrix class
   // So then we need to set it to the real value we had passed in
   this->preallocated = preallocate;

   // If we want to handle memory ourselves
   if (this->preallocated)
   {
      // Must remember to delete this in the destructor
      this->values = new T[this->nnzs];
      this->row_position = new int[this->rows + 1];
      this->col_index = new int[this->nnzs];
   }
}

/*
Constructor for the CSRMatrix object where memory has been allocated outside
*/
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_position), col_index(col_index)
{
}

/*
Destructor for the CSRMatrix object
*/
template <class T>
CSRMatrix<T>::~CSRMatrix()
{
   // Delete the values array
   if (this->preallocated)
   {
      delete[] this->row_position;
      delete[] this->col_index;
   }
   // The super destructor is called after we finish here
   // This will delete this->values if preallocated is true
}

/*
Generate pseudo-random data for a pre-allocated CSRMatrix object.
*/
template <class T>
void CSRMatrix<T>::genValues(int *irow, int *jcol, T *val)
{
   // seed for random number generation
   srand(time(nullptr));

   // initialise vector to store generated position indicies
   vector<int> pos_vec;
   pos_vec.reserve(nnzs);

   // initalise a boolean array to keep track of position indicies already used
   unique_ptr<bool[]> used(new bool[this->rows * this->cols]);

   // generate position indicies
   int n = 0, pos;
   while (n < this->nnzs)
   {
      pos = (rand() % (this->rows * this->cols));
      // check the positon index generated is unique, if not re-do
      if (used[pos] != 1)
      {
         pos_vec.push_back(pos);

         // flag as a used position and increment counter for while loop
         used[pos] = 1;
         n++;
      }
   }

   // sort the generated position indicies in row major order
   sort(pos_vec.begin(), pos_vec.end());

   // populate the output arrays
   for (int p = 0; p < nnzs; p++)
   {
      // back-calculate the (row, col) tuples
      jcol[p] = pos_vec[p] % this->cols;
      irow[p] = (pos_vec[p] - jcol[p]) / this->cols;
      // generate a corresponding value
      val[p] = (rand() % 10) + 1;
   }
}

/*
Generate pseudo-random data for a pre-allocated CSRMatrix object such
that it is symmetric positive definite (SPD).

The input matrix must be square, and (nnzs - rows) must be an even number.
*/
template <class T>
void CSRMatrix<T>::genValuesSPD(int *irow, int *jcol, T *val)
{
   if (irow == nullptr || jcol == nullptr || val == nullptr)
   {
      cerr << "Data has not been created properly" << endl;
      return;
   }

   if ((this->nnzs - this->rows) % 2 != 0 || this->rows != this->cols)
   {
      cerr << "symmetric matrix not possible for the matrix dimensions and/or nnzs" << endl;
      return;
   }

   // seed for random number generation
   srand(time(nullptr));
   random_device rd;
   default_random_engine re(rd());
   int lb = -10, ub = 10;
   uniform_int_distribution<int> val_dist(lb, ub);

   // initialise vector to store generated position-value pairs
   vector<pair<int, T>> pos_val_pairs;

   // initalise a boolean array to keep track of position indicies already used
   // WARNING: for very large sparse matrices, this becomes a very large array and may cause a crash...
   unique_ptr<bool[]> used(new bool[this->rows * this->cols]);

   T vlu;
   int n = 0, row, col;
   long long pos, pos_; // very large sparse matrix indices (up to rows * cols) will overflow int!
   double density = double(this->nnzs) / (this->rows * this->cols);

   // generate positon-value pairs for the off-diagonal terms
   while (n < this->nnzs / 2 - this->rows / 2)
   {
      row = (rand() % (this->rows - 1)) + 1; // generated value can be in any row
      col = (rand() % row);                  // force column to be in the lower triangle
      pos = row * this->cols + col;
      pos_ = col * this->cols + row;

      // check the positon index generated is unique, if not re-do
      if (used[pos] != 1)
      {
         // generate random value
         vlu = val_dist(re);
         // pair value with the position, and force a symmetry entry
         pos_val_pairs.push_back(make_pair(pos, vlu));
         pos_val_pairs.push_back(make_pair(pos_, vlu));

         // flag as a used position and increment counter for while loop
         used[pos] = 1;
         n++;
      }
   }

   // create the diagonal elements
   for (int k = 0; k < this->rows; k++)
   {
      vlu = (abs(ub) + abs(lb)) / 2 * density * this->rows + abs(val_dist(re)); // make diagonally dominant
      pos_val_pairs.push_back(make_pair(k * this->cols + k, vlu));
   }

   // sort the generated position-value pairs in row major order
   sort(pos_val_pairs.begin(), pos_val_pairs.end());

   // populate the output arrays
   for (int p = 0; p < this->nnzs; p++)
   {
      // back-calculate the (row, col) tuples
      jcol[p] = pos_val_pairs[p].first % this->cols;
      irow[p] = (pos_val_pairs[p].first - jcol[p]) / this->cols;
      // corresponding value
      val[p] = pos_val_pairs[p].second;
   }
}

/* 
Generate data to construct a banded square matrix where the diagonal values correspond
to the values in the input stencil
*/
template <class T>
void CSRMatrix<T>::genValuesBand(int *irow, int *jcol, T *val, int bandwidth, T *stencil)
{
   // counter for constructing row-major order data
   int count = 0;

   // generate data for the top B rows
   for (int j = 0; j < bandwidth; j++)
   {
      for (int i = 0; i < bandwidth + j + 1; i++)
      {
         val[count] = stencil[bandwidth - j + i];
         jcol[count] = i;
         irow[count] = j;
         count++;
      }
   }

   // generate data for the middle section (where the full stencil can be applied)
   for (int k = bandwidth; k < this->rows - bandwidth; k++)
   {
      for (int j = 0; j < 2 * bandwidth + 1; j++)
      {
         val[count] = stencil[j];
         jcol[count] = (k - bandwidth) + j;
         irow[count] = k;
         count++;
      }
   }

   // generate data for the bottom B rows
   int k = 0;
   for (int j = this->rows - bandwidth; j < this->rows; j++)
   {
      int m = 0;
      for (int i = this->cols - 2 * bandwidth + k; i < this->cols; i++)
      {
         val[count] = stencil[m];
         jcol[count] = i;
         irow[count] = j;
         count++;
         m++;
      }
      k++;
   }
}

/*
Set the values, row_position and col_index arrays of a pre-allocated
CSRMatrix object given row indices, column indicies and values.

Assumes the matrix has been initialised with the correct rows, cols
and nnzs for the input data and the inputs are in row-major order.
*/
template <class T>
void CSRMatrix<T>::setValues(int *irow, int *jcol, T *val)
{
   if (irow == nullptr || jcol == nullptr || val == nullptr)
   {
      cerr << "Data has not been created properly" << endl;
      return;
   }

   // set values
   for (int v = 0; v < this->nnzs; v++)
   {
      this->values[v] = val[v];
   }
   // set col_index
   for (int j = 0; j < this->nnzs; j++)
   {
      this->col_index[j] = jcol[j];
   }
   // set row_position
   int row_count = 0, k = 0;
   for (int i = 0; i < this->rows; i++)
   {
      while (irow[k] == i)
         k++;

      this->row_position[i + 1] = k;
   }
   this->row_position[0] = 0;
}

/*
Print out the matrix in CSR format
*/
template <class T>
void CSRMatrix<T>::printMatrix()
{
   cout << "Printing matrix..." << endl;
   cout << "Values: ";
   for (int j = 0; j < this->nnzs; j++)
   {
      cout << this->values[j] << " ";
   }
   cout << endl;
   cout << "row_position: ";
   for (int j = 0; j < this->rows + 1; j++)
   {
      cout << this->row_position[j] << " ";
   }
   cout << endl;
   cout << "col_index: ";
   for (int j = 0; j < this->nnzs; j++)
   {
      cout << this->col_index[j] << " ";
   }
   cout << endl
        << endl;
}

/*
Populate a pre-allocated Matrix object with the non-zero entries
of the CSRMatrix object.
*/
template <class T>
void CSRMatrix<T>::sparse2dense(Matrix<T> &dense_mat)
{
   if (dense_mat.rows != this->rows || dense_mat.cols != this->cols)
   {
      cerr << "Size of dense matrix does not match sparse matrix" << endl;
      return;
   }

   // initialise dense_mat to all zeros
   for (int i = 0; i < this->rows * this->cols; i++)
   {
      dense_mat.values[i] = 0;
   }

   // loop to populate non-zero entries of dense_mat
   int row_start = -1;
   int row_end = -1;
   for (int row = 0; row < this->rows; row++)
   {
      row_start = this->row_position[row];
      row_end = this->row_position[row + 1];

      for (int j = row_start; j < row_end; j++)
      {
         dense_mat.values[row * this->cols + col_index[j]] = this->values[j];
      }
   }
}

/*
Do a matrix-vector product | output = this * input
*/
template <class T>
void CSRMatrix<T>::matVecMult(double *input_vec, double *output_vec)
{
   if (input_vec == nullptr || output_vec == nullptr)
   {
      cerr << "Input or output haven't been created" << endl;
      return;
   }

   // Set the output to zero
   for (int i = 0; i < this->rows; i++)
   {
      output_vec[i] = 0.0;
   }

   int val_counter = 0;
   // Loop over each row
   for (int i = 0; i < this->rows; i++)
   {
      // Loop over all the entries in this col
      for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
      {
         // This is an example of indirect addressing
         // Can make it harder for the compiler to vectorise!
         output_vec[i] += this->values[val_index] * input_vec[this->col_index[val_index]];
      }
   }
}

/*
Symbolic matrix-matrix multiplication to return number of non-zeros
in the output matrix of output = this * mat_right
*/
template <class T>
int CSRMatrix<T>::matMatMult_SYM(CSRMatrix<T> &mat_right)
{
   // Check our dimensions match
   if (this->cols != mat_right.rows)
   {
      cerr << "Input dimensions for matrices don't match" << endl;
      return 0;
   }

   int j, k, ip = 0;

   // an array to keep track of which column indices have already been visited in the current row
   unique_ptr<int[]> xb(new int[mat_right.cols]);
   for (int n = 0; n < mat_right.cols; n++)
   {
      xb[n] = -1;
   }

   // looping over rows in A (which corresponds to the rows in the output)
   for (int i = 0; i < this->rows; i++)
   {
      // loop over the non-zero elements of A in the current row
      for (int jp = this->row_position[i]; jp < this->row_position[i + 1]; jp++)
      {
         j = this->col_index[jp]; // j is the row in B corresponding to the column of the current non-zero value in the current row of A

         // Loop over the non-zero elements of B in the row corresponding
         // to the column index of the current non-zero element in A
         for (int kp = mat_right.row_position[j]; kp < mat_right.row_position[j + 1]; kp++)
         {
            k = mat_right.col_index[kp]; // k is the column index of the current non-zero values in B
                                         // which corresponds to the column index in the current row of the output

            // check to see if the current (row, col) index in the output
            // has already been visited. If it has NOT, add a new entry and mark
            // this position as visited.
            if (xb[k] != i)
            {
               xb[k] = i; // mark this position as visited
               ip++;      // increment the number of non-zeros
            }
         }
      }
   }
   return ip;
}

/*
Do numeric matrix-matrix multiplication | output = this * mat_right
*/
template <class T>
void CSRMatrix<T>::matMatMult(CSRMatrix<T> &mat_right, CSRMatrix<T> &output)
{

   // Check our dimensions match
   if (this->cols != mat_right.rows)
   {
      cerr << "Input dimensions for matrices don't match" << endl;
      return;
   }

   // Check if our output matrix has had space allocated to it
   if (output.values != nullptr)
   {
      // Check our dimensions match
      if (this->rows != output.rows || mat_right.cols != output.cols)
      {
         cerr << "Input dimensions for matrices don't match" << endl;
         return;
      }
   }
   // The output hasn't been preallocated, so we are going to do that
   else
   {
      cerr << "OUTPUT HASN'T BEEN ALLOCATED" << endl;
   }

   int j, k, ip = 0;

   // an array to keep track of which column indices have already been visited in the current row
   unique_ptr<int[]> xb(new int[mat_right.cols]);
   // array to act as temporary storage for the current row of the output matrix
   unique_ptr<double[]> x(new double[mat_right.cols]);

   for (int n = 0; n < output.cols; n++)
   {
      xb[n] = -1;
      x[n] = 0;
   }

   // looping over rows in A (which corresponds to the rows in the output)
   for (int i = 0; i < this->rows; i++)
   {
      output.row_position[i] = ip; // ip is cumulative number of nonzeros in the output (updated at the end of each row iteration)

      // loop over the non-zero elements of A in the current row
      for (int jp = this->row_position[i]; jp < this->row_position[i + 1]; jp++)
      {
         j = this->col_index[jp]; // j is the row in B corresponding to the column of the current non-zero value in the current row of A

         // Loop over the non-zero elements of B in the row corresponding
         // to the column index of the current non-zero element in A
         for (int kp = mat_right.row_position[j]; kp < mat_right.row_position[j + 1]; kp++)
         {
            k = mat_right.col_index[kp]; // k is the column index of the current non-zero values in B
                                         // which corresponds to the column index in the current row of the output

            // check to see if the current (row, col) index in the output
            // has already been visited. If it has NOT, add a new entry and mark
            // this position as visited. If it has, only update the value in that position.
            if (xb[k] != i)
            {
               output.col_index[ip] = k;                       // store column index of the value
               xb[k] = i;                                      // mark this position as visited
               x[k] = this->values[jp] * mat_right.values[kp]; // add a new value
               ip++;                                           // increment the number of non-zeros
            }
            else
            {
               x[k] += this->values[jp] * mat_right.values[kp]; // update existing value
            }
         }
      }
      // put values of the current row in the output matrix into the correct
      // position in the CSR values array
      for (int vp = output.row_position[i]; vp < ip; vp++)
      {
         output.values[vp] = x[output.col_index[vp]];
      }
   }
   // last entry in row position = nnzs
   output.row_position[this->rows] = ip;
}

/*
Solve Ax=b using Simultaneous Over-Relaxation Method.
*/
template <class T>
int CSRMatrix<T>::SOR(T *RHS_vec, T *sol_vec, int it_max, double tol)
{
   double res = 0., res2 = 0., res_prv = 0., sum = 0., omega = 1.0;
   unique_ptr<double[]> Ax(new T[this->rows]);
   double idiag = 1;
   int k_relax = 10;

   // begin iterations
   for (int k = 0; k < it_max; k++)
   {
      // calculate (discontinuous) dot product of each row of A with x -- consider non-zero values in A only
      for (int i = 0; i < this->rows; i++)
      {
         sum = 0;
         for (int val_ind = this->row_position[i]; val_ind < this->row_position[i + 1]; val_ind++)
         {
            if (i == this->col_index[val_ind])
            {
               idiag = this->values[val_ind];
               continue;
            }
            sum += this->values[val_ind] * sol_vec[this->col_index[val_ind]];
         }
         // update solution vector
         sol_vec[i] = (omega / idiag) * (RHS_vec[i] - sum) + (1 - omega) * sol_vec[i];
      }

      // calculate residual
      res2 = 0;
      this->matVecMult(sol_vec, Ax.get());

      for (int i = 0; i < this->rows; i++)
      {
         res2 += pow(Ax[i] - RHS_vec[i], 2);
      }
      res = sqrt(res2);

      // Check for convergence
      if (res < tol)
      {
         return k;
      }

      // update omega to optimal estimate after k_relax iterations
      if (k == k_relax)
         res_prv = res;
      if (k == k_relax + 1)
      {
         omega = 2 / (1 + sqrt(1 - res / res_prv));
         cout << "omega_opt = " << omega << endl;
      }
   }
}

/*
Solve Ax = b using Multigrid V-Cycle with a Gauss-Seidel smoother
*/
template <class T>
void CSRMatrix<T>::MGV(T *RHS_vec, T *sol_vec)
{

   const double tol = 1e-6;
   const double it_smth = 5;

   // if at coarsest level, smooth out with GS
   if (this->cols <= 3)
   {
      this->SOR(RHS_vec, sol_vec, it_smth, tol);
   }
   else // smooth solution
   {
      // define all required variables and arrays
      int nnzr, nnz;
      int coarse_size = (this->cols - 1) / 2;
      unique_ptr<double[]> res(new double[this->cols]);
      unique_ptr<double[]> res_coarse(new double[coarse_size]);
      unique_ptr<T[]> Ax(new T[this->cols]);
      unique_ptr<double[]> error_coarse(new double[this->cols]);
      unique_ptr<double[]> error_fine(new double[this->cols]);

      // GS pre-smoothing
      this->SOR(RHS_vec, sol_vec, it_smth, tol);

      // calculate residual
      this->matVecMult(sol_vec, Ax.get());
      for (int i = 0; i < this->cols; i++)
      {
         res[i] = RHS_vec[i] - Ax[i];
      }

      // create restrict & interpolation operators
      nnzr = 3 * coarse_size;
      shared_ptr<CSRMatrix<double>> restrict_mat(new CSRMatrix<double>(coarse_size, this->cols, nnzr, true));
      shared_ptr<CSRMatrix<double>> interp_mat(new CSRMatrix<double>(this->cols, coarse_size, nnzr, true));

      for (int i = 0; i < nnzr; i += 3)
      {
         restrict_mat.get()->values[i] = 0.25;
         interp_mat.get()->values[i] = 0.5;
         restrict_mat.get()->values[i + 1] = 0.5;
         interp_mat.get()->values[i + 1] = 1;
         restrict_mat.get()->values[i + 2] = 0.25;
         interp_mat.get()->values[i + 2] = 0.5;
         restrict_mat.get()->col_index[i] = i / 3 * 2;
         restrict_mat.get()->col_index[i + 1] = i / 3 * 2 + 1;
         restrict_mat.get()->col_index[i + 2] = i / 3 * 2 + 2;
      }

      for (int i = 0; i < coarse_size; i++)
      {
         interp_mat.get()->col_index[i * 3] = i;
         interp_mat.get()->col_index[i * 3 + 1] = i;
         interp_mat.get()->col_index[i * 3 + 2] = i;
         restrict_mat.get()->row_position[i] = 3 * i;
      }
      restrict_mat.get()->row_position[coarse_size] = nnzr;
      interp_mat.get()->row_position[0] = 0;

      for (int i = 0; i < this->cols / 2; i++)
      {
         interp_mat.get()->row_position[2 * i + 1] = 3 * i + 1;
         interp_mat.get()->row_position[2 * i + 2] = 3 * i + 2;
      }

      interp_mat.get()->row_position[this->cols] = nnzr;

      // restrict residual to coarser grid
      restrict_mat.get()->matVecMult(res.get(), res_coarse.get());

      // restrict matrix A to next coarser grid using matmatmult of restriction and interpolation operator
      nnz = restrict_mat.get()->matMatMult_SYM(*this);
      shared_ptr<CSRMatrix<T>> middle_man(new CSRMatrix<T>(coarse_size, this->cols, nnz, true));

      restrict_mat.get()->matMatMult(*this, *middle_man);
      nnz = middle_man.get()->matMatMult_SYM(*interp_mat.get());
      shared_ptr<CSRMatrix<T>> coarse_A(new CSRMatrix<T>(coarse_size, coarse_size, nnz, true));

      middle_man.get()->matMatMult(*interp_mat.get(), *coarse_A.get());

      // solve recursively for error on the next coarsest grid
      for (int i = 0; i < coarse_size; i++)
      {
         error_coarse[i] = 0;
      }
      coarse_A.get()->MGV(error_coarse.get(), res_coarse.get());

      // interpolate error back to fine grid
      interp_mat.get()->matVecMult(error_coarse.get(), error_fine.get());

      // add interpolated error to the solution
      for (int i = 0; i < this->cols; i++)
      {
         sol_vec[i] += error_fine[i];
      }

      // GS post-smoothing
      this->SOR(RHS_vec, sol_vec, it_smth, tol);
   }
}
