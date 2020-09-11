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

#include "Matrix.h"

using namespace std;

/*
Constructor for the Matrix object with memory preallocation
*/
template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate) : rows(rows), cols(cols), size_of_values(rows * cols), preallocated(preallocate)
{
    // If we want to handle memory ourselves
    if (this->preallocated)
    {
        // Must remember to delete this in the destructor
        this->values = new T[size_of_values];
    }
}

/*
Constructor for the CSRMatrix object where memory has been allocated outside
*/
template <class T>
Matrix<T>::Matrix(int rows, int cols, T *values_ptr) : rows(rows), cols(cols), size_of_values(rows * cols), values(values_ptr)
{
}

/*
Copy constructor to create a deep copy of a Matrix object
*/
template <class T>
Matrix<T>::Matrix(const Matrix<T> &Mat_to_copy)
{
    this->preallocated = Mat_to_copy.preallocated;
    this->rows = Mat_to_copy.rows;
    this->cols = Mat_to_copy.cols;
    this->size_of_values = Mat_to_copy.size_of_values;

    this->values = new T[this->size_of_values];
    for (int i = 0; i < this->size_of_values; i++)
    {
        this->values[i] = Mat_to_copy.values[i];
    }
}

/*
Destructor for the Matrix object
*/
template <class T>
Matrix<T>::~Matrix()
{
    // Delete the values array
    if (this->preallocated)
    {
        delete[] this->values;
    }
}

/*
Operator Overloading -- multiplication
*/
template <class T>
Matrix<T> *Matrix<T>::operator*(Matrix &output)
{
    auto output_prealloc = new Matrix(this->rows, output.cols, true);
    this->matMatMult(output, *output_prealloc);
    return output_prealloc;
}

/*
Operator Overloading -- addition
*/
template <class T>
Matrix<T> *Matrix<T>::operator+(Matrix &output)
{
    auto output_prealloc = new Matrix(this->rows, output.cols, true);
    this->matMatAdd(output, *output_prealloc);
    return output_prealloc;
}

/*
Generate pseudo-random values to populate the matrix
*/
template <class T>
void Matrix<T>::FillValues()
{
    if (this->preallocated)
    {
        srand(time(nullptr));
        for (int i = 0; i < this->rows * this->cols; i++)
        {
            this->values[i] = rand() % 10; // integers between 0-9
        }
    }
}

/*
Generate pseudo-random values to populate the matrix and turn into SPD
*/
template <class T>
void Matrix<T>::FillValuesSPD()
{
    // seed for random number generation
    srand(time(nullptr));
    random_device rd;
    default_random_engine re(rd());
    int lb = -10, ub = 10;
    uniform_int_distribution<int> val_dist(lb, ub);

    T vlu;

    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            vlu = val_dist(re);
            this->values[i * cols + j] = vlu;
            this->values[j * cols + i] = vlu;
        }
    }

    for (int k = 0; k < this->rows; k++)
    {
        vlu = abs(val_dist(re)) + (abs(ub) + abs(lb)) / 2 * this->cols; // make diagonally dominant
        this->values[k * cols + k] = vlu;
    }
}

/*
Print out the matrix values as a row-major order list
*/
template <class T>
void Matrix<T>::printValues()
{
    cout << "Printing values..." << endl;
    for (int i = 0; i < this->size_of_values; i++)
    {
        cout << this->values[i] << " ";
    }
    cout << endl
         << endl;
}

/*
Print out the matrix values as a 2D array
*/
template <class T>
void Matrix<T>::printMatrix()
{
    cout << "Printing matrix..." << endl;
    for (int j = 0; j < this->rows; j++)
    {
        for (int i = 0; i < this->cols; i++)
        {
            // We have explicitly used a row-major ordering here
            cout << this->values[i + j * this->cols] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

/*
<T>AXPY Constant times a vector plus a vector
*/
template <class T>
void Matrix<T>::daxpy(int n, double alpha, T *dx, int incx, T *dy, int incy)
{
    for (int i = 0; i < n; i++)
    {
        dy[i * incy] += alpha * dx[i * incx];
    }
}

/*
Perform a row swap, used in partial pivoting in Gaussian elimination
*/
template <class T>
void Matrix<T>::swapRows(int row1, int row2)
{
    double temp_val;
    for (int c = 0; c < this->cols; c++)
    {
        temp_val = this->values[row1 * this->cols + c];
        this->values[row1 * this->cols + c] = this->values[row2 * this->cols + c];
        this->values[row2 * this->cols + c] = temp_val;
    }
}

template <class T>
T Matrix<T>::ddot(int n, T *dx, int incx, T *dy, int incy)
{
    T output = 0;

    for (int i = 0; i < n; i++)
    {
        output += dy[i * incy] * dx[i * incx];
    }

    return output;
}

/*
Do a matrix-vector product | output_vec = this * input_vec
*/
template <class T>
void Matrix<T>::matVecMult(T *input_vec, T *output_vec)
{
    if (input_vec == nullptr || output_vec == nullptr)
    {
        cerr << "Input or output haven't been created" << endl;
        return;
    }

    // Set the output to zero
    for (int i = 0; i < this->rows; i++)
    {
        output_vec[i] = 0;
    }

    // loop order such that cache utilisation is optimised for
    // row major ordered input matrix
    for (int i = 0; i < this->rows; i++) // Loop over each row
    {
        for (int j = 0; j < this->cols; j++) // Loop over all each column
        {
            output_vec[i] += this->values[i * this->cols + j] * input_vec[j];
        }
    }
}

/*
Do matrix matrix multiplication | output = this * mat_right
(optional boolean flag for whether or not to do multiplication with transpose of mat_right)
*/
template <class T>
void Matrix<T>::matMatMult(Matrix &mat_right, Matrix &output, bool transpose_mat_right)
{
    // assign variables to rows and columns to avoid long if statements when
    // checking matrix dimensions are all conformable for either case (A*B and A*B.T)
    int m = this->rows;
    int n = this->cols;
    int p = -1, q = -1;
    if (!transpose_mat_right)
    {
        p = mat_right.cols;
        q = mat_right.rows;
    }
    else
    {
        p = mat_right.rows;
        q = mat_right.cols;
    }
    // Check dimensions match (multiplying matrices)
    if (n != q)
    {
        cerr << "Matrix dimensions are not conformable" << endl;
        return;
    }
    // Check if output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check dimensions match (output matrix)
        if (m != output.rows || p != output.cols)
        {
            cerr << "Matrix dimensions are not conformable" << endl;
            return;
        }
    }
    // Pre-allocate space to output matrix if it hasn't already been
    else
    {
        output.values = new T[m * p];
        output.preallocated = true;
    }

    // Set output values to zero before hand
    for (int i = 0; i < output.size_of_values; i++)
    {
        output.values[i] = 0;
    }

    // if statement is at the highest level for the sake of performance
    if (!transpose_mat_right)
    {
        // Multiplication algorithm for  A * B
        for (int i = 0; i < this->rows; i++)
        {
            for (int k = 0; k < this->cols; k++)
            {
                for (int j = 0; j < mat_right.cols; j++)
                {
                    output.values[i * output.cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
                }
            }
        }
    }
    else
    {
        // Multiplication algorithm for  A * B.T
        for (int i = 0; i < this->rows; i++)
        {
            for (int k = 0; k < this->cols; k++)
            {
                for (int j = 0; j < mat_right.cols; j++)
                {
                    output.values[i * output.cols + k] += this->values[i * this->cols + j] * mat_right.values[k * mat_right.cols + j];
                }
            }
        }
    }
}

/*
Do matrix-matrix addition | output = this + mat_right
*/
template <class T>
void Matrix<T>::matMatAdd(Matrix &mat_right, Matrix &output)
{

    // Check dimensions match
    if (this->rows != mat_right.rows || this->cols != mat_right.cols)
    {
        cerr << "Input dimensions for matrices don't match" << endl;
        return;
    }

    // Check if output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check dimensions match
        if (this->rows != output.rows || this->cols != output.cols)
        {
            cerr << "Input dimensions for matrices don't match" << endl;
            return;
        }
    }
    // Pre-allocate space to output matrix if it hasn't already been
    else
    {
        output.values = new T[this->rows * mat_right.cols];
        output.preallocated = true;
    }

    // Set values to zero before hand
    for (int i = 0; i < output.size_of_values; i++)
    {
        output.values[i] = 0;
    }

    // Multiplication algorithm (row major ordering)
    for (int i = 0; i < size_of_values; i++)
    {
        output.values[i] = this->values[i] + mat_right.values[i];
    }
}

/*
Solve Ax=b using in-place LU decompostion (input matrix overwritten)
*/
template <class T>
void Matrix<T>::LU_solve(T *RHS_vec, T *sol_vec)
{
    // Gaussian elimination and row swapping variables
    double pivot_val, row_op_SF;
    double max_val, cur_val;
    int max_row, cur_row;
    // array to keep track of row swaps
    unique_ptr<int[]> row_swaps(new int[this->rows]);
    for (int i = 0; i < this->rows; i++)
        row_swaps[i] = i;

    // loop through each pivot row
    for (int k = 0; k < this->rows - 1; k++)
    {
        // determine row number of the maximum value below the pivot value
        max_row = k;
        max_val = abs(this->values[k * (this->cols + 1)]);
        for (int r = k + 1; r < this->rows; r++)
        {
            cur_val = abs(this->values[r * this->cols + k]);
            if (cur_val > max_val)
            {
                max_row = r;
                max_val = cur_val;
            }
        }
        // max_row = cblas_idamax(this->rows - k, (this->values + k * this->cols + k), this->cols) + k; // *** ALTERNATIVE USING BLAS ***

        // swap rows
        this->swapRows(k, max_row);
        // cblas_dswap(this->cols, (this->values + k * this->cols), 1, (this->values + max_row * this->cols), 1); // *** ALTERNATIVE USING BLAS ***

        // encode the row swap
        cur_row = row_swaps[k];
        row_swaps[k] = row_swaps[max_row];
        row_swaps[max_row] = cur_row;

        // Perform Gaussian elimination
        pivot_val = this->values[k * (this->cols + 1)]; // diagonal entries as pivots
        for (int i = k + 1; i < this->rows; i++)        // loop through rows below the pivot row
        {
            // scale factor for row operations to get zeros below pivot value
            row_op_SF = this->values[i * this->cols + k] / pivot_val;
            // do row operation to update U entries
            this->daxpy(this->cols - k, -1 * row_op_SF, (this->values + k * (this->cols + 1)), 1, (this->values + i * this->cols + k), 1);
            // cblas_daxpy(this->cols - k, -1 * row_op_SF, (this->values + k * (this->cols + 1)), 1, (this->values + i * this->cols + k), 1); // *** ALTERNATIVE USING BLAS ***

            // Update L entries
            this->values[i * this->cols + k] = row_op_SF;
        }
    }

    // Now solve for x in two stages
    // Set Ux = y  -->  solve Ly = b for y using forward substituion
    double val_sum = 0;

    unique_ptr<T[]> y_vec(new T[this->rows]);
    for (int k = 0; k < this->rows; k++)
    {
        val_sum = 0;
        for (int j = 0; j < k; j++)
        {
            val_sum += this->values[k * this->cols + j] * y_vec[j];
        }
        // below implicitly assuming the 1's on the diagonal for L-matrix
        // and ensuring row swaps also applied to RHS vector(b)
        y_vec[k] = (RHS_vec[row_swaps[k]] - val_sum);
    }

    // Now solve Ux = y for x using backward substituion
    for (int k = this->rows - 1; k >= 0; k--)
    {
        val_sum = 0;
        for (int j = k + 1; j < this->cols; j++)
        {
            val_sum += this->values[k * this->cols + j] * sol_vec[j];
        }
        sol_vec[k] = (y_vec[k] - val_sum) / this->values[k * this->cols + k];
    }

    // // *** ALTERNATIVE USING BLAS *** (replaces the two elimination sub-routines above)
    // for (int k = 0; k < this->rows; k++)
    //     sol_vec[k] = RHS_vec[row_swaps[k]];
    // cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, this->rows, this->values, this->rows, sol_vec, 1);
    // cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, this->rows, this->values, this->rows, sol_vec, 1);
}

/*
Solve Ax=b using Simultaneous Over-Relaxation method. 
*/
template <class T>
int Matrix<T>::SOR(T *RHS_vec, T *sol_vec, int it_max, double tol)
{
    // const double tol = 1e-6;
    double res = 0., res2 = 0., res_prv = 0., sum = 0., omega = 1.0;
    unique_ptr<double[]> Ax(new double[rows]);
    int k_relax = 10;

    // begin iterations
    for (int k = 0; k < it_max; k++)
    {
        // calculate (discontinuous) dot product of each row of A with x
        for (int i = 0; i < this->rows; i++)
        {
            sum = this->ddot(this->cols, (this->values + i * this->cols), 1, sol_vec, 1);
            sum -= this->values[i * this->cols + i] * sol_vec[i];
            // update solution vector
            sol_vec[i] = (omega / this->values[i * this->cols + i]) * (RHS_vec[i] - sum) + (1 - omega) * sol_vec[i];
        }

        // calculate residual
        res2 = 0;
        this->matVecMult(sol_vec, Ax.get());
        for (int l = 0; l < this->rows; l++)
        {
            res2 += pow(Ax[l] - RHS_vec[l], 2);
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
Solve Ax=b using Conjugate Gradient method. 
*/
template <class T>
void Matrix<T>::ConjGrad(T *RHS_vec, T *sol_vec, int it_max, double tol)
{
    // initialising the residual and conjugate vectors
    // and calculate the starting residual
    unique_ptr<double[]> residual(new double[this->rows]);
    unique_ptr<double[]> conjugate(new double[this->rows]);

    unique_ptr<double[]> init_RHS(new T[this->rows]);
    this->matVecMult(sol_vec, init_RHS.get());
    double res_old = 0;

    for (int i = 0; i < this->rows; i++)
    {
        residual[i] = RHS_vec[i] - init_RHS[i];
        conjugate[i] = residual[i];
        res_old += residual[i] * residual[i];
    }

    // intialise loop variables
    double res_new = 0;
    double alpha = 0;
    unique_ptr<double[]> temp_vec(new double[this->rows]);

    int iter = 0;
    while (res_old > tol)
    {
        // Compute conjugate gradient coefficient
        this->matVecMult(conjugate.get(), temp_vec.get());
        alpha = res_old / this->ddot(this->rows, conjugate.get(), 1, temp_vec.get(), 1);

        // updating the solution and residual vectors
        this->daxpy(this->rows, alpha, conjugate.get(), 1, sol_vec, 1);
        this->daxpy(this->rows, -alpha, temp_vec.get(), 1, residual.get(), 1);

        // Compute new conjugate gradient coefficient
        res_new = this->ddot(this->rows, residual.get(), 1, residual.get(), 1);

        // updating conjugate vector
        for (int j = 0; j < this->rows; j++)
        {
            conjugate[j] = residual[j] + (res_new / res_old) * conjugate[j];
        }

        res_old = res_new;
        iter++;
        if (iter > it_max)
            break;
    }
    cout << "Iterations: " << iter << endl;
}

/*
Solve Ax = b using Multigrid V-Cycle with a Gauss-Seidel smoother
*/
template <class T>
void Matrix<T>::MGV(T *RHS_vec, T *sol_vec)
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
        // define all required variables
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

        // create restrict & interpolation operator
        shared_ptr<Matrix<double>> restrict_mat(new Matrix<double>(coarse_size, this->cols, true));
        shared_ptr<Matrix<double>> interp_mat(new Matrix<double>(this->cols, coarse_size, true));
        for (int i = 0; i < restrict_mat.get()->rows; i++)
        {
            for (int j = 0; j < restrict_mat.get()->cols; j++)
            {
                if (j == i * 2 || j == (i + 1) * 2)
                {
                    restrict_mat.get()->values[i * this->cols + j] = 0.25;
                    interp_mat.get()->values[j * coarse_size + i] = 0.5;
                }
                else if (j == i * 2 + 1)
                {
                    restrict_mat.get()->values[i * this->cols + j] = 0.5;
                    interp_mat.get()->values[j * coarse_size + i] = 1;
                }
                else
                {
                    restrict_mat.get()->values[i * this->cols + j] = 0;
                    interp_mat.get()->values[j * coarse_size + i] = 0;
                }
            }
        }

        // restrict residual to coarser grid
        restrict_mat.get()->matVecMult(res.get(), res_coarse.get());

        // create matrix of A on coarser grid using restriction and interpolation operators
        shared_ptr<Matrix<T>> coarse_A(new Matrix<T>(coarse_size, coarse_size, true));
        shared_ptr<Matrix<T>> middle_man(new Matrix<T>(coarse_size, this->cols, true));

        restrict_mat.get()->matMatMult(*this, *middle_man.get());
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
