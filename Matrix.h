//#pragma once
#ifndef MATRIX_H
#define MATRIX_H

template <class T>
class Matrix
{
public:
    // constructor where we want to preallocate ourselves
    Matrix(int rows, int cols, bool preallocate);
    // constructor where we already have allocated memory outside
    Matrix(int rows, int cols, T *values_ptr);
    // copy constructor
    Matrix(const Matrix<T> &Mat_to_copy);
    // destructor
    virtual ~Matrix();

    // Operator Overloading
    Matrix *operator*(Matrix &output); // multiplication
    Matrix *operator+(Matrix &output); // addition

    // Generate pseudo-random values
    void FillValues();
    void FillValuesSPD();

    // Print out the values in our matrix
    virtual void printValues();
    virtual void printMatrix();

    // Perform some operations with our matrix
    void daxpy(int n, double alpha, T *dx, int incx, T *dy, int incy);
    void swapRows(int row1, int row2);
    T ddot(int n, T *dx, int incx, T *dy, int incy);
    virtual void matVecMult(T *input_vec, T *output_vec);
    void matMatMult(Matrix &mat_right, Matrix &output, bool transpose_mat_right = false);
    void matMatAdd(Matrix &mat_right, Matrix &output);

    // Linear solver routines
    void LU_solve(T *RHS_vec, T *sol_vec);
    int SOR(T *RHS_vec, T *sol_vec, int it_max, double tol);
    virtual void ConjGrad(T *RHS_vec, T *sol_vec, int it_max, double tol); // same method for Matrix and CSRMatrix
    void MGV(T *RHS_vec, T *sol_vec);

    // Public matrix data
    T *values = nullptr;
    int rows = -1;
    int cols = -1;

    // We want our subclass to know about this
protected:
    bool preallocated = false;

    // Private variables - there is no need for other classes
    // to know about these variables
private:
    int size_of_values = -1;
};

#endif
