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
#include <string>
#include <fstream>
// #include <cblas.h>

#include "Matrix.h"
#include "Matrix.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"

#include "test_solve.h"

using namespace std;

template <typename T>
void solver(Matrix<T> *A, T *b, T *x, const string routine, const string matType, const int max_iter = 1000, const double tol = 1e-8)
{
    if (matType == "sparse")
    {
        cout << "Solving sparse linear system..." << endl;
        CSRMatrix<T> *As = static_cast<CSRMatrix<T> *>(A); // if sparse matrix entered, down-cast Matrix object to CSRMatrix object

        if (routine == "SOR")
            As->SOR(b, x, max_iter, tol);
        else if (routine == "CG")
            As->ConjGrad(b, x, max_iter, tol);
        else if (routine == "MGV")
            As->MGV(b, x);
        else
            cout << "Invalied routine for this matType... enter SOR, CG or MGV" << endl;
    }
    else if (matType == "dense")
    {
        cout << "Solving dense linear system..." << endl;

        if (routine == "LU")
            A->LU_solve(b, x);
        else if (routine == "SOR")
            A->SOR(b, x, max_iter, tol);
        else if (routine == "CG")
            A->ConjGrad(b, x, max_iter, tol);
        else if (routine == "MGV")
            A->MGV(b, x);
        else
            cout << "Invalied routine for this matType... enter LU, SOR, CG or MGV" << endl;
    }
    else
        cout << "matType must be entered as dense or sparse" << endl;
}

int main()
{
    //========================================================================
    //      DEFINE LINEAR SYSTEM HERE - EXAMPLE GIVEN FOR A CSR MATRIX
    //========================================================================

    // define matrix dimensions (rows, cols) and number of non-zeros
    int rows = 100;
    int cols = 100;
    int nnzs = 0.05 * rows * cols;

    // Initialise an empty, pre-allcocated matrix
    shared_ptr<CSRMatrix<double>> A(new CSRMatrix<double>(rows, cols, nnzs, true));

    unique_ptr<int[]> irows(new int[nnzs]);      // allocate space for row indices of non-zero values
    unique_ptr<int[]> jcols(new int[nnzs]);      // allocatie space for column indices of non-zero values
    unique_ptr<double[]> vals(new double[nnzs]); // allocation space for values of non-zero values

    // generate and set symmetric positive definite data for the matrix
    A.get()->genValuesSPD(irows.get(), jcols.get(), vals.get());
    A.get()->setValues(irows.get(), jcols.get(), vals.get());

    // define RHS vector of linear system
    unique_ptr<double[]> b(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < rows; i++)
    {
        b[i] = (rand() % 100);
    }

    //========================================================================
    // SOLVE LINEAR SYSTEM USING A CHOSEN ROUTINE FOR THE MATRIX TYPE DEFINED
    //========================================================================

    // initialise the solution vector of the linear system
    unique_ptr<double[]> x(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        x[i] = 0;
    }

    // the array x will be populated with the solution
    solver(A.get(), b.get(), x.get(), "SOR", "sparse");

    // output solution to text file
    ofstream ofs("solution.txt", ios::trunc);
    for (int i = 0; i < rows; i++)
        ofs << x[i] << ",";
    ofs.close();
}

////////////////////////////////////////////////////////////////////////////////
// code below was used for testing and performance analysis purposes, leaving it
// in here but outside main an commented out for clarity.

/*

//////////////////////////////////////////////////////////////////////////////////
////                          PERFORMANCE TESTING                             ////
//////////////////////////////////////////////////////////////////////////////////

ofstream ofs("./plotting/CG_dense2.txt", ios::trunc);
const int repeat = 5;

// const int bandwidth = 8;
// double stencil[2 * bandwidth + 1] = {1, 2, 3, 4, 5, 6, 7, 8, 100, 8, 7, 6, 5, 4, 3, 2, 1};

for (int k = 11; k < 16; k++)
{
    int rows = int(pow(2, k));
    int cols = rows;
    // int nnzs = rows + 2 * bandwidth * rows - bandwidth * (bandwidth + 1);

    shared_ptr<Matrix<double>> A(new Matrix<double>(rows, cols, true));
    // shared_ptr<CSRMatrix<double>> A(new CSRMatrix<double>(rows, cols, nnzs, true));
    // unique_ptr<int[]> irows(new int[nnzs]);
    // unique_ptr<int[]> jcols(new int[nnzs]);
    // unique_ptr<double[]> vals(new double[nnzs]);

    double time = 0, max = -1., min = 999999999., avg = 0;
    for (int j = 0; j < repeat; j++)
    {
        A.get()->FillValuesSPD();
        // A.get()->genValuesBand(irows.get(), jcols.get(), vals.get(), bandwidth, stencil);
        // A.get()->setValues(irows.get(), jcols.get(), vals.get());

        unique_ptr<double[]> b(new double[rows]);
        // srand(time(nullptr));
        for (int i = 0; i < rows; i++)
        {
            b[i] = (rand() % 100); // not using srand(time(nullptr)) therefore rand() is the same each time
        }

        unique_ptr<double[]> x(new double[rows]);
        for (int i = 0; i < cols; i++)
        {
            x[i] = 0;
        }

        clock_t start = clock();
        solver(A.get(), b.get(), x.get(), "CG", "dense");
        clock_t end = clock();

        time = (double)(end - start) / (double)(CLOCKS_PER_SEC)*1000.0;
        avg += time;
        if (time > max)
            max = time;
        if (time < min)
            min = time;
    }
    ofs << rows << "," << max << "," << min << "," << avg / repeat << endl;
}
ofs.close();

////////////////////////////////////////////////////////////////////////////////
// tests in test_solve.cpp

// TestDenseLUSolve(10);
// TestDenseSOR(14);
// TestSparseSOR(14, 0.05);
// TestDenseCG(14);
// TestSparseCG(15, 0.05);
// TestDenseMGV();
// TestSparseMGV();
// TestSparseMatMatMult();

*/
