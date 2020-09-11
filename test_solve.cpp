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

using namespace std;

void TestDenseLUSolve(int k)
{
    cout << "--------------------------" << endl;
    cout << "   TEST DENSE LU SOLVE" << endl;
    cout << "--------------------------" << endl;

    // define size of matrix
    int rows = int(pow(2, k));
    int cols = rows;
    cout << "matrix size: " << rows << " x " << cols << endl;

    // Create matrix object with pre-allocated space
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
    // Fill matrix values
    dense_mat.get()->FillValuesSPD();

    // create deep copy for testing purposes (using copy constructor)
    shared_ptr<Matrix<double>> copy_mat(new Matrix<double>(*dense_mat.get()));

    // define RHS vector of linear system
    unique_ptr<double[]> RHS_vec(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < rows; i++)
    {
        RHS_vec[i] = (rand() % 100);
    }

    // solve linear system with LU decomposition (and report runtime)
    unique_ptr<double[]> sol_vec(new double[rows]);
    clock_t start = clock();
    dense_mat.get()->LU_solve(RHS_vec.get(), sol_vec.get()); // SOLVER CALLED HERE
    clock_t end = clock();

    printf("runtime (ms): %.0f", (double)(end - start) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    // when multiplied by the original coefficients matrix
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    copy_mat.get()->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;
}

void TestDenseSOR(int k)
{
    cout << "--------------------------" << endl;
    cout << "      TEST DENSE SOR" << endl;
    cout << "--------------------------" << endl;

    // define size of matrix
    int rows = int(pow(2, k));
    int cols = rows;
    cout << "matrix size: " << rows << " x " << cols << endl;

    // Create matrix object with pre-allocated space
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
    // Fill matrix values
    dense_mat.get()->FillValuesSPD();

    // // define RHS vector of linear system
    unique_ptr<double[]> RHS_vec(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < rows; i++)
    {
        RHS_vec[i] = (rand() % 100);
    }

    // define initial guess for solution
    unique_ptr<double[]> sol_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        sol_vec[i] = 0;
    }

    // solve linear system with SOR (and report runtime)
    clock_t start = clock();
    int iter = dense_mat.get()->SOR(RHS_vec.get(), sol_vec.get(), 1000, 1e-6); // SOLVER CALLED HERE
    clock_t end = clock();

    cout << "Iterations: " << iter << endl;
    printf("runtime (ms): %.0f", (double)(end - start) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    // when multiplied by the original coefficients matrix
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    dense_mat.get()->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;
}

void TestSparseSOR(int k, double dens)
{
    cout << "--------------------------" << endl;
    cout << "      TEST SPARSE SOR" << endl;
    cout << "--------------------------" << endl;

    // Define matrix
    int rows = int(pow(2, k));
    int cols = rows;
    int tmp = int(dens * rows * cols);
    int nnzs = tmp - tmp % 2;

    unique_ptr<int[]> irows(new int[nnzs]);
    unique_ptr<int[]> jcols(new int[nnzs]);
    unique_ptr<double[]> vals(new double[nnzs]);

    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
    sparse_mat.get()->genValuesSPD(irows.get(), jcols.get(), vals.get());
    sparse_mat.get()->setValues(irows.get(), jcols.get(), vals.get());

    cout << "matrix size: nz=" << nnzs << " (" << rows << " x " << cols << ")" << endl;

    // define RHS vector of linear system
    srand(time(nullptr));
    unique_ptr<double[]> RHS_vec(new double[rows]);
    for (int i = 0; i < cols; i++)
    {
        RHS_vec[i] = rand() % 100;
    }

    // define initial guess for solution
    unique_ptr<double[]> sol_vec(new double[rows]);
    for (int i = 0; i < cols; i++)
    {
        sol_vec[i] = 0;
    }

    // solve linear system with SOR (and report runtime)
    clock_t start1 = clock();
    int iter = sparse_mat.get()->SOR(RHS_vec.get(), sol_vec.get(), 1000, 1e-6); // SOLVER CALLED HERE
    clock_t end1 = clock();

    cout << "Iterations: " << iter << endl;
    printf("runtime (ms): %.0f", (double)(end1 - start1) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    // when multiplied by the original coefficients matrix
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    sparse_mat.get()->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;
}

void TestDenseCG(int k)
{
    cout << "--------------------------" << endl;
    cout << "      TEST DENSE CG" << endl;
    cout << "--------------------------" << endl;

    int rows = int(pow(2, k));
    int cols = rows;
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
    dense_mat->FillValuesSPD(); // symmetric postive definite

    cout << "matrix size: " << rows << " x " << cols << endl;

    // define RHS vector of linear system
    unique_ptr<double[]> RHS_vec(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < cols; i++)
    {
        RHS_vec[i] = rand() % 100;
    }

    // define initial guess
    unique_ptr<double[]> sol_vec(new double[rows]);
    for (int i = 0; i < cols; i++)
    {
        sol_vec[i] = 0;
    }

    // solve linear system with CG (and report runtime)
    clock_t start1 = clock();
    dense_mat.get()->ConjGrad(RHS_vec.get(), sol_vec.get(), 1000, 1e-10); // SOLVER CALLED HERE
    clock_t end1 = clock();

    printf("runtime (ms): %.0f", (double)(end1 - start1) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    // when multiplied by the original coefficients matrix
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    dense_mat.get()->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;
}

void TestSparseCG(int k, double dens)
{
    cout << "--------------------------" << endl;
    cout << "      TEST SPARSE CG" << endl;
    cout << "--------------------------" << endl;

    // Define matrix
    int rows = int(pow(2, k));
    int cols = rows;
    int tmp = int(dens * rows * cols);
    int nnzs = tmp - tmp % 2;

    unique_ptr<int[]> irows(new int[nnzs]);
    unique_ptr<int[]> jcols(new int[nnzs]);
    unique_ptr<double[]> vals(new double[nnzs]);

    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
    sparse_mat.get()->genValuesSPD(irows.get(), jcols.get(), vals.get());
    sparse_mat.get()->setValues(irows.get(), jcols.get(), vals.get());

    cout << "matrix size: nz=" << nnzs << " (" << rows << "x" << cols << ")" << endl;

    // define RHS vector of linear system
    unique_ptr<double[]> RHS_vec(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < cols; i++)
    {
        RHS_vec[i] = (rand() % 100);
    }

    // define initial guess
    unique_ptr<double[]> sol_vec(new double[rows]);
    for (int i = 0; i < cols; i++)
    {
        sol_vec[i] = 0;
    }

    // solve linear system with CG (and report runtime)
    clock_t start1 = clock();
    sparse_mat.get()->ConjGrad(RHS_vec.get(), sol_vec.get(), 1000, 1e-10); // SOLVER CALLED HERE
    clock_t end1 = clock();

    printf("runtime (ms): %.0f", (double)(end1 - start1) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    // when multiplied by the original coefficients matrix
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    sparse_mat.get()->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;

    // delete objects and arrays created on the heap
}

void TestDenseMGV()
{
    cout << "--------------------------" << endl;
    cout << "   TEST DENSE MULTIGRID   " << endl;
    cout << "--------------------------" << endl;

    // define size of matrix
    // rows & cols must be of the form 2^k-1, where k is some integer.
    int rows = 1023;
    int cols = 1023;

    // generate a banded matrix
    const int bandwidth = 1;
    double stencil[2 * bandwidth + 1] = {1, 8, 1};
    int nnzs = rows + 2 * bandwidth * rows - bandwidth * (bandwidth + 1);

    unique_ptr<int[]> irows(new int[nnzs]);
    unique_ptr<int[]> jcols(new int[nnzs]);
    unique_ptr<double[]> vals(new double[nnzs]);

    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
    sparse_mat.get()->genValuesBand(irows.get(), jcols.get(), vals.get(), bandwidth, stencil);
    sparse_mat.get()->setValues(irows.get(), jcols.get(), vals.get());

    // Create matrix object with pre-allocated space
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
    // Fill values
    sparse_mat.get()->sparse2dense(*dense_mat.get());

    cout << "matrix size: " << rows << " x " << cols << endl;

    // define RHS vector of linear system
    unique_ptr<double[]> RHS_vec(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < cols; i++)
    {
        RHS_vec[i] = rand() % 100;
    }
    // define initial guess for solution
    unique_ptr<double[]> sol_vec(new double[rows]);
    for (int i = 0; i < cols; i++)
    {
        sol_vec[i] = 0;
    }

    // solve linear system with SOR (and report runtime)
    clock_t start1 = clock();
    dense_mat.get()->MGV(RHS_vec.get(), sol_vec.get()); // SOLVER CALLED HERE
    clock_t end1 = clock();

    printf("runtime (ms): %.0f", (double)(end1 - start1) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    dense_mat->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;
}

void TestSparseMGV()
{
    cout << "--------------------------" << endl;
    cout << "   TEST SPARSE MULTIGRID  " << endl;
    cout << "--------------------------" << endl;

    // define size of matrix
    // rows & cols must be of the form 2^k-1, where k is some integer.
    int rows = 16383;
    int cols = 16383;

    // generate a banded matrix
    const int bandwidth = 1;
    double stencil[2 * bandwidth + 1] = {1, 8, 1};
    int nnzs = rows + 2 * bandwidth * rows - bandwidth * (bandwidth + 1);

    unique_ptr<int[]> irows(new int[nnzs]);
    unique_ptr<int[]> jcols(new int[nnzs]);
    unique_ptr<double[]> vals(new double[nnzs]);

    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
    sparse_mat.get()->genValuesBand(irows.get(), jcols.get(), vals.get(), bandwidth, stencil);
    sparse_mat.get()->setValues(irows.get(), jcols.get(), vals.get());

    cout << "matrix size: nz=" << nnzs << " (" << rows << "x" << cols << ")" << endl;

    // define RHS vector of linear system
    unique_ptr<double[]> RHS_vec(new double[rows]);
    srand(time(nullptr));
    for (int i = 0; i < cols; i++)
    {
        RHS_vec[i] = rand() % 100;
    }

    // define initial guess for solution
    unique_ptr<double[]> sol_vec(new double[rows]);
    for (int i = 0; i < cols; i++)
    {
        sol_vec[i] = 0;
    }

    // solve linear system with SOR (and report runtime)
    clock_t start1 = clock();
    sparse_mat.get()->MGV(RHS_vec.get(), sol_vec.get()); // SOLVER CALLED HERE
    clock_t end1 = clock();

    printf("runtime (ms): %.0f", (double)(end1 - start1) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Check that the solution vector gives back the RHS
    unique_ptr<double[]> ans_vec(new double[rows]);
    for (int i = 0; i < rows; i++)
    {
        ans_vec[i] = 0;
    }
    sparse_mat.get()->matVecMult(sol_vec.get(), ans_vec.get());

    cout << "solution gives back the input RHS? ";
    bool ans = true;
    for (int i = 0; i < rows; i++)
    {
        if (abs(RHS_vec[i] - ans_vec[i]) > 1e-6 || isnan(sol_vec[i]))
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;
}

void TestSparseMatMatMult()
{
    cout << "--------------------------" << endl;
    cout << "  TEST SPARSE MatMatMult" << endl;
    cout << "--------------------------" << endl;

    // Define Matrix A
    int rowA = 800;
    int colA = 1000;
    int nnzA = 78312;
    auto *irowA = new int[nnzA];
    auto *jcolA = new int[nnzA];
    auto *valA = new double[nnzA];
    auto *A_mat = new CSRMatrix<double>(rowA, colA, nnzA, true);
    A_mat->genValues(irowA, jcolA, valA);
    A_mat->setValues(irowA, jcolA, valA);

    // Define Matrix B
    int rowB = 1000;
    int colB = 1100;
    int nnzB = 43894;
    auto *irowB = new int[nnzB];
    auto *jcolB = new int[nnzB];
    auto *valB = new double[nnzB];
    auto *B_mat = new CSRMatrix<double>(rowB, colB, nnzB, true);
    B_mat->genValues(irowB, jcolB, valB);
    B_mat->setValues(irowB, jcolB, valB);

    cout << "Input matrices: nz=" << nnzA << " (" << rowA << "x" << colA << ") | nz=" << nnzB << " (" << rowB << "x" << colB << ")" << endl;

    // Do symbolic multiplication to determine sparsity of output (and report runtime)
    clock_t start1 = clock();
    int nnzC = A_mat->matMatMult_SYM(*B_mat); // SYMBOLIC MULTIPLICATION
    clock_t end1 = clock();

    cout << "Output matrix: nz=" << nnzC << " (" << rowA << "x" << colB << ")" << endl
         << endl;
    printf("runtime for symbolic (ms): %.0f", (double)(end1 - start1) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Initialise Matrix C
    auto *C_mat = new CSRMatrix<double>(rowA, colB, nnzC, true);
    // Do numeric multiplcation (and report run time)
    clock_t start2 = clock();
    A_mat->matMatMult(*B_mat, *C_mat); // NUMERIC MULTIPLICATION
    clock_t end2 = clock();

    printf("runtime for numeric (ms): %.0f", (double)(end2 - start2) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl;

    // Convert all matrices to dense and multiply using dense algorithm
    auto *dense_matA = new Matrix<double>(A_mat->rows, A_mat->cols, true);
    auto *dense_matB = new Matrix<double>(B_mat->rows, B_mat->cols, true);
    auto *dense_matC = new Matrix<double>(C_mat->rows, C_mat->cols, true);
    A_mat->sparse2dense(*dense_matA);
    B_mat->sparse2dense(*dense_matB);
    C_mat->sparse2dense(*dense_matC);

    clock_t start3 = clock();
    auto *dense_matC_test = new Matrix<double>(C_mat->rows, C_mat->cols, true);
    dense_matA->matMatMult(*dense_matB, *dense_matC_test);
    clock_t end3 = clock();

    printf("runtime for dense (ms): %.0f", (double)(end3 - start3) / (double)(CLOCKS_PER_SEC)*1000.0);
    cout << endl
         << endl;

    // Check that the dense and sparse multiplication algorithms
    // gives the same answer
    cout << "dense and sparse solutions the same? ";
    bool ans = true;
    for (int i = 0; i < rowA * colB; i++)
    {
        if (abs(dense_matC->values[i] - dense_matC_test->values[i]) > 1e-6)
        {
            ans = false;
            break;
        }
    }
    // print true or false for solution checking
    cout << boolalpha << ans << endl
         << endl;

    // delete objects and arrays created on the heap
    delete A_mat;
    delete dense_matA;
    delete B_mat;
    delete dense_matB;
    delete C_mat;
    delete dense_matC;
    delete dense_matC_test;

    delete[] irowA;
    delete[] irowB;
    delete[] jcolA;
    delete[] jcolB;
    delete[] valA;
    delete[] valB;
}
