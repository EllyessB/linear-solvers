# C++ Linear Solvers Group Project

## Introduction
This project required the design and implementation of a C++ library for solving linear systems Ax=b, where A is a positive definite matrix. In this library, the following solver routines have been implemented:
* LU decomposition (direct - dense only)
* Simultaneous Over Relaxation (iterative) 
* Conjugate Gradient (iterative)
* Multigrid (iterative)

## Installation:
Clone the repository from Github by either:
* Using command line:
`git clone https://github.com/EllyessB/linear-solvers.git`
* Downloading the repository as a .zip

## Usage:
Using Unix command line arguments for the g++ compiler from within the library directory, run:

```g++ -O3 *.cpp```

(where the ```-O3``` flag turns on compiler optimisations for improved performance). Then run:

```./a.out```

to execute the program.

## User Guide:
### Classes:
This library includes a base class **Matrix**, and a derived class **CSRMatrix**.
* The **Matrix** class is for **dense matrices** and requires a size defined by rows and columns which can be initialized using the class constructor:
```cpp
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
```
* The user may then fill in the **values** attribute according to row-major ordering, as follows:
```cpp
   for (int k = 0; k < rows * cols; k++) {
        dense_mat.get()->values[k] = some_value;
   }
```
* The **CSRMatrix** class is for **sparse matrices** in [CSR format](https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/). The required inputs to initialize are rows, columns and the number of non zero (nnzs) values:
```cpp
    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
```
* The user may then fill in the **values**, **row_position** and **col_index** attributes in a similar way to the above.

An alternative construction of these matrix objects may be used if the user does not wish to preallocate memory, but instead control memory themselves. In the alternative constructor, the attribute data is directly passed in as arguments.


### Solver:
To use the linear solvers, in the main.cpp file there is a **solver()** method which allows the user to call a linear solver to solve Ax=b. The procedure to use one of the solvers is as follows:
1. Initialise a matrix object of the desired size / dimensions (including nnzs for sparse matrices)
2. Populate the values (and row_position and col_index for sparse matrices) attributes for the system being considered (this may be done, for example, by using the matrix generators described below, or by any other means).
3. Define a RHS vector for the system and initialise a solution vector (with an appropriate initial guess for use with an iterative solver).
4. Call the solver `solver(A, b, x, routine, matType)`
    * matType is either "dense" or "sparse"
    * routine is either "LU", "SOR", "CG" or "MGV" (note that LU is only available for dense matrices)
5. A text file (solution.txt) is output containing the entries of the solution vector.

Here is an example using a sparse matrix:
```cpp
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
        ofs << x[i] << endl;
```

**Note** that if using an iterative solver, the linear system must be appropriate for the solver to guarantee convergence:
* **Simultaneous Over Relaxation**
    * Symmetric Positive Definite
* **Conjugate Gradient**
    * Symmetric Positive Definite
* **Multigrid**
    * Banded, diagonally dominant and symmetric
    * Dimension must be `2^k - 1` where `k` is an integer.


### Matrix Generators:
Included in the library are three matrix generators for the following types:
* **Dense symmetric positive definite matrix** can be produced using the Matrix class method **FillValuesSPD** as displayed below. The values are filled with pseudo-randomly generated numbers.
```cpp
    int rows = 1000;
    int cols = 1000;
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
    dense_mat->FillValuesSPD(); 
```

* **Sparse symmetric positive definite matrix in CSR format** can be produced using the CSRMatrix class method **genValuesSPD** and **setValues** to input the values in CSR format as displayed below. The values are filled with psuedo-randomly generated numbers at psuedo-random locations in the lower triangles, which is mirrored in the upper triangle. The diagonal is filled such that the resulting matrix is diagonally dominant.
```cpp
    int rows = 1000;
    int cols = 1000;
    int nnzs = 0.05 * rows * cols;

    unique_ptr<int[]> irows(new int[nnzs]);
    unique_ptr<int[]> jcols(new int[nnzs]);
    unique_ptr<double[]> vals(new double[nnzs]);

    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
    sparse_mat.get()->genValuesSPD(irows.get(), jcols.get(), vals.get());
    sparse_mat.get()->setValues(irows.get(), jcols.get(), vals.get());
```

* **Banded matrix** can be produced using the CSRMatrix class method **genValuesBand** as displayed below. The values are filled according to an input bandwidth and stencil. 
```cpp
    int rows = 1000;
    int cols = 1000;

    // generate a banded matrix
    const int bandwidth = 1;
    double stencil[2 * bandwidth + 1] = {1, -2, 1};
    int nnzs = rows + 2 * bandwidth * rows - bandwidth * (bandwidth + 1);

    unique_ptr<int[]> irows(new int[nnzs]);
    unique_ptr<int[]> jcols(new int[nnzs]);
    unique_ptr<double[]> vals(new double[nnzs]);

    shared_ptr<CSRMatrix<double>> sparse_mat(new CSRMatrix<double>(rows, cols, nnzs, true));
    sparse_mat.get()->genValuesBand(irows.get(), jcols.get(), vals.get(), bandwidth, stencil);
    sparse_mat.get()->setValues(irows.get(), jcols.get(), vals.get());
```
Sparse matrices can be converted to dense matrices using the CSRMatrix class method **sparse2dense**:
```cpp
    // Create matrix object with pre-allocated space
    shared_ptr<Matrix<double>> dense_mat(new Matrix<double>(rows, cols, true));
    // Fill values
    sparse_mat.get()->sparse2dense(*dense_mat.get());
```

## Further information:

More detailed documentation for each method in the code can be found in the relevant .cpp and .h files.

## Credits:
Group: Ollie Bell, Ellyess Benmoufok and Nadya Mere
