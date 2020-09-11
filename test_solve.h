#ifndef TEST_SOLVE_H
#define TEST_SOLVE_H

void TestDenseLUSolve(int k);
void TestDenseSOR(int k);
void TestSparseSOR(int k, double dens);
void TestDenseCG(int k);
void TestSparseCG(int k, double dens);
void TestDenseMGV();
void TestSparseMGV();
void TestSparseMatMatMult();

#endif