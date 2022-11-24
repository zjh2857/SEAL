#pragma once
// #include "seal/memorymanager.h"
// #include "seal/modulus.h"
// #include "seal/util/defines.h"
// #include "seal/util/iterator.h"
// #include "seal/util/pointer.h"
// // #include "seal/util/hello.cuh"
#include "seal/util/uintarithsmallmod.h"
// #include "seal/util/uintcore.h"
#include <complex>
#include "seal/util/iterator.h"
// #include <stdexcept>
// using namespace std;
// using seal::util;
int test1(std::uint64_t *value,int log_n,struct seal::util::MultiplyUIntModOperand const * roots,std::uint64_t modulue);
int test1(std::complex<double> *value,int log_n,std::complex<double> const * roots,std::uint64_t modulue);
int Product(
            seal::util::ConstCoeffIter operand1, seal::util::ConstCoeffIter operand2, size_t coeff_count, const seal::Modulus &modulus,
            seal::util::CoeffIter result);