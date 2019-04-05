// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// SEALNet
#include "sealnet/stdafx.h"
#include "sealnet/encryptionparameters_wrapper.h"
#include "sealnet/utilities.h"

// SEAL
#include "seal/encryptionparams.h"
#include "seal/smallmodulus.h"
#include "seal/util/hash.h"

using namespace std;
using namespace seal;
using namespace sealnet;

SEALNETNATIVE HRESULT SEALCALL EncParams_Create1(int scheme, void **enc_params)
{
    IfNullRet(enc_params, E_POINTER);

    try
    {
        scheme_type schemetype = static_cast<scheme_type>(scheme);
        EncryptionParameters *params = new EncryptionParameters(schemetype);
        *enc_params = params;
        return S_OK;
    }
    catch (const invalid_argument&)
    {
        return E_INVALIDARG;
    }
}

SEALNETNATIVE HRESULT SEALCALL EncParams_Create2(void *copy, void **enc_params)
{
    EncryptionParameters *copypt = FromVoid<EncryptionParameters>(copy);
    IfNullRet(copypt, E_POINTER);
    IfNullRet(enc_params, E_POINTER);

    EncryptionParameters *params = new EncryptionParameters(*copypt);
    *enc_params = params;
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_Destroy(void *thisptr)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);

    delete params;
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_Set(void *thisptr, void *assign)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    EncryptionParameters *assignpt = FromVoid<EncryptionParameters>(assign);
    IfNullRet(assignpt, E_POINTER);

    *params = *assignpt;
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_GetPolyModulusDegree(void *thisptr, uint64_t *degree)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(degree, E_POINTER);

    *degree = params->poly_modulus_degree();
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_SetPolyModulusDegree(void *thisptr, uint64_t degree)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);

    try
    {
        params->set_poly_modulus_degree(degree);
        return S_OK;
    }
    catch (const invalid_argument&)
    {
        return E_INVALIDARG;
    }
}

SEALNETNATIVE HRESULT SEALCALL EncParams_GetCoeffModulus(void *thisptr, uint64_t *length, void **coeffs)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(length, E_POINTER);

    BuildCoeffPointers(params->coeff_modulus(), length, coeffs);
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_SetCoeffModulus(void *thisptr, uint64_t length, void **coeffs)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(coeffs, E_POINTER);

    SmallModulus* *coeff_array = reinterpret_cast<SmallModulus**>(coeffs);
    vector<SmallModulus> coefficients(length);

    for (uint64_t i = 0; i < length; i++)
    {
        coefficients[i] = *coeff_array[i];
    }

    try
    {
        params->set_coeff_modulus(coefficients);
        return S_OK;
    }
    catch (const invalid_argument&)
    {
        return E_INVALIDARG;
    }
}

SEALNETNATIVE HRESULT SEALCALL EncParams_GetScheme(void *thisptr, int *scheme)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(scheme, E_POINTER);

    *scheme = static_cast<int>(params->scheme());
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_GetPlainModulus(void *thisptr, void **plain_modulus)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(plain_modulus, E_POINTER);

    const auto plainmodulus = &params->plain_modulus();
    *plain_modulus = const_cast<SmallModulus*>(plainmodulus);
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_SetPlainModulus1(void *thisptr, void *modulus)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    SmallModulus *smallmodulus = FromVoid<SmallModulus>(modulus);
    IfNullRet(smallmodulus, E_POINTER);

    try
    {
        params->set_plain_modulus(*smallmodulus);
        return S_OK;
    }
    catch (const logic_error&)
    {
        return HRESULT_FROM_WIN32(ERROR_INVALID_OPERATION);
    }
}

SEALNETNATIVE HRESULT SEALCALL EncParams_SetPlainModulus2(void *thisptr, uint64_t plainModulus)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);

    try
    {
        params->set_plain_modulus(plainModulus);
        return S_OK;
    }
    catch (const logic_error&)
    {
        return HRESULT_FROM_WIN32(ERROR_INVALID_OPERATION);
    }
}

SEALNETNATIVE HRESULT SEALCALL EncParams_NoiseStandardDeviation(void *thisptr, double *nsd)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(nsd, E_POINTER);

    *nsd = params->noise_standard_deviation();
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_SetNoiseStandardDeviation(void *thisptr, double nsd)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);

    try
    {
        params->set_noise_standard_deviation(nsd);
        return S_OK;
    }
    catch (const invalid_argument&)
    {
        return E_INVALIDARG;
    }
}

SEALNETNATIVE HRESULT SEALCALL EncParams_NoiseMaxDeviation(void *thisptr, double *nmd)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    IfNullRet(nmd, E_POINTER);

    *nmd = params->noise_max_deviation();
    return S_OK;
}

SEALNETNATIVE HRESULT SEALCALL EncParams_Equals(void *thisptr, void *otherptr, bool *result)
{
    EncryptionParameters *params = FromVoid<EncryptionParameters>(thisptr);
    IfNullRet(params, E_POINTER);
    EncryptionParameters *other = FromVoid<EncryptionParameters>(otherptr);
    IfNullRet(other, E_POINTER);
    IfNullRet(result, E_POINTER);

    *result = (*params == *other);
    return S_OK;
}
