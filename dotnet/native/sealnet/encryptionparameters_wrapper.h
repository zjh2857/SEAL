// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

///////////////////////////////////////////////////////////////////////////
//
// This API is provided as a simple interface for Microsoft SEAL library
// that can be PInvoked by .Net code.
//
///////////////////////////////////////////////////////////////////////////

#include "sealnet/defines.h"
#include <stdint.h>

SEALNETNATIVE HRESULT SEALCALL EncParams_Create1(int scheme, void **enc_params);

SEALNETNATIVE HRESULT SEALCALL EncParams_Create2(void *copy, void **enc_params);

SEALNETNATIVE HRESULT SEALCALL EncParams_Destroy(void *thisptr);

SEALNETNATIVE HRESULT SEALCALL EncParams_Set(void *thisptr, void *assign);

SEALNETNATIVE HRESULT SEALCALL EncParams_GetPolyModulusDegree(void *thisptr, uint64_t *degree);

SEALNETNATIVE HRESULT SEALCALL EncParams_SetPolyModulusDegree(void *thisptr, uint64_t degree);

SEALNETNATIVE HRESULT SEALCALL EncParams_GetCoeffModulus(void *thisptr, uint64_t *length, void **coeffs);

SEALNETNATIVE HRESULT SEALCALL EncParams_SetCoeffModulus(void *thisptr, uint64_t length, void **coeffs);

SEALNETNATIVE HRESULT SEALCALL EncParams_GetScheme(void *thisptr, int *scheme);

SEALNETNATIVE HRESULT SEALCALL EncParams_GetPlainModulus(void *thisptr, void **plain_modulus);

SEALNETNATIVE HRESULT SEALCALL EncParams_SetPlainModulus1(void *thisptr, void *modulus);

SEALNETNATIVE HRESULT SEALCALL EncParams_SetPlainModulus2(void *thisptr, uint64_t plainModulus);

SEALNETNATIVE HRESULT SEALCALL EncParams_NoiseStandardDeviation(void *thisptr, double *nsd);

SEALNETNATIVE HRESULT SEALCALL EncParams_SetNoiseStandardDeviation(void *thisptr, double nsd);

SEALNETNATIVE HRESULT SEALCALL EncParams_NoiseMaxDeviation(void *thisptr, double *nmd);

SEALNETNATIVE HRESULT SEALCALL EncParams_Equals(void *thisptr, void *otherptr, bool *result);
