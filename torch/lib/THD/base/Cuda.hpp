#pragma once

#ifdef WITH_CUDA
#include <THC/THC.h>

extern THCState** _THDCudaState;

inline THCState* THDGetCudaState() {
  return *_THDCudaState;
}

int THDGetStreamId(cudaStream_t stream);

#include "Cuda.h"
#endif
