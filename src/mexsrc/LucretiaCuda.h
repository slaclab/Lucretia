#ifndef mex_h
  #define mex_h
  #include <mex.h>
#endif
#ifndef LUCRETIA_CUDA
  #ifdef __CUDACC__
    #include "gpu/mxGPUArray.h"
    #include "curand_kernel.h"
    #define CUDAHOSTFUN __host__
    #define CUDAGLOBALFUN __global__
    #define CUDAHOSTDEVICEFUN __host__ __device__
    #define TFLAG TFlag_gpu
    #define NGOODRAY ngoodray_gpu
    #include <curand_kernel.h>
    #include <cuda.h>
    #include <curand.h>
    #include "math_functions.h"
    #define threadsPerBlock 512
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
       if (code != cudaSuccess) 
       {
          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort)
            mexErrMsgTxt("CUDA Error");
       }
    }
  #else  
    #define CUDAHOSTFUN
    #define CUDAGLOBALFUN
    #define CUDAHOSTDEVICEFUN
    #define TFLAG TFlag
    #define NGOODRAY ngoodray
  #endif
#endif
#define LUCRETIA_CUDA