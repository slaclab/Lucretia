# 
# Makefile for construction of Lucretia mex files
#  Written for use on a Linux OS (EL5+ & SL5+ 64-bit supported)
#  - edit MATPATH variable to link to the location of your
#    Matlab  installation
#  - mex should be on your search path (ie ${MATPATH}/bin)
#  - type "make clean" to cleanup any previous build attempts
#  - type "make" to build mex executable files
#  - type "make install" to move mex files to their correct
#    installation directories
#  - type "make gpu" to build mex files which support GPU operation
#    requires >R2013a, compiled mex files have _gpu suffix
#  - type "make g4track" to build (cpu) version of TrackThru which
#    also allows interfacing to GEANT4. This requires GEANT4 (v.10+)
#    installed fully (with "geant4-config" script on the system search
#    path)
#
# Auth: GRW: Aug 4 2011
# Mod:  GRW: Jun 10 2014
#         - Made Makefile more OS & build environment independent
#       GRW: Apr 2 2014
#         - Added compilation steps for g4track
#       GRW: July 17 2013
#         - Added GPU compile steps
#
#===================================================================

# ===== Link to matlab installation directory (needed for GPU compilation)
MATPATH = /var/MATLAB/R2016a
# ===== mex compiler to use
MCOMP:= mex -v# used to build mex files
CCOMP:= mex -c -v# used for compiling library objects
# ===============================
# Support Linux or OSX (64-bit systems only)
ifeq ($(shell uname),Linux)
  MEXEXT = mexa64
else
  MEXEXT = mexmaci64
endif  
# include -DLUCRETIA_DPREC to compile for double precision, else single
MFLAGS = -DLUCETIA_DPREC
# include -DLUCRETIA_MLRAND to utilize Matlab's random number generation (slow) else use C generator (faster)
# MFLAGS := ${MFLAGS} -DLUCRETIA_MLRAND
G4LIBS = $(shell geant4-config --libs)
G4FLAGS = $(shell geant4-config --cflags)
NVCC = nvcc# Cuda compiler

LMAIN  = libLucretiaMain.a
LMATL  = libLucretiaMatlab.a
LG4LIB = libg4track.a

cpu : GetRmats.$(MEXEXT) GetTwiss.$(MEXEXT) RmatAtoB.$(MEXEXT) TrackThru.$(MEXEXT) VerifyLattice.$(MEXEXT)
g4track : clean g4_init g4track/libg4track.a TrackThru_g4.$(MEXEXT)
gpu : gpu_init TrackThru_gpu.$(MEXEXT)

install::
	-mv GetRmats.$(MEXEXT) ../RMatrix
	-mv GetTwiss.$(MEXEXT) ../Twiss
	-mv RmatAtoB.$(MEXEXT) ../RMatrix
	-mv TrackThru.$(MEXEXT) ../Tracking
	-mv VerifyLattice.$(MEXEXT) ../LatticeVerification
	-mv TrackThru_gpu.$(MEXEXT) ../Tracking
	-mv TrackThru_g4.$(MEXEXT) ../Tracking/TrackThru.$(MEXEXT)

cpu_init::

gpu_init::
  MFLAGS=-DLUCETIA_DPREC CC=${NVCC} CXX=${NVCC}
#  MFLAGS="${MFLAGS} CFLAGS='--relocatable-device-code=true -gencode=arch=compute_20,code=sm_20 -I$TMW_ROOT/toolbox/distcomp/gpu/extern/include --compiler-options=-ansi,-D_GNU_SOURCE,-fPIC,-fno-omit-frame-pointer,-pthread'"
#  MFLAGS="${MFLAGS} CLIBS='$$RPATH $$MLIBS -lm -lmwgpu $$TMW_ROOT/bin/$Arch/libcudart.so.5.0'"
#  MFLAGS="${MFLAGS} CXX='$NVCC'"
#  MFLAGS="${MFLAGS} CXXFLAGS='--relocatable-device-code=true -gencode=arch=compute_20,code=sm_20 -I$TMW_ROOT/toolbox/distcomp/gpu/extern/include --compiler-options=-ansi,-D_GNU_SOURCE,-fPIC,-fno-omit-frame-pointer,-pthread'"
#  MFLAGS="${MFLAGS} CXXLIBS='$$RPATH $$MLIBS -lm -lmwgpu $$TMW_ROOT/bin/$Arch/libcudart.so.5.0'"

.PHONY : g4_init

g4_init:
	-cp TrackThru.c TrackThru_g4.c

# Mex files:
GetRmats.$(MEXEXT) : $(LMAIN) $(LMATL) GetRmats.c
	$(MCOMP) $(MFLAGS) GetRmats.c -L. -lLucretiaMatlab -lLucretiaMain
GetTwiss.$(MEXEXT) : $(LMAIN) $(LMATL) GetTwiss.c
	$(MCOMP) $(MFLAGS) GetTwiss.c -L. -lLucretiaMatlab -lLucretiaMain
RmatAtoB.$(MEXEXT) : $(LMAIN) $(LMATL) RmatAtoB.c
	$(MCOMP) $(MFLAGS) RmatAtoB.c -L. -lLucretiaMatlab -lLucretiaMain
TrackThru.$(MEXEXT) : $(LMAIN) $(LMATL) TrackThru.c
	$(MCOMP) $(MFLAGS) TrackThru.c -L. -lLucretiaMatlab -lLucretiaMain
VerifyLattice.$(MEXEXT) : $(LMAIN) $(LMATL) VerifyLattice.c
	$(MCOMP) $(MFLAGS) VerifyLattice.c -L. -lLucretiaMatlab -lLucretiaMain
TrackThru_gpu.$(MEXEXT) : GPULINK TrackThru_gpu.o
	g++ -O -pthread -shared -Wl,--version-script,$(MATPATH)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -o  "TrackThru_gpu.$(MEXEXT)"  TrackThru_gpu.o  "LucretiaMatlab_gpu.o"  "LucretiaMatlabErrMsg_gpu.o"  "LucretiaPhysics_gpu.o"  "LucretiaCommon_gpu.o" "gpulink.o"  -Wl,-rpath-link,$(MATPATH)/bin/glnxa64 -L$(MATPATH)/bin/glnxa64 -lmx -lmex -lmat -lm -lmwgpu $(MATPATH)/bin/glnxa64/libcudart.so.5.0
TrackThru_g4.$(MEXEXT) : $(LMAIN) $(LMATL) g4track/$(LG4LIB) TrackThru_g4.c
	$(MCOMP) $(MFLAGS) -v TrackThru_g4.c -L. -lLucretiaMatlab -lLucretiaMain -Lg4track/ -lg4track $(G4LIBS)
	rm TrackThru_g4.c

# LucretiaMain library:

 $(LMAIN) : $(LMATL)\
	$(LMAIN)(LucretiaCommon.o) \
	$(LMAIN)(LucretiaPhysics.o)

# LucretiaMatlab library

 $(LMATL) : \
	$(LMATL)(LucretiaMatlab.o) \
	$(LMATL)(LucretiaMatlabErrMsg.o)

GPULINK : LucretiaPhysics_gpu.o LucretiaCommon_gpu.o LucretiaMatlab_gpu.o LucretiaMatlabErrMsg_gpu.o TrackThru_gpu.o
	$(NVCC) -gencode=arch=compute_20,code=sm_20 -dlink -rdc=true --compiler-options=-ansi,-D_GNU_SOURCE,-fPIC,-fno-omit-frame-pointer,-pthread LucretiaMatlab_gpu.o LucretiaMatlabErrMsg_gpu.o LucretiaPhysics_gpu.o LucretiaCommon_gpu.o TrackThru_gpu.o -o gpulink.o

clean::
	-rm -f *.o
	-rm -f *.lst
	-rm -f *.a
	-rm -f *.$(MEXEXT)

# now for the next level of dependencies:

$(LMAIN)(LucretiaCommon.o) : \
   LucretiaCommon.c LucretiaCommon.h LucretiaPhysics.h \
   LucretiaGlobalAccess.h LucretiaVersionProto.h \
   LucretiaDictionary.h
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) LucretiaCommon.c
	ar -ru $(LMAIN) LucretiaCommon.o
	rm LucretiaCommon.o


$(LMAIN)(LucretiaPhysics.o) : \
   LucretiaPhysics.c LucretiaCommon.h LucretiaPhysics.h \
   LucretiaGlobalAccess.h LucretiaVersionProto.h 
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) LucretiaPhysics.c
	ar -ru $(LMAIN) LucretiaPhysics.o
	rm LucretiaPhysics.o


$(LMATL)(LucretiaMatlab.o) : \
   LucretiaMatlab.c LucretiaGlobalAccess.h LucretiaMatlab.h \
   LucretiaVersionProto.h
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) -I$(MATPATH)/extern/include LucretiaMatlab.c
	ar -ru $(LMATL) LucretiaMatlab.o
	rm LucretiaMatlab.o


$(LMATL)(LucretiaMatlabErrMsg.o) : \
   LucretiaMatlabErrMsg.c LucretiaGlobalAccess.h
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) -I$(MATPATH)/extern/include LucretiaMatlabErrMsg.c
	ar -ru $(LMATL) LucretiaMatlabErrMsg.o
	rm LucretiaMatlabErrMsg.o


# GPU object file dependencies:

LucretiaCommon_gpu.o : \
   LucretiaCommon.c LucretiaCommon.h LucretiaPhysics.h \
   LucretiaGlobalAccess.h LucretiaVersionProto.h \
   LucretiaDictionary.h 
	-cp LucretiaCommon.c LucretiaCommon_gpu.cu
	sed -i 's/void TrackBunchThruDrift_kernel/__global__ void TrackBunchThruDrift_kernel/g' LucretiaCommon_gpu.cu
	sed -i 's/void TrackBunchThruQSOS_kernel/__global__ void TrackBunchThruQSOS_kernel/g' LucretiaCommon_gpu.cu
	sed -i 's/void TrackBunchThruMult_kernel/__global__ void TrackBunchThruMult_kernel/g' LucretiaCommon_gpu.cu
	sed -i 's/void TrackBunchThruSBend_kernel/__global__ void TrackBunchThruSBend_kernel/g' LucretiaCommon_gpu.cu
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) LucretiaCommon_gpu.cu


LucretiaPhysics_gpu.o : \
   LucretiaPhysics.c LucretiaCommon.h LucretiaPhysics.h \
   LucretiaGlobalAccess.h LucretiaVersionProto.h 
	-cp LucretiaPhysics.c LucretiaPhysics_gpu.cu
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) LucretiaPhysics_gpu.cu

LucretiaMatlab_gpu.o : \
   LucretiaMatlab.c LucretiaGlobalAccess.h LucretiaMatlab.h \
   LucretiaVersionProto.h
	-cp LucretiaMatlab.c LucretiaMatlab_gpu.cu
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) -I$(MATPATH)/extern/include LucretiaMatlab_gpu.cu


LucretiaMatlabErrMsg_gpu.o : \
   LucretiaMatlabErrMsg.c LucretiaGlobalAccess.h
	-cp LucretiaMatlabErrMsg.c LucretiaMatlabErrMsg_gpu.cu
	$(CCOMP) $(MFLAGS) $(COPTIMFLAGS) -I$(MATPATH)/extern/include LucretiaMatlabErrMsg_gpu.cu


TrackThru_gpu.o : TrackThru_gpu.cu
	$(NVCC) -c  -I$(MATPATH)/extern/include -I$(MATPATH)/simulink/include -DMATLAB_MEX_FILE --relocatable-device-code=true -gencode=arch=compute_20,code=sm_20 -I$(MATPATH)/toolbox/distcomp/gpu/extern/include --compiler-options=-ansi,-D_GNU_SOURCE,-fPIC,-fno-omit-frame-pointer,-pthread  -DLUCRETIA_DPREC -DMX_COMPAT_32 -O $(CFLAGS)  "TrackThru_gpu.cu"


TrackThru_gpu.cu : \
   TrackThru.c
	-cp TrackThru.c TrackThru_gpu.cu
