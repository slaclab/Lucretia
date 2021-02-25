#include "mex.h"
#include "pInterp.hh"
#include <math.h>

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  mwSize ngrid, m, n, nx, ny, nxv, nyv, *dims;
  mwIndex index, *subs;
  const mxArray *cell_element_ptr;
  mxArray *pFi;
  double s_x, s_y, s_z, o_x, o_y, o_z, pO ;
  double *xv, *yv, *F, *xi, *yi, *Fi ;
  ngrid = mxGetM(prhs[3]);
  plhs[0] = mxCreateCellMatrix(ngrid,1);
  nx = mxGetN(prhs[3]);
  ny = mxGetN(prhs[4]);
  xv = mxGetPr(prhs[0]) ;
  if (xv == NULL)
    mexErrMsgIdAndTxt("Lucretia:PInterp", "Incorrect input format");
  yv = mxGetPr(prhs[1]) ;
  if (yv == NULL)
    mexErrMsgIdAndTxt("Lucretia:PInterp", "Incorrect input format");
  F = mxGetPr(prhs[2]) ;
  if (F == NULL)
    mexErrMsgIdAndTxt("Lucretia:PInterp", "Incorrect input format");
  xi = mxGetPr(prhs[3]) ;
  if (xi == NULL)
    mexErrMsgIdAndTxt("Lucretia:PInterp", "Incorrect input format");
  yi = mxGetPr(prhs[4]) ;
  if (yi == NULL)
    mexErrMsgIdAndTxt("Lucretia:PInterp", "Incorrect input format");
  nxv=mxGetNumberOfElements(prhs[0]);
  nyv=mxGetNumberOfElements(prhs[1]);
  s_x = (1.0-nxv)/(xv[0] - xv[nxv-1]);
  s_y = (1.0-nyv)/(yv[0] - yv[nyv-1]);
  s_z = 0;
  o_x = 1.0-xv[0]*s_x;
  o_y = 1.0-yv[0]*s_y;
  o_z = 0;
  double zv=0;
  int ix, iy;
  mwIndex fi_ind ;
  dims= (mwSize*) mxMalloc(sizeof(mwSize)*3); dims[0]=nx; dims[1]=ny; dims[2]=ngrid;
  plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL) ;
  mxFree(dims) ;
  Fi = mxGetPr(plhs[0]) ;
  for (index=0; index<ngrid; index++)  {
    for (ix=0; ix<nx; ix++) {
      for (iy=0; iy<ny; iy++) {
        //mexPrintf("x: %d y: %d index: %d xi: %g yi: %g\n",ix,iy,index,xi[index*ngrid+ix],yi[index*ngrid+iy]);
        interpolate_linear(&pO, F, &xi[index*ngrid+ix], &yi[index*ngrid+iy], &zv, 1, nxv, nyv, 1, 1, s_x, o_x, s_y, o_y, s_z, o_z);
        //interpolate_bicubic(&pO, F, &xi[index*ngrid+ix], &yi[index*ngrid+iy], &zv, 1, nxv, nyv, 1, 1, s_x, o_x, s_y, o_y, s_z, o_z);
        Fi[nx*ny*index+nx*iy+ix]=pO;
      }
    }
  }
}

/* ======== Interpolation Routines ========= */

int access(int M, int N, int O, int x, int y, int z) {
  if (x<0) x=0; else if (x>=N) x=N-1;
  if (y<0) y=0; else if (y>=M) y=M-1;
  if (z<0) z=0; else if (z>=O) z=O-1;
  return x+(y+z*N)*M ;
}

int access_unchecked(int M, int N, int x, int y, int z) {
  return x+(y+z*N)*M ;
}

void indices_linear(int &f000_i,int &f100_i,int &f010_i,int &f110_i,int &f001_i,int &f101_i,int &f011_i,int &f111_i, const int x, const int y, const int z, const mwSize &M, const mwSize &N, const mwSize &O) {
  if (x<=1 || y<=1 || z<=1 || x>=N-2 || y>=M-2 || z>=O-2) {
    f000_i = access(M,N,O, x,   y  , z);
    f100_i = access(M,N,O, x+1, y  , z);
    
    f010_i = access(M,N,O, x,   y+1, z);
    f110_i = access(M,N,O, x+1, y+1, z);
    
    f001_i = access(M,N,O, x,   y  , z+1);
    f101_i = access(M,N,O, x+1, y  , z+1);
    
    f011_i = access(M,N,O, x,   y+1, z+1);
    f111_i = access(M,N,O, x+1, y+1, z+1);
  } else {
    f000_i = access_unchecked(M,N, x,   y  , z);
    f100_i = access_unchecked(M,N, x+1, y  , z);
    
    f010_i = access_unchecked(M,N, x,   y+1, z);
    f110_i = access_unchecked(M,N, x+1, y+1, z);
    
    f001_i = access_unchecked(M,N, x,   y  , z+1);
    f101_i = access_unchecked(M,N, x+1, y  , z+1);
    
    f011_i = access_unchecked(M,N, x,   y+1, z+1);
    f111_i = access_unchecked(M,N, x+1, y+1, z+1);
  }
}

void indices_cubic(
        int f_i[64],
        const int x, const int y, const int z,
        const mwSize &M, const mwSize &N, const mwSize &O) {
  if (x<=2 || y<=2 || z<=2 || x>=N-3 || y>=M-3 || z>=O-3) {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          f_i[i+4*(j+4*k)] = access(M,N,O, x+i-1, y+j-1, z+k-1);
  } else {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          f_i[i+4*(j+4*k)] = access_unchecked(M,N, x+i-1, y+j-1, z+k-1);
  }
}

void interpolate_nearest(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) {
  const mwSize LO = M*N*O;
  for (mwSize i=0; i<ND; ++i) {
    const double &x = pX[i];
    const double &y = pY[i];
    const double &z = pZ[i];
    
    const int x_round = int(round(s_x*x+o_x))-1;
    const int y_round = int(round(s_y*y+o_y))-1;
    const int z_round = int(round(s_z*z+o_z))-1;
    
    const int f00_i = access(M,N,O, x_round,y_round,z_round);
    for (mwSize j=0; j<P; ++j) {
      pO[i + j*ND] = pF[f00_i + j*LO];
    }
  }
}

void interpolate_linear(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) {
  const mwSize LO = M*N*O;
  for (mwSize i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];
    
    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;
    
    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);
    
    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;
    
    const double wx0 = 1.0-dx;
    const double wx1 = dx;
    
    const double wy0 = 1.0-dy;
    const double wy1 = dy;
    
    const double wz0 = 1.0-dz;
    const double wz1 = dz;
    
    int f000_i, f100_i, f010_i, f110_i;
    int f001_i, f101_i, f011_i, f111_i;
    
    indices_linear(
            f000_i, f100_i, f010_i, f110_i,
            f001_i, f101_i, f011_i, f111_i,
            int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);
    
    for (mwSize j=0; j<P; ++j) {
      
      pO[i + j*ND] =
              wz0*(
              wy0*(wx0 * pF[f000_i + j*LO] + wx1 * pF[f100_i + j*LO]) +
              wy1*(wx0 * pF[f010_i + j*LO] + wx1 * pF[f110_i + j*LO])
              )+
              wz1*(
              wy0*(wx0 * pF[f001_i + j*LO] + wx1 * pF[f101_i + j*LO]) +
              wy1*(wx0 * pF[f011_i + j*LO] + wx1 * pF[f111_i + j*LO])
              );
    }
    
  }
}

void interpolate_bicubic(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z)
{
  const mwSize LO = M*N*O;
  for (mwSize i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];
    
    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;
    
    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);
    
    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;
    
    const double dxx = dx*dx;
    const double dxxx = dxx*dx;
    
    const double dyy = dy*dy;
    const double dyyy = dyy*dy;
    
    const double dzz = dz*dz;
    const double dzzz = dzz*dz;
    
    const double wx0 = 0.5 * (    - dx + 2.0*dxx -       dxxx);
    const double wx1 = 0.5 * (2.0      - 5.0*dxx + 3.0 * dxxx);
    const double wx2 = 0.5 * (      dx + 4.0*dxx - 3.0 * dxxx);
    const double wx3 = 0.5 * (         -     dxx +       dxxx);
    
    const double wy0 = 0.5 * (    - dy + 2.0*dyy -       dyyy);
    const double wy1 = 0.5 * (2.0      - 5.0*dyy + 3.0 * dyyy);
    const double wy2 = 0.5 * (      dy + 4.0*dyy - 3.0 * dyyy);
    const double wy3 = 0.5 * (         -     dyy +       dyyy);
    
    const double wz0 = 0.5 * (    - dz + 2.0*dzz -       dzzz);
    const double wz1 = 0.5 * (2.0      - 5.0*dzz + 3.0 * dzzz);
    const double wz2 = 0.5 * (      dz + 4.0*dzz - 3.0 * dzzz);
    const double wz3 = 0.5 * (         -     dzz +       dzzz);
    
    int f_i[64];
    
    indices_cubic(
            f_i,
            int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);
    
    for (mwSize j=0; j<P; ++j) {
      
      pO[i + j*ND] =
              wz0*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*0)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*0)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*0)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*0)] + j*LO])
              ) +
              wz1*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*1)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*1)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*1)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*1)] + j*LO])
              ) +
              wz2*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*2)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*2)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*2)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*2)] + j*LO])
              ) +
              wz3*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*3)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*3)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*3)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*3)] + j*LO])
              );
    }
    
  }
}