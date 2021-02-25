  int access(int M, int N, int O, int x, int y, int z) ;
  int access_unchecked(int M, int N, int x, int y, int z) ;
  void indices_linear(
        int &f000_i,
        int &f100_i,
        int &f010_i,
        int &f110_i,
        int &f001_i,
        int &f101_i,
        int &f011_i,
        int &f111_i,
        const int x, const int y, const int z,
        const mwSize &M, const mwSize &N, const mwSize &O) ;
  void indices_cubic(int f_i[64], const int x, const int y, const int z,
                                      const mwSize &M, const mwSize &N, const mwSize &O) ;
  void interpolate_nearest(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) ;
  void interpolate_linear(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) ;
  void interpolate_bicubic(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) ;