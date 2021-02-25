/* LucretiaPhysics.c
 * This is the home of all physics and math procedures required by Lucretia.
 * In general they either call each other or they are called by procedures
 * in LucretiaCommon.c.
 *
 * Contents:
 *
 * LucretiaPhysicsVersion
 * GetDriftMap
 * GetQuadMap
 * GetPlensMap
 * GetSextMap
 * GetLcavMap
 * GetSolenoidMap
 * PropagateRayThruMult
 * GetBendFringeMap
 * RotateRmat
 * TwissThruRmat
 * CoupledTwissThruRmat
 * LorentzDelay
 * ConvolveSRWFWithBeam
 * BinRays
 * PrepareBunchForLRWFFreq
 * ComputeTLRFreqKicks
 * ComplexProduct
 * GetMADSBendMap
 * GetLucretiaSBendMap
 * CalculateSRPars
 * poidev
 * SRSpectrumAW
 * SRSpectrumHB
 * SynRadC
 * GetCoordMap
 * GetRMSCoord
 * GetMeanCoord
 *
 * AUTH: PT, 03-aug-2004 */
/* MOD:
 * 13-Feb-2012, GW:
 * Add RMS calculation function
 * 24-May-2007, PT:
 * change PropagateRayThruMult to use new argument list when calling
 * CheckP0StopPart.
 * 23-may-2007, PT:
 * bugfix:  SRWFs with z<0 result in an error message, not a crash
 * 08-mar-2006, PT:
 * add propagation of coupled Twiss parameters and solenoid map.
 * 10-feb-2006, PT:
 * add map for coordinate-change element.
 * 09-dec-2005, PT:
 * remove #include of LucretiaDictionary.h.
 * 06-dec-2005, PT:
 * Add Poisson-distributed random number generator, functions
 * in support of synchrotron radiation simulation.
 * 12-sep-2005, PT:
 * add new and improved code to compute the transfer map
 * for a bend in the Lucretia formalism directly, namely
 * GetLucretiaSBendMap.
 */
#ifdef __CUDACC__
#include "gpu/mxGPUArray.h"
#include "curand_kernel.h"
#endif
#include "LucretiaPhysics.h"
#ifndef LUCRETIA_COMMON
#include "LucretiaCommon.h"
#endif
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "LucretiaVersionProto.h"
#include "LucretiaGlobalAccess.h"
#ifndef mex_h
  #define mex_h
  #include <mex.h>
#endif

/* File-scoped variables */

char LucretiaPhysicsVers[] = "LucretiaPhysics Version = 24-May-2007" ;

/*==================================================================*/

/* Return the version string to any calling routine that wants it.
 * RET:   char* LucretiaPhysicsVers, always.
 * ABORT: never.
 * FAIL:  never.                                                    */

char* LucretiaPhysicsVersion( )
{
  return LucretiaPhysicsVers ;
}

/*==================================================================*/

/* Compute the R-matrix of a drift space of given length.
 * RET:   none.
 * ABORT: never.
 * FAIL:  Never.                                            */

void GetDriftMap( double L, Rmat R )
{
  int i,j ;
  for (i = 0 ; i < 6 ; i++)
  {
    for (j = 0 ; j < 6 ; j++)
    {
      if (i==j)
        R[i][j] = 1. ;
      else
        R[i][j] = 0. ;
    }
  }
  R[0][1] = L ;
  R[2][3] = L ;
  return ;
}

/*==================================================================*/

/* Compute the R matrix and the T5xx terms of a quadrupole magnet.
 * RET:   none.
 * ABORT: never.
 * FAIL:  if L or E<=0 will fail on divide-by-zero or sqrt(neg).
 * Calling routine is responsible for preventing such a call.
 */

void GetQuadMap( double L, double B, double Tilt, double E,
        Rmat R, double T5xx[10] )
{
  double Kmod ;                     /* K1 for quad */
  double rootK, LrK ;               /* sqrt(Kmod) and L * sqrt(Kmod) */
  double xcos,xsin,ycos,ysin ;      /* sine and cosine like shortcuts */
  double xsc,ysc ;                  /* more shortcuts */
  double xs2,ys2 ;                  /* square of xsin and ysin */
  double sintilt,costilt ;          /* sine and cosine of the tilt angle */
  double cstilt ;                   /* product of sintilt and costilt */
  double s2tilt, c2tilt ;           /* sin^2 and cos^2 of tilt angle */
  int i ;
  
  /* check for zero strength, if so return a drift matrix */
  
  if (B==0.)
  {
    GetDriftMap( L, R ) ;
    for (i = 0 ; i < 10 ; i++)
      T5xx[i] = 0. ;
    T5xx[4] = 0.5*L ;
    T5xx[9] = 0.5*L ;
    return ;
  }
  
  /* compute the K1 (here Kmod) in 1/m^2 by dividing B (the integrated
   * gradient in m.T/m) by the length, then by the energy in T.m */
  
  Kmod = fabs(B)/L/E/GEV2TM ;
  
  /* if the B is negative, add pi/2 to the tilt angle (this is my clever
   * way of handling the polarity of the quad with a minimum of fuss) */
  
  if (B<0)
    Tilt = Tilt + PI/2 ;
  
  /* compute local variables */
  
  rootK = sqrt(Kmod) ;
  LrK = L * rootK ;
  xcos = cos(LrK) ;
  xsin = sin(LrK)/rootK ;
  xsc = xcos * xsin ;
  xs2 = xsin * xsin ;
  ycos = cosh(LrK) ;
  ysin = sinh(LrK)/rootK ;
  ysc = ycos * ysin ;
  ys2 = ysin * ysin ;
  if (fabs(Tilt) == PI/2)
  {
    costilt = 0 ;
    sintilt = 1 ;
  }
  else
  {
    costilt = cos(Tilt) ;
    sintilt = sin(Tilt) ;
  }
  cstilt = costilt*sintilt ;
  c2tilt = costilt*costilt ;
  s2tilt = sintilt*sintilt ;
  
  /* assign R-matrix terms */
  
  R[0][0] = xcos * c2tilt + ycos*s2tilt ;
  R[0][1] = xsin * c2tilt + ysin*s2tilt ;
  R[0][2] = (xcos-ycos) * cstilt ;
  R[0][3] = (xsin-ysin) * cstilt ;
  R[0][4] = 0. ;
  R[0][5] = 0. ;
  
  R[1][0] = Kmod * (ysin*s2tilt - xsin*c2tilt) ;
  R[1][1] = R[0][0] ;
  R[1][2] = -Kmod * cstilt * ( xsin + ysin ) ;
  R[1][3] = R[0][2] ;
  R[1][4] = 0. ;
  R[1][5] = 0. ;
  
  R[2][0] = R[0][2] ;
  R[2][1] = R[0][3] ;
  R[2][2] = ycos * c2tilt + xcos * s2tilt ;
  R[2][3] = ysin * c2tilt + xsin * s2tilt ;
  R[2][4] = 0. ;
  R[2][5] = 0. ;
  
  R[3][0] = R[1][2] ;
  R[3][1] = R[0][2] ;
  R[3][2] = Kmod * (ysin * c2tilt - xsin * s2tilt) ;
  R[3][3] = ycos * c2tilt + xcos * s2tilt ;
  R[3][4] = 0. ;
  R[3][5] = 0. ;
  
  for (i = 0 ; i<4 ; i++)
  {
    R[4][i] = 0 ;
    R[5][i] = 0 ;
  }
  
  R[4][4] = 1.;
  R[4][5] = 0. ;
  R[5][4] = 0. ;
  R[5][5] = 1. ;
  
  T5xx[0] = 0.25 * Kmod * (c2tilt*(L-xsc) + s2tilt*(ysc-L)) ; // T511
  T5xx[1] =  0.5 * Kmod * (s2tilt*ys2-c2tilt*xs2) ; // T512
  T5xx[2] =  0.5 * Kmod * cstilt * (2*L - xsc - ysc) ; // T513
  T5xx[3] = -0.5 * Kmod * cstilt * (xs2 + ys2) ; // T514
  T5xx[4] = 0.25 * (L + c2tilt*xsc + s2tilt*ysc) ; // T522
  T5xx[5] = T5xx[3] ; // T523
  T5xx[6] = 0.5 * cstilt * (xsc-ysc) ; // T524
  T5xx[7] = 0.25 * Kmod * (s2tilt*(L-xsc) + c2tilt*(ysc-L)) ; // T533
  T5xx[8] = 0.5 * Kmod * (c2tilt*ys2 - s2tilt*xs2) ; // T534
  T5xx[9] = 0.25 * (L + c2tilt*ysc + s2tilt*xsc) ; // T544
  
  return ;
}

/*==================================================================*/

/* Compute the R matrix of a plasma lens.
 * RET:   none.
 * ABORT: never.
 * FAIL:  if L or E<=0 will fail on divide-by-zero or sqrt(neg).
 * Calling routine is responsible for preventing such a call.
 * approximate 1st order transport matrix for
 * plasma lens returned (1/k focusing in each plane)
 */

void GetPlensMap( double L, double B, double E, Rmat R, double T5xx[10] )
{
  double Kmod ;                     /* K1 for quad */
  double rootK, LrK ;               /* sqrt(Kmod) and L * sqrt(Kmod) */
  double xcos,xsin,ycos,ysin ;      /* sine and cosine like shortcuts */
  double xsc,ysc ;                  /* more shortcuts */
  double xs2,ys2 ;                  /* square of xsin and ysin */
  int i ;
  
  /* check for zero strength, if so return a drift matrix */
  GetDriftMap( L, R ) ;
  if (B==0.)
    return ;
  
  /* compute the K1 (here Kmod) in 1/m^2 by dividing B (the integrated
   * gradient in m.T/m) by the length, then by the energy in T.m */
  
  Kmod = fabs(B)/L/E/GEV2TM ;
  
  /* compute local variables */
  
  rootK = sqrt(Kmod) ;
  LrK = L * rootK ;
  xcos = cos(LrK) ;
  xsin = sin(LrK)/rootK ;
  xsc = xcos * xsin ;
  xs2 = xsin * xsin ;
  ycos = cosh(LrK) ;
  ysin = sinh(LrK)/rootK ;
  ysc = ycos * ysin ;
  ys2 = ysin * ysin ;
  
  /* assign R-matrix terms */
  
  R[0][0] = xcos ;
  R[0][1] = xsin ;
  R[1][0] = -Kmod * xsin ;
  R[1][1] = R[0][0] ;
  
  R[2][2] = R[0][0] ;
  R[2][3] = R[0][1] ;
  R[3][2] = R[1][0] ;
  R[3][3] = R[1][1] ;
  
  /*T5xx[0] = 0.25 * Kmod * ( (L-xsc) + (ysc-L) ) ;
  T5xx[1] =  0.5 * Kmod * ( ys2- xs2 ) ;
  T5xx[2] =  0.5 * Kmod * (2*L - xsc - ysc) ;
  T5xx[3] = -0.5 * Kmod * (xs2 + ys2) ;
  T5xx[4] = 0.25 * (L + xsc + ysc) ;
  T5xx[5] = T5xx[3] ;
  T5xx[6] = 0.5 * (xsc-ysc) ;
  T5xx[7] = 0.25 * Kmod * ( (L-xsc) + (ysc-L) ) ;
  T5xx[8] = 0.5 * Kmod * ( ys2 - xs2 ) ;
  T5xx[9] = 0.25 * (L + ysc + xsc ) ;*/
  
  return ;
}

/*==================================================================*/

/* Compute the R matrix and the T5xx terms of a solenoid magnet.
 * RET:   none.
 * ABORT: never.
 * FAIL:  if L or E<=0 will fail on divide-by-zero or sqrt(neg).
 * Calling routine is responsible for preventing such a call.
 */

void GetSolenoidMap( double L, double B, double E,
        Rmat R, double T5xx[10] )
{
  double Ks ; /* normalized strength B/2Brho */
  double C,S, C2, SC ; /* sine and cosine terms of various types */
  double KSC, SCovK, S2ovK, KS2 ; /* more terms */
  double Ks2Lov2, KsL ; /* more terms */
  int i ; /* general purpose counting */
  
  /* zero the T5xx terms except for the ones which are always L/2 */
  
  for (i = 0 ; i < 10 ; i++)
    T5xx[i] = 0. ;
  T5xx[4] = 0.5*L ;
  T5xx[9] = 0.5*L ;
  
  /* check for zero strength, if so return a drift matrix */
  
  if (B==0.)
  {
    GetDriftMap( L, R ) ;
    return ;
  }
  
  /* otherwise calculate temporary variables */
  
  Ks = B/L/E/GEV2TM/2 ; /* standard TRANSPORT definition of Ks */
  KsL = L*Ks ;
  C = cos(KsL) ;
  S = sin(KsL) ;
  C2 = C*C ;
  SC = S*C ;
  KSC = SC*Ks ;
  SCovK = SC/Ks ;
  S2ovK = S*S/Ks ;
  KS2 = S*S*Ks ;
  Ks2Lov2 = Ks*Ks*L/2 ;
  
  /* calculate R-matrix terms */
  
  R[0][0] = C2 ;
  R[0][1] = SCovK ;
  R[0][2] = SC ;
  R[0][3] = S2ovK ;
  R[0][4] = 0 ;
  R[0][5] = 0 ;
  
  R[1][0] = -KSC ;
  R[1][1] = C2 ;
  R[1][2] = -KS2 ;
  R[1][3] = SC ;
  R[1][4] = 0 ;
  R[1][5] = 0 ;
  
  R[2][0] = -SC ;
  R[2][1] = -S2ovK ;
  R[2][2] = C2 ;
  R[2][3] = SCovK ;
  R[2][4] = 0 ;
  R[2][5] = 0 ;
  
  R[3][0] = KS2 ;
  R[3][1] = -SC ;
  R[3][2] = -KSC ;
  R[3][3] = C2 ;
  R[3][4] = 0 ;
  R[3][5] = 0 ;
  
  R[4][0] = 0 ;
  R[4][1] = 0 ;
  R[4][2] = 0 ;
  R[4][3] = 0 ;
  R[4][4] = 1 ;
  R[4][5] = 0 ;
  
  R[5][0] = 0 ;
  R[5][1] = 0 ;
  R[5][2] = 0 ;
  R[5][3] = 0 ;
  R[5][4] = 0 ;
  R[5][5] = 1 ;
  
  /* now for the T matrix terms */
  
  T5xx[0] = Ks2Lov2 ;
  T5xx[3] = -KsL ;
  T5xx[5] =  KsL ;
  T5xx[7] = Ks2Lov2 ;
  
  return ;
  
}

/*==================================================================*/

/* get the second order map components for a sextupole.  Based on
 * the expressions on page 160 of D.C. Carey's 1987 text on beam
 * optics. */

/* Ret:  None. */
/* ABORT: never.
 * FAIL:  if P<=0 will fail on divide-by-zero or sqrt(neg).
 * Calling routine is responsible for preventing such a call.
 */

void GetSextMap( double L, double B, double Tilt, double P,
        double TijkTransv[4][10] )
{
  double c,s,cs,c2_minus_s2 ;
  double k2t2, k2t3, k2t4   ;
  double k2t                ;
  
  /* Carey's notation uses
   *
   * k^2 = B_pole/a^2/brho;
   *
   * the Lucretia B parameter is integrated second deriviative,
   *
   * B = L B'' = L * 2 B_pole/a^2
   *
   * start by computing the product of Carey's k^2 * t, where he uses
   * t to mean L: */
  
  k2t  = B/2/P/GEV2TM ;
  k2t2 = k2t  * L ;
  k2t3 = k2t2 * L ;
  k2t4 = k2t3 * L ;
  
  /* compute some trig functions */
  
  c = cos(Tilt) ;
  s = sin(Tilt) ;
  cs = c*s ;
  c2_minus_s2 = c*c - s*s ;
  
  /* without further ado, compute the position map; terms are in order
   * x^2, xpx, xy, xpy, px^2, pxy, pxpy, y^2, ypy, py^2 in the arrays,
   * but they will be populated in a different order below (x-only, then
   * y-only, then mixed terms) for convenience */
  
  TijkTransv[0][0] = k2t2 * ( -c2_minus_s2 * c + 2*cs * s ) / 2  ;
  TijkTransv[0][1] = k2t3 * ( -c2_minus_s2 * c + 2*cs * s ) / 3  ;
  TijkTransv[0][4] = k2t4 * ( -c2_minus_s2 * c + 2*cs * s ) / 12 ;
  TijkTransv[0][7] = -TijkTransv[0][0] ;
  TijkTransv[0][8] = -TijkTransv[0][1] ;
  TijkTransv[0][9] = -TijkTransv[0][4] ;
  TijkTransv[0][2] = k2t2 * (-2*cs*c - c2_minus_s2 * s)     ;
  TijkTransv[0][3] = k2t3 * (-2*cs*c - c2_minus_s2 * s) / 3 ;
  TijkTransv[0][5] = TijkTransv[0][3]                            ;
  TijkTransv[0][6] = k2t4 * (-2*cs*c - c2_minus_s2 * s) / 6 ;
  
  TijkTransv[2][0] = k2t2 * ( -c2_minus_s2 * s - 2*cs * c ) / 2  ;
  TijkTransv[2][1] = k2t3 * ( -c2_minus_s2 * s - 2*cs * c ) / 3  ;
  TijkTransv[2][4] = k2t4 * ( -c2_minus_s2 * s - 2*cs * c ) / 12 ;
  TijkTransv[2][7] = -TijkTransv[2][0] ;
  TijkTransv[2][8] = -TijkTransv[2][1] ;
  TijkTransv[2][9] = -TijkTransv[2][4] ;
  TijkTransv[2][2] = k2t2 * (-2*cs*s + c2_minus_s2 * c)     ;
  TijkTransv[2][3] = k2t3 * (-2*cs*s + c2_minus_s2 * c) / 3 ;
  TijkTransv[2][5] = TijkTransv[2][3]                            ;
  TijkTransv[2][6] = k2t4 * (-2*cs*s + c2_minus_s2 * c) / 6 ;
  
  TijkTransv[1][0] = k2t *  ( -c2_minus_s2 * c + 2*cs * s )     ;
  TijkTransv[1][1] = k2t2 * ( -c2_minus_s2 * c + 2*cs * s )     ;
  TijkTransv[1][4] = k2t3 * ( -c2_minus_s2 * c + 2*cs * s ) / 3 ;
  TijkTransv[1][7] = -TijkTransv[1][0] ;
  TijkTransv[1][8] = -TijkTransv[1][1] ;
  TijkTransv[1][9] = -TijkTransv[1][4] ;
  TijkTransv[1][2] = k2t * 2 * (-2*cs*c - c2_minus_s2 * s)      ;
  TijkTransv[1][3] = k2t2    * (-2*cs*c - c2_minus_s2 * s)      ;
  TijkTransv[1][5] = TijkTransv[1][3]                                ;
  TijkTransv[1][6] = k2t3 * 2 * (-2*cs*c - c2_minus_s2 * s) / 3 ;
  
  TijkTransv[3][0] = k2t  * ( -c2_minus_s2 * s - 2*cs * c )     ;
  TijkTransv[3][1] = k2t2 * ( -c2_minus_s2 * s - 2*cs * c )     ;
  TijkTransv[3][4] = k2t3 * ( -c2_minus_s2 * s - 2*cs * c ) / 3 ;
  TijkTransv[3][7] = -TijkTransv[3][0] ;
  TijkTransv[3][8] = -TijkTransv[3][1] ;
  TijkTransv[3][9] = -TijkTransv[3][4] ;
  TijkTransv[3][2] = k2t * 2 * (-2*cs*s + c2_minus_s2 * c)      ;
  TijkTransv[3][3] = k2t2    * (-2*cs*s + c2_minus_s2 * c)      ;
  TijkTransv[3][5] = TijkTransv[3][3]                                ;
  TijkTransv[3][6] = k2t3 * 2 * (-2*cs*s + c2_minus_s2 * c) / 3 ;
  
  return ;
}

/*==================================================================*/

/* Get the R matrix for a bending magnet's fringe field.  This
 * function is translated/stolen from DIMAD's ENTEX. */

/* RET:    none.
 * ABORT:  never.
 * FAIL:   Will fail on div-zero or range error if Eangle,
 * L, or P are zero. */

void GetBendFringeMap( double L, double intB, double intG,
        double P, double Eangle, double H,
        double Hgap, double Fint, double S,
        Rmat R, double T5xx[10] )
{
  double K0, K1 ;
  double extan, exsin, exsec ;
  double cfocl, entc ;
  double extan2, exsec2 ;
  double K0K0, K0ov2 ;
  
  /* compute the K0 and K1 parameters */
  
  K0 = intB / L / P / GEV2TM ;
  K1 = intG / L / P / GEV2TM ;
  
  /* compute the tangent, sine, and secant of the pole-face angle */
  
  extan = tan( Eangle ) ;
  exsin = sin( Eangle ) ;
  exsec = 1/cos( Eangle ) ;
  
  /* compute fringe-focusing term */
  
  cfocl = 2*Fint*K0*Hgap*exsec*(1+exsin*exsin) ;
  
  /* put in the linear optics terms */
  
  R[0][0] = 1. ;
  R[0][1] = 0. ;
  R[0][2] = 0. ;
  R[0][3] = 0. ;
  R[0][4] = 0. ;
  R[0][5] = 0. ;
  
  R[1][0] = K0 * extan ;
  R[1][1] = 1. ;
  R[1][2] = 0. ;
  R[1][3] = 0. ;
  R[1][4] = 0. ;
  R[1][5] = 0. ;
  
  R[2][0] = 0. ;
  R[2][1] = 0. ;
  R[2][2] = 1. ;
  R[2][3] = 0. ;
  R[2][4] = 0. ;
  R[2][5] = 0. ;
  
  R[3][0] = 0. ;
  R[3][1] = 0. ;
  R[3][2] = -K0 * tan( Eangle - cfocl ) ;
  R[3][3] = 1. ;
  R[3][4] = 0. ;
  R[3][5] = 0. ;
  
  R[4][0] = 0. ;
  R[4][1] = 0. ;
  R[4][2] = 0. ;
  R[4][3] = 0. ;
  R[4][4] = 1. ;
  R[4][5] = 0. ;
  
  R[5][0] = 0. ;
  R[5][1] = 0. ;
  R[5][2] = 0. ;
  R[5][3] = 0. ;
  R[5][4] = 0. ;
  R[5][5] = 1. ;
  
  /* now for the T-matrix terms: */
  
  extan2 = extan * extan ;
  exsec2 = exsec * exsec ;
  K0K0 = K0 * K0 ;
  K0ov2 = 0.5 *  K0 ;
  entc = 0.5 * K0 * H * exsec2 * exsec ;
  
  T5xx[0] = -S*K0ov2 * extan2 ;      /* T111 */
  T5xx[1] =  S*K0ov2 * exsec2 ;      /* T133 */
  T5xx[2] = entc + (K1+K0K0*(S-1)*extan2/4)*extan ; /* T211 */
  T5xx[3] = S * K0 * extan2 ;          /* T212 */
  T5xx[4] = (-K1+K0K0*((S+1)*(extan2+1)/4 + S*extan2/2))*extan-entc ; /* T233 */
  T5xx[5] = -S * K0 * extan2 ;         /* T234 */
  T5xx[6] = S * extan2*K0 ;            /* T313 */
  T5xx[7] = -2*entc+(-2.*K1-K0K0*(S-1)*exsec2/2)*extan; /* T413 */
  T5xx[8] = -S*K0*extan2 ;           /* T414 */
  T5xx[9] = -S*K0*exsec2 ;           /* T423 */
  
  /* and that's it. */
  
  return ;
  
}

/*==================================================================*/

/* Get the R matrix for an RF accelerating structure. */

/* RET:   none.
 * ABORT: never.
 * FAIL:  function will fail on floating-point error  if the length
 * is zero (infinite focusing from end-fields), if
 * P0 + Egain < 0 (excessive deceleration), or if P0 == 0.
 * The calling routine
 * is responsible for catching such errors. */

void GetLcavMap( double L,   double P0, double freq,
        double Egain, double Psin, Rmat R, int flag )
{
  double k1, k2 ;
  double R12, R22 ;
  double rel_egain ;
  double Pf ;
  
  /* start with the horizontal transverse coordinates */
  
  rel_egain = Egain/P0 ;
  Pf = P0 + Egain ;
  if (fabs(rel_egain) < MIN_EGAIN)
    GetDriftMap( L, R ) ;
  else
  {
    R12 = P0 * L * log( 1.+Egain/P0 ) / Egain ;
    R22 = P0 / (P0 + Egain) ;
    k1 = -Egain / L / P0 / 2. ;
    k2 =  -k1 * R22 ;
    
    R[0][0] = 1. + k1 * R12 ;
    R[0][1] = R12 ;
    R[1][0] = k2 + (k2*R12+R22)*k1 ;
    R[1][1] = R22 + k2*R12 ;
  }
  
  /* if the user only wants one set of transverse coordinates, return */
  
  if (flag == 1)
    return ;
  
  /* otherwise fill in the zeros outside of the block-diagonal and the
   * other transverse coordinates */
  
  R[0][2] = 0. ;
  R[0][3] = 0. ;
  R[1][2] = 0. ;
  R[1][3] = 0. ;
  
  R[2][0] = 0. ;
  R[2][1] = 0. ;
  R[3][0] = 0. ;
  R[3][1] = 0. ;
  
  R[2][2] = R[0][0] ;
  R[2][3] = R[0][1] ;
  R[3][2] = R[1][0] ;
  R[3][3] = R[1][1] ;
  
  /* if the user only wanted the 2 sets of transverse coords return */
  
  if (flag==2)
    return;
  
  /* otherwise, fill in the x-z and y-z zeros and the longitudinal
   * coordinates */
  
  R[0][4] = 0. ;
  R[0][5] = 0. ;
  R[1][4] = 0. ;
  R[1][5] = 0. ;
  
  R[2][4] = 0. ;
  R[2][5] = 0. ;
  R[3][4] = 0. ;
  R[3][5] = 0. ;
  
  R[4][0] = 0. ;
  R[4][1] = 0. ;
  R[4][2] = 0. ;
  R[4][3] = 0. ;
  
  R[5][0] = 0. ;
  R[5][1] = 0. ;
  R[5][2] = 0. ;
  R[5][3] = 0. ;
  
  R[4][4] = 1. ;
  R[4][5] = 0. ;
  R[5][5] = P0 / Pf ;
  R[5][4] = Psin/Pf * 2 * PI / CLIGHT * (freq*1e6) ;
  
  return ;
}

/*==================================================================*/

/* Propagate a ray through a thin-lens multipole of arbitrary order.
 * Transverse propagation only. */

/* RET:   none.
 * ABORT: never.
 * FAIL:  Will fail if the values in PIvec are not positive integers.
 * It is the calling routine's responsibility to ensure that
 * PIvec is so defined. */
#ifdef __CUDACC__
__device__ void PropagateRayThruMult_gpu( double L, double* Bvec, double* Tvec,
        double* PIvec, int nTerms, double* Angle,
        double dB, double Tilt,
        double* xi, double* xf, int zmotion,
        int SRFlag, double Lrad, double* bstop, int* bngoodray,double* bx,double* by,
        int elemno, int ray, double splitScale, double* PascalMatrix, double* Bang, double* MaxMultInd, curandState_t *rState )
{
  
  int count,c2 ;
  int MaxPM, PInd ;
  double P0, xcent, ycent ;
  double BxL = 0 ;
  double ByL = 0 ;
  double BxnL, BynL ;
  double BL,xsign,ysign ;
  double TiltAll ;
  double SR_dP = 0 ;
  int Stop, *stp=NULL ;
  
  /* get pointers to the Pascal matrix and the factorial vector,
   * as well as their current size.  The calling routine needs to
   * set this up to ensure that they are the right size! */
  MaxPM = (int)*MaxMultInd ;
  
  /* propagate the rays to the center of the element */
  
  *(xf+0) = *(xi+0) + L/2 * *(xi+1) ;
  *(xf+1) = *(xi+1) ;
  *(xf+2) = *(xi+2) + L/2 * *(xi+3) ;
  *(xf+3) = *(xi+3) ;
  P0 = *(xi+5) ;
  
  xcent = *xf ;
  ycent = *(xf+2) ;
  
  /* loop over multipole moments and over terms in each moment, and
   * build the BxL and ByL fields */
  for (count=0 ; count<nTerms ; count++)
  {
    PInd = abs((int)(PIvec[count])) ;
    xsign = 1 ; ysign = 1 ;
    BxnL = 0 ; BynL = 0 ;
    for (c2=0 ; c2<=PInd ; c2++)
    {
      BL = PascalMatrix[PInd+c2*MaxPM] * pow(xcent,PInd-c2)
      * pow(ycent,c2) ;
      if (c2 % 2 == 0) /* By component */
      {
        BynL += ysign * BL ;
        ysign *= -1 ;
      }
      else             /* Bx component */
      {
        BxnL += xsign * BL ;
        xsign *= -1 ;
      }
    }
    
    /* apply the field rotation and sum the resulting field into the
     * total field at this point */
    
    TiltAll = Tilt + Tvec[count] ;
    TiltAll *= (double)(PInd+1) ;
    BxL += (BxnL*cos(TiltAll) - BynL*sin(TiltAll))
    * (splitScale*Bvec[count]) / Bang[PInd] ;
    ByL += (BynL*cos(TiltAll) + BxnL*sin(TiltAll))
    * (splitScale*Bvec[count]) / Bang[PInd] ;
    
  }
  
  /* If SR is required, calculate it and apply 1/2 of it here, 1/2 at end of elt */
  
  if (SRFlag > SR_None && Lrad>0)
  {
    SR_dP = ComputeSRMomentumLoss_gpu( P0,sqrt(BxL*BxL+ByL*ByL), Lrad, SRFlag, rState ) ;
    *(xf+5) = P0 - SR_dP ;
    Stop = CheckP0StopPart( bstop,bngoodray,bx,by,elemno,ray,P0-SR_dP, UPSTREAM, stp ) ;
    if (Stop == 1)
      goto egress ;
    P0 -= SR_dP / 2;
  }
  
  /* Apply the kick, keeping in mind that the K0L terms apply a bend
   * to the reference trajectory which must be taken out */
  
  *(xf+1) += -ByL/P0/GEV2TM * (splitScale*dB) - Angle[0] ;
  *(xf+3) +=  BxL/P0/GEV2TM * (splitScale*dB) - Angle[1] ;
  
  /* propagate the new trajectory through the second half-length drift */
  
  *(xf  ) += *(xf+1) * L/2 ;
  *(xf+2) += *(xf+3) * L/2 ;
  
  /* if zmotion is requested, perform it now (assumes that the calling
   * routine has already set y[raystart+4] = x[raystart+4] */
  
  if (zmotion == 1)
  {
    *(xf+4) += L/4 * ( *(xi+1) * *(xi+1) +
            *(xi+3) * *(xi+3) +
            *(xf+1) * *(xf+1) +
            *(xf+3) * *(xf+3)   ) ;
  }
  
  egress:
    
    return ;
}
#endif
void PropagateRayThruMult( double L, double* Bvec, double* Tvec,
        double* PIvec, int nTerms, double* Angle,
        double dB, double Tilt,
        double* xi, double* xf, int zmotion,
        int SRFlag, double Lrad, double* bstop, int* bngoodray,double* bx,double* by,
        int elemno, int ray, double splitScale )
{
  
  int count,c2 ;
  double* PascalMatrix ;
  double* Bang ;
  int MaxPM, PInd ;
  double P0, xcent, ycent ;
  double BxL = 0 ;
  double ByL = 0 ;
  double BxnL, BynL ;
  double BL,xsign,ysign ;
  double TiltAll ;
  double SR_dP = 0 ;
  int Stop, *stp=NULL ;
  
  /* get pointers to the Pascal matrix and the factorial vector,
   * as well as their current size.  The calling routine needs to
   * set this up to ensure that they are the right size! */
  
  PascalMatrix = GetPascalMatrix( ) ;
  Bang         = GetFactorial( ) ;
  MaxPM = (int)(GetMaxMultipoleIndex( )) ;
  
  /* propagate the rays to the center of the element */
  
  *(xf+0) = *(xi+0) + L/2 * *(xi+1) ;
  *(xf+1) = *(xi+1) ;
  *(xf+2) = *(xi+2) + L/2 * *(xi+3) ;
  *(xf+3) = *(xi+3) ;
  P0 = *(xi+5) ;
  
  xcent = *xf ;
  ycent = *(xf+2) ;
  
  /* loop over multipole moments and over terms in each moment, and
   * build the BxL and ByL fields */
  
  for (count=0 ; count<nTerms ; count++)
  {
    PInd = abs((int)(PIvec[count])) ;
    xsign = 1 ; ysign = 1 ;
    BxnL = 0 ; BynL = 0 ;
    for (c2=0 ; c2<=PInd ; c2++)
    {
      BL = PascalMatrix[PInd+c2*MaxPM] * pow(xcent,PInd-c2)
      * pow(ycent,c2) ;
      if (c2 % 2 == 0) /* By component */
      {
        BynL += ysign * BL ;
        ysign *= -1 ;
      }
      else             /* Bx component */
      {
        BxnL += xsign * BL ;
        xsign *= -1 ;
      }
    }
    
    /* apply the field rotation and sum the resulting field into the
     * total field at this point */
    
    TiltAll = Tilt + Tvec[count] ;
    TiltAll *= (double)(PInd+1) ;
    BxL += (BxnL*cos(TiltAll) - BynL*sin(TiltAll))
    * (splitScale*Bvec[count]) / Bang[PInd] ;
    ByL += (BynL*cos(TiltAll) + BxnL*sin(TiltAll))
    * (splitScale*Bvec[count]) / Bang[PInd] ;
    
  }
  
  /* If SR is required, calculate it and apply 1/2 of it here, 1/2 at end of elt */
  
  if (SRFlag > SR_None && Lrad>0)
  {
    SR_dP = ComputeSRMomentumLoss( P0,sqrt(BxL*BxL+ByL*ByL), Lrad, SRFlag ) ;
    *(xf+5) = P0 - SR_dP ;
    Stop = CheckP0StopPart( bstop,bngoodray,bx,by,elemno,ray,P0-SR_dP, UPSTREAM, stp ) ;
    if (Stop == 1)
      goto egress ;
    P0 -= SR_dP / 2;
  }
  
  /* Apply the kick, keeping in mind that the K0L terms apply a bend
   * to the reference trajectory which must be taken out */
  
  *(xf+1) += -ByL/P0/GEV2TM * (splitScale*dB) - Angle[0] ;
  *(xf+3) +=  BxL/P0/GEV2TM * (splitScale*dB) - Angle[1] ;
  
  /* propagate the new trajectory through the second half-length drift */
  
  *(xf  ) += *(xf+1) * L/2 ;
  *(xf+2) += *(xf+3) * L/2 ;
  
  /* if zmotion is requested, perform it now (assumes that the calling
   * routine has already set y[raystart+4] = x[raystart+4] */
  
  if (zmotion == 1)
  {
    *(xf+4) += L/4 * ( *(xi+1) * *(xi+1) +
            *(xi+3) * *(xi+3) +
            *(xf+1) * *(xf+1) +
            *(xf+3) * *(xf+3)   ) ;
  }
  
  egress:
    
    return ;
}

/*==================================================================*/

/* rotate an R-matrix through an xy angle.  The rotated matrix is
 * returned, the original matrix destroyed.  The rotation is designed
 * to agree with the MAD rotation convention. To be utterly specific,
 * what is returned is
 *
 * Rnew = Theta * R * inv(Theta)
 *
 * where Theta = [cos(theta) 0 -sin(theta) 0 ; etc.].
 *
 * RET:   none.
 * ABORT: never.
 * FAIL:  never.                                            */

void RotateRmat( Rmat R, double Tilt )
{
  Rmat theta = {
    {0,0,0,0,0,0},
    {0,0,0,0,0,0},
    {0,0,0,0,0,0},
    {0,0,0,0,0,0},
    {0,0,0,0,1,0},
    {0,0,0,0,0,1}
  };
  double sintilt,costilt ;
  
  /* pre-calculate the sine and cosine of the tilt angle */
  
  sintilt = sin(Tilt) ;
  costilt = cos(Tilt) ;
  
  /* make matrix theta equal to inv(theta), above */
  
  theta[0][0] =  costilt ;
  theta[0][2] = sintilt ;
  theta[1][1] =  costilt ;
  theta[1][3] = sintilt ;
  
  theta[2][0] = -sintilt ;
  theta[2][2] = costilt ;
  theta[3][1] = -sintilt ;
  theta[3][3] = costilt ;
  
  /* matrix multiply */
  
  RmatProduct( R, theta, R ) ;
  
  /* flip the sign of the sines */
  
  theta[0][2] = -sintilt ;
  theta[1][3] = -sintilt ;
  theta[2][0] = sintilt ;
  theta[3][1] = sintilt ;
  
  /* another matrix multiply */
  
  RmatProduct( theta, R, R ) ;
  
  return;
  
}

/*==================================================================*/

/* Propagate a set of x and y Twiss parameters through an R-matrix.
 *
 * RET:   1 if both x and y calculations are successful, 0 if
 * either or both failed.  Failure can occur if betx or bety
 * == 0, if either the x or y submatrix has zero determinant,
 * or if the phase advance calculation is ill-conditioned.
 * ABORT: never.
 * FAIL:  never.                                             */

int TwissThruRmat( Rmat R, struct beta0* init, struct beta0* final)
{
  double gamtwis, det, dmu, TanDmuDenom ;
  
  /* unpack horizontal parameters */
  
  double betai  = init->betx ;
  double alfai  = init->alfx ;
  double etaxi  = init->etax ;
  double etapxi = init->etapx ;
  double etayi  = init->etay ;
  double etapyi = init->etapy ;
  double nui    = init->nux ;
  
  /* get pointers to final locations of parameters */
  
  double *betaf = &(final->betx) ;
  double *alfaf = &(final->alfx) ;
  double *etaf  = &(final->etax) ;
  double *etapf = &(final->etapx) ;
  double *nuf   = &(final->nux) ;
  
  /* initialize position and angle indices, and plane counter */
  
  int pos = 0;
  int ang = 1 ;
  int plane = 0 ;
  
  /* here we enter a loop which executes twice, once per plane */
  
  while (1)
  {
    
    /* check for bad betatron function or determinant */
    
    if (betai == 0)
      goto egress ;
    gamtwis = (1+alfai*alfai)/betai ;
    det = R[pos][pos] * R[ang][ang] - R[ang][pos] * R[pos][ang] ;
    if (det == 0)
      goto egress ;
    TanDmuDenom = R[pos][pos]*betai - R[pos][ang]*alfai;
    if (TanDmuDenom == 0)
      goto egress ;
    
    /* having dealt with all possible disasters, we can now do the math */
    
    *betaf = (          R[pos][pos]*R[pos][pos]  * betai
            -      2*R[pos][pos]*R[pos][ang]  * alfai
            +        R[pos][ang]*R[pos][ang]  * gamtwis ) / det ;
    
    *alfaf = ( -        R[pos][pos]*R[ang][pos]  * betai
            + (det+2*R[pos][ang]*R[ang][pos]) * alfai
            -        R[pos][ang]*R[ang][ang]  * gamtwis ) / det ;
    
    dmu = atan( R[pos][ang]/TanDmuDenom ) ;
    if (dmu < 0.) dmu += PI ;
    if (R[pos][ang] < 0.) dmu -= PI ;
    *nuf = nui + dmu / 2. / PI ;
    
    *etaf  = R[pos][0] * etaxi
            + R[pos][1] * etapxi
            + R[pos][2] * etayi
            + R[pos][3] * etapyi
            + R[pos][5]          ;
    
    *etapf = R[ang][0] * etaxi
            + R[ang][1] * etapxi
            + R[ang][2] * etayi
            + R[ang][3] * etapyi
            + R[ang][5]          ;
    
    /* increment plane index */
    
    plane++ ;
    
    /* exit if we've already been thru twice */
    
    if (plane > 1)
      break ;
    
    /* otherwise set up for the loop which does the y calculation */
    
    pos = 2 ;
    ang = 3 ;
    betai  = init->bety ;
    alfai  = init->alfy ;
    nui    = init->nuy ;
    betaf = &(final->bety) ;
    alfaf = &(final->alfy) ;
    etaf  = &(final->etay) ;
    etapf = &(final->etapy) ;
    nuf   = &(final->nuy) ;
    
  } /* end of calculation loop */
  
  /* here is the egress:  if two full loops were not completed, then return
   * bad status, otherwise return good status */
  
  egress:
    
    if (plane < 2)
      return 0;
    else
      return 1 ;
    
}

/*==================================================================*/

/* propagate coupled Twiss parameters through an R-matrix */

/* RET:    1 if calculation successful, 0 if failure.
 * ABORT:  never
 * FAIL:
 *
 */

int CoupledTwissThruRmat( Rmat R, double* WolskiIn, double* WolskiOut,
        int TransposeFlag )
        
{
  
  int CTTRStatus = 0 ; /* failure implicit at this stage */
  int i,j,k,l ;
  int w,u,v ;
  
  /* since we do linear acceleration, we need to get a normalizing
   * factor from the determinant of R; do that now */
  
  double dval = GetMatrixNormalizer( (double*)R ) ;
  if ( dval <= 0 )
    goto egress ;
  
  /* otherwise, it's just WolskiOut = R * WolskiIn * R^T; note, however,
   * that R is in the appropriate C convention, ie R[0][1] = R_12, whereas
   * WolskiIn may be transposed thanks to Matlab, ie W[1][0] = beta_12.
   * Thankfully, there's a transpose flag to tell us what to do here: */
  
  for (i=0 ; i<36 ; i++)
    WolskiOut[i] = 0 ;
  
  for (i=0 ; i<6 ; i++)
    for (j=0 ; j<=i ; j++)
    {
      for (k=0 ; k<6 ; k++)
        for (l=0 ; l<6 ; l++)
        {
          if (TransposeFlag == 1)
          {
            w = 6*j+i ;
            u = 6*l+k ;
            v = 6*i+j ;
          }
          else
          {
            w = 6*i+j ;
            u = 6*k+l ;
            v = 6*j+i ;
          }
          WolskiOut[w] += R[i][k] * R[j][l] * WolskiIn[u] / dval ;
        }
      
      WolskiOut[v] = WolskiOut[w] ;
    }
  
  /* if we got this far then we succeeded */
  
  CTTRStatus = 1 ;
  
  egress:
    
    return CTTRStatus ;
    
}

/*==================================================================*/

/* Convolve a bunch with the short-range wakefield */

/* RET:    integer status (1=good, 0 = bad)
 * ABORT:  never.
 * Fail:           */

int ConvolveSRWFWithBeam( struct Bunch* ThisBunch, int WFno, int flag )
{
  int stat = 1 ;
  double* z ;
  double* k ;
  double  BinWidth ;
  struct SRWF* TheWF ;
  int  i,j ;
  static double* TrailingKvals = NULL ;
  int nwakevals ;
  
  /* get access to the desired WF data in the bunch structure */
  
  if (flag==0)
    TheWF = ThisBunch->ZSR[WFno] ;
  else
    TheWF = ThisBunch->TSR[WFno] ;
  
  /* if the WF in question has already been convolved, we don't need to do
   * anything.  If the wakefield was previously convolved but something has
   * happened (compression, collimation, etc) to make the previous convolution
   * invalid, it is the responsibility of the invalidating routine to blow away
   * the convolved-WF data. */
  
  if (TheWF != NULL)
    goto egress ;
  
  TheWF = (struct SRWF*)calloc(1,sizeof(struct SRWF) ) ;
  if (TheWF == NULL)
  {
    stat = 0 ;
    BadSRWFAllocMsg( WFno+1, flag ) ;
    goto egress ;
  }
  
  /* get access to the appropriate wakefield information: */
  
  nwakevals = GetSRWFParameters( WFno, flag, &z, &k, &BinWidth ) ;
  if ( (nwakevals <= 0) || (BinWidth == 0.) )
  {
    stat = 0 ;
    BadSRWFMessage( WFno+1, flag ) ;
    goto egress ;
  }
  
  /* SRWFs must have z >= 0; if this is not the case, exit with bad status */
  
  for (i=0 ; i<nwakevals ; i++)
    if (z[i] < 0)
    {
      stat = 0 ;
      BadSRWFMessage( WFno+1, flag ) ;
      goto egress ;
    }
  
  /* bin the rays according to the desired binwidth */
  
  stat = BinRays( ThisBunch, TheWF, BinWidth ) ;
  if (stat==0)
    BadSRWFAllocMsg( WFno+1, flag ) ;
  if (stat != 1)
    goto egress ;
  
  /* since BinRays does not allocate the Vx, Vy, or K buffers,
   * take care of them now */
  
  TheWF->binVx = (double*)calloc(TheWF->nbin,sizeof(double)) ;
  TheWF->binVy = (double*)calloc(TheWF->nbin,sizeof(double)) ;
  if (flag==0)
    TheWF->K = (double*)calloc(TheWF->nbin,sizeof(double)) ;
  else
    TheWF->K = (double*)calloc(TheWF->nbin*TheWF->nbin,sizeof(double)) ;
  
  if ( (TheWF->K == NULL)     ||
          (TheWF->binVx == NULL) ||
          (TheWF->binVy == NULL)    )
  {
    stat = 0 ;
    BadSRWFAllocMsg( WFno+1, flag ) ;
    goto egress ;
  }
  
  /* if only 1 bin, take care of its value now */
  
  if (TheWF->nbin==1)
  {
    TheWF->K[0] = 0.5 * k[0] * TheWF->binQ[0] / 1e9 ;
    goto egress ;
  }
  
  /* convolve the wakefield  */
  /* Invert sign of K if source and target bin charge types different */
  
  for (i=0 ; i<TheWF->nbin ; i++)
  {
    if (flag==0)
      TheWF->K[i] += 0.5 * k[0] * TheWF->binQ[i] ;
    if (i==TheWF->nbin-1)
      break ;
    
    TrailingKvals = SplineSRWF(z,k,TheWF->binx,nwakevals,i,TheWF->nbin) ;
    
    for (j=i+1 ; j<TheWF->nbin ; j++)
    {
      if (flag==0)
        if (TheWF->ptype[i]==TheWF->ptype[j])
          TheWF->K[j] += TheWF->binQ[i] * TrailingKvals[j-i-1] ;
        else
          TheWF->K[j] -= TheWF->binQ[i] * TrailingKvals[j-i-1] ;
      else
        if (TheWF->ptype[i]==TheWF->ptype[j])
          TheWF->K[TheWF->nbin*i+j] +=  TrailingKvals[j-i-1] ;
        else
          TheWF->K[TheWF->nbin*i+j] -=  TrailingKvals[j-i-1] ;
    }
    
    TheWF->binx[i] = 0. ;
    
  }
  
  /* Since the WF data structure holds wakefields in V/C/m and V/C/m^2,
   * the K values thus computed are in V.  Beam energies are in GeV, so
   * save some floating point work later by converting all K values now
   * to GV/C/m and GV/C/m^2 */
  
  j = TheWF->nbin ;
  if (flag==1)
    j *= j ;
  for (i=0 ; i<j ; i++)
    TheWF->K[i] /= 1e9 ;
  
  /* if we reached this point without a status failure, attach the new wakefield to
   * the bunch structure, set status, exit. */
  
  egress:
    
    if (stat!=0)
    {
      if (flag==0)
        ThisBunch->ZSR[WFno] = TheWF ;
      else
        ThisBunch->TSR[WFno] = TheWF ;
    }
    return stat ;
}

/*==================================================================*/

/* Bin a bunch according to the BinWidth parameter of a wakefield,
 * and set the WF's pointers to point at the binned data. */

/* RET:    integer status (1=good, 0 = bad, -1 = no valid rays).
 * As a side effect,
 * set the data pointers of a SRWF structure (specifically
 * binno, binQ, binx, biny), and put the z positions of
 * the bin centers in binx).
 * ABORT:  never.
 * Fail:           */

int BinRays( struct Bunch* ThisBunch, struct SRWF* TheWF,
        double BinWidth )
{
  int stat = 1 ;
  int nrayvalid=0 ;
  int	count ;
  double Qtot, Q0p5, Q0p34, Qplus, Qminus ;
  double* SortkeyZ ;
  int icent=0, iplus=0, iminus=0, i,j ;
  double binstart, binstop, sigz ;
  double zpos = 0 ;
  int nbinsmax ;
  
  /* allocate a bin number for each ray */
  
  TheWF->binno = (int*)calloc(ThisBunch->nray,sizeof(int)) ;
  if (TheWF->binno == NULL)
  {
    stat = 0 ;
    goto egress ;
  }
  
  /* count the number of still-valid rays */
  
  Qtot = 0 ;
  for (count = 0 ; count< ThisBunch->nray ; count++)
  {
    if (ThisBunch->stop[count] == 0.)
    {
      nrayvalid++ ;
      Qtot += ThisBunch->Q[count] ;
      zpos += ThisBunch->Q[count] *
              ThisBunch->x[6*count+4] ;
    }
  }
  
  /* special cases:  only 0 or 1 valid ray, or user wants
   * all rays in 1 bin (signified by zero binwidth) */
  
  if (nrayvalid == 0)
  {
    stat = -1 ;
    goto egress ;
  }
  
  if ( (nrayvalid == 1) || (BinWidth==0) )
  {
    TheWF->nbin = 1 ;
    TheWF->binQ  = (double*)calloc(1,sizeof(double)) ;
    TheWF->ptype  = (unsigned short int*)calloc(1,sizeof(unsigned short int)) ;
    TheWF->binx  = (double*)calloc(1,sizeof(double)) ;
    TheWF->biny  = (double*)calloc(1,sizeof(double)) ;
    if (
            (TheWF->binx  == NULL) ||
            (TheWF->biny  == NULL) ||
            (TheWF->binQ  == NULL) ||
            (TheWF->ptype  == NULL) )
    {
      stat = 0 ;
      goto egress ;
    }
    TheWF->binQ[0] = Qtot;
    for (count = 0 ; count< ThisBunch->nray ; count++)
      if (ThisBunch->stop[count] == 0.)
        TheWF->binno[count] = 0 ;
    if (Qtot != 0.)
      TheWF->binx[0] = zpos / Qtot ;
    else
      TheWF->binx[0] = 0. ;
    goto egress ;
  }
  
  /* now the general case: more than 1 valid ray */
  
  Q0p5 = 0.5 * Qtot ;
  Q0p34 = 0.34 * Qtot ;
  
  /* obtain a "sort key" for the 5th DOF */
  
  SortkeyZ = GetRaySortkey( ThisBunch->x, ThisBunch->nray, 5 ) ;
  
  /* use the Sortkey to find the ray which is at the charge median (ie,
   * 50% of charge in front of it, 50% behind it) */
  
  Qtot = 0. ;
  for (count=0 ; count<ThisBunch->nray ; count++)
  {
    i = (int)SortkeyZ[count] ;
    icent = count ;
    if (ThisBunch->stop[i] == 0.)
    {
      Qtot += ThisBunch->Q[i] ;
      if (Qtot >= Q0p5)
        break ;
    }
  }
  
  /* loop forward and backward to find 34% of the charge in each
   * direction from the median point */
  
  Qplus = 0. ;
  for (count=icent+1 ; count<ThisBunch->nray ; count++)
  {
    iplus = count ;
    i = (int)SortkeyZ[count] ;
    if (ThisBunch->stop[i] == 0.)
    {
      Qplus += ThisBunch->Q[i] ;
      if (Qplus >= Q0p34)
        break ;
    }
  }
  
  Qminus = 0. ;
  for (count=icent ; count>=0 ; count--)
  {
    iminus = count ;
    i = (int)SortkeyZ[count] ;
    if (ThisBunch->stop[i] == 0.)
    {
      Qminus += ThisBunch->Q[i] ;
      if (Qminus >= Q0p34)
        break ;
    }
  }
  
  /* compute an effective sigz by scaling the interval given by iplus and
   * iminus by the ratio of 68% charge to the actual found charge */
  
  i = (int)SortkeyZ[iplus] ;
  j = (int)SortkeyZ[iminus] ;
  sigz = (ThisBunch->x[6*i+4] - ThisBunch->x[6*j+4]) ;
  
  if (Qplus+Qminus != 0.)
    sigz *= Q0p34/(Qplus+Qminus) ;
  
  /* if sigz is zero, set it to the full length between the headmost and
   * tailmost rays, or if that is zero, to the minimum allowed bunch length */
  
  if (sigz == 0.)
  {
    for (iminus = 0 ; iminus < ThisBunch->nray ; iminus++)
    {
      j = (int)SortkeyZ[iminus] ;
      if (ThisBunch->stop[j] == 0)
        break ;
    }
    
    for (iplus = ThisBunch->nray-1 ; iplus > 0 ; iplus--)
    {
      i = (int)SortkeyZ[iplus] ;
      if (ThisBunch->stop[i] == 0)
        break ;
    }
    sigz = (ThisBunch->x[6*i+4] - ThisBunch->x[6*j+4]) ;
    if (sigz == 0.)
      sigz = MIN_BUNCH_LENGTH ;
  }
  
  
  BinWidth *= sigz ; /* sets BinWidth in m, rather than sigmas */
  
  /* we know that the maximum number of bins is equal to the total z extent of the
   * bunch divided by the binwidth.  Allocate that many slots now, bearing in mind
   * that we won't necessarily fill all of them up */
  
  nbinsmax = (int)ceil((ThisBunch->x[6*(int)SortkeyZ[ThisBunch->nray-1]+4] -
          ThisBunch->x[6*(int)SortkeyZ[0]+4]) / BinWidth) ;
  TheWF->binQ  = (double*)calloc(nbinsmax,sizeof(double)) ;
  TheWF->ptype  = (unsigned short int*)calloc(nbinsmax,sizeof(unsigned short int)) ;
  TheWF->binx  = (double*)calloc(nbinsmax,sizeof(double)) ;
  TheWF->biny  = (double*)calloc(nbinsmax,sizeof(double)) ;
  
  if ( (TheWF->binQ  == NULL) ||
          (TheWF->binx  == NULL)  ||
          (TheWF->biny  == NULL) ||
          (TheWF->ptype  == NULL) )
  {
    stat = 0 ;
    goto egress ;
  }
  
  
  /* now some of the bins may contain no charge.  Loop over rays and figure out
   * how many of them have actual charge in them.  While we're at it, accumulate
   * information on the z position of rays in a given bin into the binx slot */
  
  TheWF->nbin = 0 ;
  TheWF->binQ[TheWF->nbin] = 0. ;
  TheWF->ptype[TheWF->nbin] = 0 ;
  TheWF->binx[TheWF->nbin] = 0. ;
  binstart = ThisBunch->x[6*(int)SortkeyZ[0]+4] ;
  binstop = binstart + BinWidth ;
  
  for (count=0 ; count < ThisBunch->nray ; count++)
  {
    i = (int)SortkeyZ[count] ;
    while (ThisBunch->x[6*i+4] > binstop)
    {
      binstart += BinWidth ;
      binstop  += BinWidth ;
      if (TheWF->binQ[TheWF->nbin] > 0.)
      {
        TheWF->binx[TheWF->nbin] /= TheWF->binQ[TheWF->nbin] ;
        TheWF->nbin++ ;
        TheWF->binQ[TheWF->nbin] = 0. ;
        TheWF->ptype[TheWF->nbin] = 0 ;
        TheWF->binx[TheWF->nbin] = 0. ;
      }
    }
    if (ThisBunch->stop[i] == 0.)
    {
      TheWF->binQ[TheWF->nbin] += ThisBunch->Q[i] ;
      TheWF->ptype[TheWF->nbin] = ThisBunch->ptype[i] ;
      TheWF->binx[TheWF->nbin] += ThisBunch->x[6*i+4] * ThisBunch->Q[i] ;
      TheWF->binno[i] = TheWF->nbin ;
    }
  }
  
  /* commit the last bin, if necessary */
  
  if (TheWF->binQ[TheWF->nbin] > 0.)
  {
    TheWF->binx[TheWF->nbin] /= TheWF->binQ[TheWF->nbin] ;
    TheWF->nbin++ ;
  }
  
  
  egress:
    
    return stat ;
    
}

/*==================================================================*/

/* prepare a bunch for participation in long-range wakefields in the
 * frequency domain */

/* RET:   integer 1 for success, 0 for failure, -1 for no valid rays
 * in the bunch.
 * ABORT: never.
 * FAIL:  never. */

int PrepareBunchForLRWFFreq(struct Bunch* ThisBunch, int WFno, int flag,
        double* Freq, double* K, int nmode,
        double binwidth )
{
  struct SRWF WFDummy ;
  struct LRWFFreq* TheWF ;
  int stat = 1;
  int modeno, binno, j, nbin ;
  double fx, fy, wx, wy ;
  int modex, modey ;
  
  /* is the wakefield already prepared?  If so, no need to do anything
   * more here */
  
  if (flag==0)
    TheWF = ThisBunch->TLRFreq[WFno] ;
  else
    TheWF = ThisBunch->TLRErrFreq[WFno] ;
  
  /* if the WF in question has already been convolved, we don't need to do
   * anything.  If the wakefield was previously convolved but something has
   * happened (compression, collimation, etc) to make the previous convolution
   * invalid, it is the responsibility of the invalidating routine to blow away
   * the convolved-WF data. */
  
  if (TheWF != NULL)
    goto egress ;
  
  TheWF = (struct LRWFFreq*)calloc(1,sizeof(struct LRWFFreq) ) ;
  if (TheWF == NULL)
  {
    stat = 0 ;
    BadLRWFAllocMsg( WFno+1, flag ) ;
    goto egress ;
  }
  
  /* bin the bunch according to the BinWidth parameter.  Since BinRays
   * is hard-coded to use a SRWF, we can use WFDummy to get the vals */
  
  WFDummy.nbin  = 0 ;
  WFDummy.binno = NULL ;
  WFDummy.binQ  = NULL ;
  WFDummy.binx  = NULL ;
  WFDummy.biny  = NULL ;
  
  stat = BinRays(ThisBunch, &(WFDummy), binwidth) ;
  if (stat==0)
    BadLRWFAllocMsg( WFno+1, flag ) ;
  if (stat != 1)
    goto egress ;
  
  /* transfer the x,y,Q,binno vectors to the real WF structure */
  
  TheWF->binno = WFDummy.binno ;
  TheWF->binQ  = WFDummy.binQ  ;
  TheWF->binx  = WFDummy.binx  ;
  TheWF->biny  = WFDummy.biny  ;
  TheWF->nbin  = WFDummy.nbin  ;
  nbin = TheWF->nbin ;
  
  /* now we need cos wt and sin wt values for each bin, and there's
   * one for each mode, and there are both x and y modes for each
   * design mode.  Allocate that now */
  
  TheWF->xphase = (struct LucretiaComplex*)calloc(nmode*nbin,sizeof(struct LucretiaComplex)) ;
  TheWF->yphase = (struct LucretiaComplex*)calloc(nmode*nbin,sizeof(struct LucretiaComplex)) ;
  TheWF->Wx     = (struct LucretiaComplex*)calloc(nmode*nbin,sizeof(struct LucretiaComplex)) ;
  TheWF->Wy     = (struct LucretiaComplex*)calloc(nmode*nbin,sizeof(struct LucretiaComplex)) ;
  TheWF->binVx  = (double*)calloc(nbin,sizeof(double)) ;
  TheWF->binVy  = (double*)calloc(nbin,sizeof(double)) ;
  
  
  if ( (TheWF->xphase==NULL)  || (TheWF->yphase==NULL)  ||
          (TheWF->binVx==NULL)  || (TheWF->binVy==NULL)    ||
          (TheWF->Wx==NULL)     || (TheWF->Wy==NULL)          )
  {
    stat = 0 ;
    BadLRWFAllocMsg( WFno+1, flag ) ;
    goto egress ;
  }
  
  /* Convert binx into the time difference between z=0 and the bin center */
  
  for (binno=0 ; binno<TheWF->nbin ; binno++)
    TheWF->binx[binno] /= CLIGHT ;
  
  /* compute the phase advances from t=0 to each bin, and compute the wakefield
   * contribution of each bin at t=0.  Convert the wakefield kicks to GV from
   * V to match dimensions of the ray momentum coordinate.  Also, remember that
   * error wakes have 2 K values per mode (x and y), whereas conventional wakes
   * have only 1 */
  
  for (modeno=0 ; modeno<nmode ; modeno++)
  {
    fx = Freq[2*modeno]  ;
    fy = Freq[2*modeno+1]  ;
    wx = 2*PI*fx * 1e6 ;
    wy = 2*PI*fy * 1e6;
    for (binno=0 ; binno<TheWF->nbin ; binno++)
    {
      j = binno + modeno * TheWF->nbin ;
      if (flag==0)
      {
        modex = modeno ;
        modey = modeno ;
      }
      else
      {
        modex = 2*modeno ;
        modey = 2*modeno + 1 ;
      }
      TheWF->xphase[j].Real =  cos(wx*TheWF->binx[binno]) ;
      TheWF->xphase[j].Imag = -sin(wx*TheWF->binx[binno]) ;
      TheWF->yphase[j].Real =  cos(wy*TheWF->binx[binno]) ;
      TheWF->yphase[j].Imag = -sin(wy*TheWF->binx[binno]) ;
      TheWF->Wx[j].Real = -K[modex] /1e9 *
              sin(wx*TheWF->binx[binno]) ;
      TheWF->Wx[j].Imag =  K[modex] /1e9 *
              cos(wx*TheWF->binx[binno]) ;
      TheWF->Wy[j].Real = -K[modey] /1e9 *
              sin(wy*TheWF->binx[binno]) ;
      TheWF->Wy[j].Imag =  K[modey] /1e9 *
              cos(wy*TheWF->binx[binno]) ;
    }
  }
  
  /* clear out binx */
  
  for (binno=0 ; binno<TheWF->nbin ; binno++)
    TheWF->binx[binno] = 0 ;
  
  /* assign this wakefield information to the bunch data structure */
  
  if (flag==0)
    ThisBunch->TLRFreq[WFno] = TheWF ;
  else
    ThisBunch->TLRErrFreq[WFno] = TheWF ;
  
  egress:
    
    return stat ;
    
}


/*==================================================================*/

/* compute the deflection on each bin of a given bunch due to long-range
 * transverse wakefields in the frequency domain */

/* RET:    +1 if successful, 0 if bunchno > TheKick->LastBunch+1.
 * Abort:  Never.
 * Fail:   If the number of modes in the wakefield is larger than the
 * allocated number of modes in TheKick, a segmentation fault
 * will occur.  The only way this can happen is if the user
 * performs multibunch tracking with WF, changes the number of
 * modes in a wakefield, and repeats the tracking without
 * first clearing tracking vars.  So far I haven't figured out
 * a good way to protect against this. */

int ComputeTLRFreqKicks(  struct LRWFFreq* TheTLR, double L, int flag,
        struct LRWFFreqKick* TheKick, int nmodes,
        int bunchno,
        struct LucretiaComplex* Damping,
        struct LucretiaComplex* xPhase,
        struct LucretiaComplex* yPhase,
        double* ModeTilt, double Tilt )
{
  
  int stat = 1 ;
  int dBunch, dbunchcount ;
  int bincount, modecount, j ;
  double TiltAll, cT, sT  ;
  struct LucretiaComplex KxNow, KyNow ;
  
  /* if the LastBunch parameter is -1, then there are no wakefields to
   * be used and we should set the parameter equal to the current bunch
   * number */
  
  if (TheKick->LastBunch==-1)
    TheKick->LastBunch=bunchno ;
  
  /* depending on the relationship between the LastBunch and ThisBunch,
   * we do different things.  If ThisBunch > LastBunch+1 (ie non-
   * sequential tracking), set error status and exit */
  
  dBunch = bunchno - TheKick->LastBunch ;
  
  /* if the current bunch is earlier than or equal to the
   * LastBunch, we can zero the stored kicks */
  
  if (dBunch < 1)
  {
    for (modecount=0 ; modecount<nmodes ; modecount++)
    {
      TheKick->xKick[modecount].Real = 0;
      TheKick->xKick[modecount].Imag = 0;
      TheKick->yKick[modecount].Real = 0;
      TheKick->yKick[modecount].Imag = 0;
    }
  }
  
  /* on the third hand, if this bunch is exactly the next bunch after
   * the last one tracked here, then we should damp it and advance its
   * phase to find the remaining voltage at the t=0 of the current bunch */
  
  else
  {
    
    /* loop over modes */
    
    for (modecount=0 ; modecount<nmodes ; modecount++)
      for (dbunchcount=0 ; dbunchcount<dBunch ; dbunchcount++)
      {
        
        /* damp the mode and phase-advance it to t = 0 of the current bunch */
        
        TheKick->xKick[modecount].Real *= Damping[modecount].Real ;
        TheKick->xKick[modecount].Imag *= Damping[modecount].Real ;
        TheKick->yKick[modecount].Real *= Damping[modecount].Imag ;
        TheKick->yKick[modecount].Imag *= Damping[modecount].Imag ;
        
        TheKick->xKick[modecount] =
                ComplexProduct( TheKick->xKick[modecount],
                xPhase[modecount]          ) ;
        TheKick->yKick[modecount] =
                ComplexProduct( TheKick->yKick[modecount],
                yPhase[modecount]          ) ;
      }
  }
  
  /* at this point we can go ahead and loop over modes, loop over bins,
   * and kick each bin based on the voltage of each mode at that bin */
  
  for (modecount=0 ; modecount<nmodes ; modecount++)
  {
    
    /* compute the total tilt angle, its sine and cosine */
    
    TiltAll = Tilt + ModeTilt[modecount] ;
    cT = cos(TiltAll) ;
    sT = sin(TiltAll) ;
    
    for (bincount=0 ; bincount<TheTLR->nbin ; bincount++)
    {
      
      /* make a shortcut to the phase factor for this mode, this bin */
      
      j = bincount + modecount * TheTLR->nbin ;
      
      /* if this is the first mode, clear existing deflections */
      
      if (modecount==0)
      {
        TheTLR->binVx[bincount] = 0 ;
        TheTLR->binVy[bincount] = 0 ;
      }
      
      /* phase-advance the current mode to the current bin time */
      
      KxNow = ComplexProduct( TheKick->xKick[modecount],
              TheTLR->xphase[j]          ) ;
      KyNow = ComplexProduct( TheKick->yKick[modecount],
              TheTLR->yphase[j]          ) ;
      
      /* sum the deflecting voltages */
      
      TheTLR->binVx[bincount] += L *
              ( KxNow.Real * cT - KyNow.Real * sT ) ;
      TheTLR->binVy[bincount] += L *
              ( KxNow.Real * sT + KyNow.Real * cT ) ;
      
    }
    
    /* at this point, each bin has been kicked by mode # modecount.  We
     * can now loop over bins again and increase that mode by the amount
     * of the excitation from the bins of the bunch. */
    
    for (bincount=0 ; bincount<TheTLR->nbin ; bincount++)
    {
      j = bincount + modecount * TheTLR->nbin ;
      KxNow = TheTLR->Wx[j] ;
      KyNow = TheTLR->Wy[j] ;
      if (flag == 0) /* not an error wake */
      {
        KxNow.Real *= (  TheTLR->binx[bincount]*cT +
                TheTLR->biny[bincount]*sT   ) ;
        KxNow.Imag *= (  TheTLR->binx[bincount]*cT +
                TheTLR->biny[bincount]*sT   ) ;
        KyNow.Real *= ( -TheTLR->binx[bincount]*sT +
                TheTLR->biny[bincount]*cT   ) ;
        KyNow.Imag *= ( -TheTLR->binx[bincount]*sT +
                TheTLR->biny[bincount]*cT   ) ;
      }
      else /* error wake */
      {
        KxNow.Real *= TheTLR->binQ[bincount] ;
        KxNow.Imag *= TheTLR->binQ[bincount] ;
        KyNow.Real *= TheTLR->binQ[bincount] ;
        KyNow.Imag *= TheTLR->binQ[bincount] ;
      }
      TheKick->xKick[modecount].Real += KxNow.Real ;
      TheKick->xKick[modecount].Imag += KxNow.Imag ;
      TheKick->yKick[modecount].Real += KyNow.Real ;
      TheKick->yKick[modecount].Imag += KyNow.Imag ;
    }
  }
  
  /* finally, one last loop thru the bins to zero the x and y positions */
  
  for (bincount=0 ; bincount<TheTLR->nbin ; bincount++)
  {
    TheTLR->binx[bincount] = 0 ;
    TheTLR->biny[bincount] = 0 ;
  }
  
  /* set the most-recent bunch in the kick vector */
  
  TheKick->LastBunch = bunchno ;
  
  /*egress:*/
  
  return stat ;
  
}



/*==================================================================*/

/* return the product of 2 complex numbers */

struct LucretiaComplex ComplexProduct(struct LucretiaComplex c1,
        struct LucretiaComplex c2 )
{
  struct LucretiaComplex c3 ;
  
  c3.Real = c1.Real*c2.Real - c1.Imag*c2.Imag ;
  c3.Imag = c1.Real*c2.Imag + c1.Imag*c2.Real ;
  return c3 ;
}

/*==================================================================*/

/* compute the map for a sector bend, based on the MAD formulas. */

void GetMADSBendMap( double L, double Theta, double intB, double intG,
        double P, Rmat R, double TijkTransv[4][10],
        double T5xx[10], int FirstCall )
        
{
  
  static double h, hov2, h2, K1, hK1, kx2, ky2 ;
  static double cx, sx, dx ;
  static double cy, sy ;
  static double J1, J2, J3 ;
  static double Pdes ;
  static double R0[6][6], Tijk0[4][10], T5xx0[10] ;
  static double Tij6[13] ;
  double delta, dp1, dp12, k ;
  int icount, jcount ;
  
  
  /* since many of the bend magnet parameters are calculated for its
   * matched momentum (ie the momentum for which intB and L yield
   * bend angle Theta), those parameters need to be computed only
   * once per bend.  If this is the first call for computing this
   * bend, do those now */
  
  if (FirstCall == 1)
  {
    
    /* compute the momentum which is matched to the bending magnet */
    
    Pdes = intB / Theta / GEV2TM ;
    
    /* compute the curvature and the nominal focusing strength for
     * the design momentum */
    
    h = Theta / L ;
    hov2 = h/2 ;
    h2 = h*h ;
    K1 = intG/L/Pdes/GEV2TM ;
    hK1 = h*K1 ;
    
    /* compute the horizontal and vertical k functions */
    
    kx2 = h*h + K1 ;
    ky2 = -K1 ;
    
    /* compute the cosine-like, sine-like, and dispersive ray values
     * at the end of the magnet for the design momentum */
    
    if (kx2 == 0)
    {
      cx = 1 ;
      sx = L ;
      dx = L * L / 2 ;
    }
    if (kx2 > 0)
    {
      k = sqrt(kx2)   ;
      cx = cos(L*k)   ;
      sx = sin(L*k)/k ;
      dx = (1-cx)/kx2 ;
    }
    if (kx2 < 0)
    {
      k = sqrt(-kx2)   ;
      cx = cosh(L*k)   ;
      sx = sinh(L*k)/k ;
      dx = (1-cx)/kx2  ;
    }
    
    if (ky2 == 0)
    {
      cy = 1 ;
      sy = L ;
    }
    if (ky2 > 0)
    {
      k = sqrt(ky2)   ;
      cy = cos(L*k)   ;
      sy = sin(L*k)/k ;
    }
    if (ky2 < 0)
    {
      k = sqrt(-ky2)   ;
      cy = cosh(L*k)   ;
      sy = sinh(L*k)/k ;
    }
    
    /* compute the J1 through J3 integrals */
    
    J1 = (L-sx)/kx2 ;
    J2 = (3*L - 4*sx + sx*cx)/(2*kx2*kx2) ;
    J3 = (15*L -22*sx + 9*sx*cx - 2*sx*cx*cx)/(6*kx2*kx2*kx2) ;
    
    /* blank out terms in the linear map which are always zero, and
     * set R_55 and R_66 to 1 */
    
    R[0][2] = 0. ;
    R[0][3] = 0. ;
    R[0][4] = 0. ;
    
    R[1][2] = 0. ;
    R[1][3] = 0. ;
    R[1][4] = 0. ;
    
    R[2][0] = 0. ;
    R[2][1] = 0. ;
    R[2][4] = 0. ;
    R[2][5] = 0. ;
    
    R[3][0] = 0. ;
    R[3][1] = 0. ;
    R[3][4] = 0. ;
    R[3][5] = 0. ;
    
    R[4][2] = 0. ;
    R[4][3] = 0. ;
    R[4][4] = 1. ;
    
    R[5][0] = 0. ;
    R[5][1] = 0. ;
    R[5][2] = 0. ;
    R[5][3] = 0. ;
    R[5][4] = 0. ;
    R[5][5] = 1. ;
    
    /* build the linear map for the transverse degrees of freedom */
    
    R[0][0] = cx ;
    R[1][1] = cx ;
    R[0][1] = sx ;
    R[1][0] = -sx * kx2 ;
    
    R[2][2] = cy ;
    R[3][3] = cy ;
    R[2][3] = sy ;
    R[3][2] = -sy * ky2 ;
    
    /* set the R_5j terms where j is a transverse coordinate, bearing
     * in mind that the 5th coordinate has a sign change between MAD
     * and Lucretia */
    
    R[4][0] = h*sx ;
    R[4][1] = h*dx  ;
    R[4][5] = h * h * J1 ;
    
    /* set the R_i6 terms */
    
    R[0][5] = h*dx ;
    R[1][5] = h*sx  ;
    R[4][5] = h*h*J1 ;
    
    /* at this point, if we are doing this for the purposes of an Rmat
     * calculation, as signified by P==0, we are done and can return */
    
    if (P<=0)
      return ;
    
    /* otherwise, cache the R matrix for future use */
    
    for (icount=0; icount<6; icount++)
      for (jcount=0 ; jcount<6 ; jcount++)
        R0[icount][jcount] = R[icount][jcount] ;
    
    /* compute a cache-worthy set of T matrix components as well.
     * Note that the MAD Tijk for off-diagonal j,k is half of the
     * equivalent TRANSPORT map, ie, when TRANSPORT says
     *
     * x_i = sum_j R_ij x_j + sum_j sum_k>=j T_ijk x_j x_k,
     *
     * MAD says
     *
     * x_i = sum_j R_ij x_j + sum_j sum_k T_ijk x_j x_k.
     *
     * Since we do something more like TRANSPORT tracking, the
     * off-diagonal terms below are doubled from the MAD ones.  */
    
    Tijk0[0][0] = -hov2 * kx2 * sx*sx ;
    Tijk0[0][1] =  2 * hov2 * sx * cx  ;
    Tijk0[0][4] =  hov2 * cx * dx  ;
    Tijk0[0][7] =  0. ;
    Tijk0[0][8] =  0. ;
    Tijk0[0][9] = -hov2 * dx ;
    
    Tijk0[1][0] =  0. ;
    Tijk0[1][1] =  0. ;
    Tijk0[1][4] = -hov2 * sx  ;
    Tijk0[1][7] =  0. ;
    Tijk0[1][8] =  0. ;
    Tijk0[1][9] =  Tijk0[1][4] ;
    
    Tijk0[2][2] =  0. ;
    Tijk0[2][3] =  2 * hov2 * sx * cy  ;
    Tijk0[2][5] =  0. ;
    Tijk0[2][6] =  2 * hov2 * dx * cy  ;
    
    Tijk0[3][2] =  0. ;
    Tijk0[3][3] =  0. ;
    Tijk0[3][5] =  0. ;
    Tijk0[3][6] =  0. ;
    
    /* T522, T533, T534, T544 */
    
    T5xx0[2] =  0.5 * sx  ;
    T5xx0[3] =  0. ;
    T5xx0[4] =  0. ;
    T5xx0[5] = -0.5 * h2 * J1 + (L+cy*sy)/4 ;
    
    /* T matrix elements with a 6 in them */
    
    Tij6[0]  = h2*sx*sx   ;                           /* T116 */
    Tij6[1]  = 2*(h2*(sx*dx+cx*J1)/4 - (sx+L*cx)/4) ; /* T126 */
    Tij6[2]  = h2*h*sx*J1/2-h*L*sx/2 ;                /* T166 */
    Tij6[3]  = 0. ;                                   /* T216 */
    Tij6[4]  = 0. ;                                   /* T226 */
    Tij6[5]  = 0. ;                                   /* T266 */
    Tij6[6]  = 0. ;                                   /* T336 */
    Tij6[7]  = 2*(h2*J1*cy/2 - (sy+L*cy)/4) ;         /* T346 */
    Tij6[8]  = 0. ;                                   /* T436 */
    Tij6[9]  = 0. ;                                   /* T446 */
    Tij6[10] = 0 ;                                    /* T516 */
    Tij6[11] = 0 ;                                    /* T526 */
    Tij6[12] = 0 ;                                    /* T566 */
    
    /* now put in the terms which depend on K1 */
    
    if (K1 != 0.)
    {
      double hK1ov2 = hK1/2 ;
      double hK1ov3 = hK1/3 ;
      Tijk0[0][0] -= hK1ov3 * (sx*sx + dx) ;
      Tijk0[0][1] -= 2 * hK1ov3 * (sx * dx)    ;
      Tijk0[0][4] -= hK1ov3 * dx*dx        ;
      Tijk0[0][7] += hK1ov2 * dx           ;
      
      Tijk0[1][0] -= hK1ov3 * sx * (1+2*cx)  ;
      Tijk0[1][1] -= 2 * hK1ov3 * dx * (1+2*cx)  ;
      Tijk0[1][4] -= hK1 * sx * dx / 1.5     ;
      Tijk0[1][7] += hK1ov2 * sx             ;
      
      Tijk0[2][2] += 2 * hK1ov2 * sx * sy  ;
      Tijk0[2][5] += 2 * hK1ov2 * dx * sy  ;
      
      Tijk0[3][2] += 2 * hK1ov2 * sx * cy  ;
      Tijk0[3][3] += 2 * hK1ov2 * sx * sy  ;
      Tijk0[3][5] += 2 * hK1ov2 * dx * cy  ;
      Tijk0[3][6] += 2 * hK1ov2 * dx * sy  ;
      
      T5xx0[0] = -hK1 * h * (sx*dx + 3*J1) / 6 + K1*(L-sx*cx)/4 ;
      T5xx0[1] =  2 * (-hK1 * h * dx*dx / 6 - K1 * sx * sx / 4) ;
      T5xx0[2] += 2 * (-hK1ov3 * h * J2 + K1 * (J1-sx*dx)/4 ) ;
      T5xx0[3] += hK1ov2 * h * J1 - K1 * (L-cy*sy)/4 ;
      T5xx0[4] += 2 * K1 * sy * sy / 4 ;
      
      Tij6[0]  += 2*(-hK1*h*(3*sx*J1-dx*dx)/6 + K1*L*sx/4) ;
      Tij6[1]  += 2*(-hK1*h*(sx*dx*dx-2*cx*J2)/6) ;
      Tij6[2]  += -hK1ov3*h2*(dx*dx*dx-2*sx*J2) ;
      Tij6[3]  += 2*(-hK1*h*(3*cx*J1+sx*dx)/6 - K1*(sx-L*cx)/4) ;
      Tij6[4]  += 2*(-hK1*h*(3*sx*J1+dx*dx)/6 + K1*L*sx/4) ;
      Tij6[5]  += -hK1ov3*h2*(sx*dx*dx-2*cx*J2)
      -hK1ov2*(cx*J1-sx*dx) ;
      Tij6[6]  +=  2*(h2*K1*J1*sy/2 - K1*L*sy/4) ;
      Tij6[8]  +=  2*(hK1ov2*h*J1*cy + K1*(sy-L*cy)/4) ;
      Tij6[9]  +=  2*(hK1ov2*h*J1*sy - K1*L*sy/4) ;
      Tij6[10] +=  2*(-hK1*h2*(3*dx*J1-4*J2)/6 - hK1*J1*(1+cx)/4) ;
      Tij6[11] += 2*(-hK1*h2*(dx*dx*dx-2*sx*J2)/6 - hK1*sx*J1/4) ;
      Tij6[12] += -hK1ov3*h2*h*(3*J3-2*dx*J2)
      -hK1*h*(sx*dx*dx-J2*(1+2*cx))/6 ;
    }
  }
  
  /* Now that we have a good cached version of the design R and T
   * matrix elements, we need to generate a version which is correct
   * for the current-momentum particle.  To do this, we need to do
   * three things:
   * -> Since tracking doesn't have a concept of reference momentum,
   * the R_i6 terms are used as the centroid offsets; so we need
   * to multiply them by the actual delta, and add the appropriate
   * T-matrix terms (ie, R[0][5] -> eta * delta + T_166 * delta^2)
   * -> Since tracking doesn't have a concept of reference momentum
   * the chromatic effects need to be included in the R matrix
   * (ie, R[0][0] -> R_11 + T_116 * delta)
   * -> Since MAD's px is the transverse momentum wrt the reference
   * momentum, and Lucretia's px is the transverse momentum wrt
   * the actual momentum, the R- and T- matrices which operate on
   * transverse momenta need to be scaled by various powers of
   * 1+delta.
   *
   * Do all that now. */
  
  delta = (P-Pdes)/Pdes ;
  dp1  = delta + 1 ;
  dp12 = dp1 * dp1 ;
  
  /* start by converting the R_i6 terms into true kicks, and while we're
   * at it put them into the return matrix from the cache matrix */
  
  R[0][5] = R0[0][5] * delta + Tij6[2]  * delta * delta ;
  R[1][5] = R0[1][5] * delta + Tij6[5]  * delta * delta ;
  R[4][5] = R0[4][5] * delta + Tij6[12] * delta * delta ;
  
  /* now modify the Rij terms by the Tij6 terms */
  
  R[0][0] = R0[0][0] + Tij6[0]  * delta ;
  R[0][1] = R0[0][1] + Tij6[1]  * delta ;
  R[1][0] = R0[1][0] + Tij6[3]  * delta ;
  R[1][1] = R0[1][1] + Tij6[4]  * delta ;
  R[2][2] = R0[2][2] + Tij6[6]  * delta ;
  R[2][3] = R0[2][3] + Tij6[7]  * delta ;
  R[3][2] = R0[3][2] + Tij6[8]  * delta ;
  R[3][3] = R0[3][3] + Tij6[9]  * delta ;
  R[4][0] = R0[4][0] + Tij6[10] * delta ;
  R[4][1] = R0[4][1] + Tij6[11] * delta ;
  
  /* now perform corrections on terms which involve transverse momentum */
  
  R[0][1] *= dp1 ;
  R[1][0] /= dp1 ;
  R[1][5] /= dp1 ;
  R[2][3] *= dp1 ;
  R[3][2] /= dp1 ;
  R[4][1] *= dp1 ;
  
  /* copy over the longitudinal terms, and adjust for delta in the
   * process.  NB I'm not sure that it's correct to make such an
   * adjustment here, since I think the implicit momentum dependence
   * of the MAD PX leads to errors in the T_ijk terms where j, k
   * are momenta.  But for now I'm leaving them in. */
  
  TijkTransv[0][0] = Tijk0[0][0] ;
  TijkTransv[0][1] = Tijk0[0][1] * dp1 ;
  TijkTransv[0][4] = Tijk0[0][4] * dp12 ;
  TijkTransv[0][7] = Tijk0[0][7] ;
  TijkTransv[0][8] = Tijk0[0][8] * dp1 ;
  TijkTransv[0][9] = Tijk0[0][9] * dp12 ;
  
  TijkTransv[1][0] = Tijk0[1][0] / dp1 ;
  TijkTransv[1][1] = Tijk0[1][1] ;
  TijkTransv[1][4] = Tijk0[1][4] * dp1 ;
  TijkTransv[1][7] = Tijk0[1][7] / dp1 ;
  TijkTransv[1][8] = Tijk0[1][8] ;
  TijkTransv[1][9] = Tijk0[1][9] * dp1 ;
  
  TijkTransv[2][2] = Tijk0[2][2] ;
  TijkTransv[2][3] = Tijk0[2][3] * dp1 ;
  TijkTransv[2][5] = Tijk0[2][5] * dp1 ;
  TijkTransv[2][6] = Tijk0[2][6] * dp12 ;
  
  TijkTransv[3][2] = Tijk0[3][2] / dp1 ;
  TijkTransv[3][3] = Tijk0[3][3] ;
  TijkTransv[3][5] = Tijk0[3][5] ;
  TijkTransv[3][6] = Tijk0[3][6] * dp1 ;
  
  /* note that we do not correct T_522 or T_544 for delta.  This is
   * because these terms care about the beam angle and not its
   * transverse momentum (a subtle distinction).  In a more perfect
   * tracking map there would be compensating V_52266 and V_54466
   * terms that would cancel the delta correction, but that world is
   * not the one we live in. */
  
  T5xx[0] = T5xx0[0]  ;
  T5xx[1] = T5xx0[1]  ;
  T5xx[2] = T5xx0[2]  ;
  T5xx[3] = T5xx0[3]  ;
  T5xx[4] = T5xx0[4]  ;
  T5xx[5] = T5xx0[5]  ;
  
  /* and that's (finally!) it. */
  
}

/* compute the map for a sector bend using standard Lucretia variables.
 * If argument T5xx != NULL, calculation is for tracking and R matrix
 * components R[1][6], R[2][6], R[5][6] contain exact x, px, z offsets for
 * a particle of the given momentum passing through the magnet.  If on the
 * other hand T5xx == NULL, calculation is for RMAT operations and R
 * matrix components R[1][6], R[2][6], R[5][6] are the standard Transport
 * dx/ddelta, dpx/ddelta, dz/ddelta.
 *
 * If TijkTransv != NULL, then also calculate second-order transverse terms
 * for the body of the bend according to MAD formulism (see GetMADSbendMap)
 *
 * RET:    none.
 * ABORT:  never.
 * FAIL:   Will fail if either the magnet length or particle momentum is
 * less than or equal to zero.  Calling routine is responsible for
 * preventing such an occurrence.
 *
 */
void GetLucretiaSBendMap( double L, double Theta, double intB, double intG,
        double P, Rmat R, double T5xx[10], double TijkTransv[4][10], double Tij6[13] )
{
  
  double h, K0, K1, kx, ky, kx2, dx, hov2 ;
  double h_K0, brho, kxL, kyL, kx2L, ky2L ;
  double cx, sx, s2x, c2x ;
  double cy, sy, s2y, c2y ;
  double sxovkx, s2xov2kx ;
  double syovky, s2yov2ky ;
  double signkx, signky ;
  double h2, J1 ;
  int ind_i, ind_j ;
  
  /* calculate temporary variables */
  
  brho = P*GEV2TM ;
  K0 = intB / L / brho ;
  K1 = intG / L / brho ;
  h = Theta / L ;
  
  /* the temp variable for the dispersive terms changes depending on whether
   * we are doing R-matrix or tracking calculations */
  
  if (T5xx != NULL)
    h_K0 = h - K0 ;
  else
    h_K0 = K0 ;
  
  kx = sqrt(fabs(h*K0+K1)) ;
  if (kx <= 5.e-10)
    kx = 0. ;
  ky = sqrt(fabs(K1)) ;
  if (ky <= 5.e-10)
    ky = 0. ;
  
  kxL = kx * L ;
  kyL = ky * L ;
  kx2L = 2*kxL ;
  ky2L = 2*kyL ;
  
  if (kx == 0)
  {
    cx = 1 ;
    sx = 0 ;
    sxovkx = L ;
    c2x = 1 ;
    s2x = 0 ;
    s2xov2kx = L ;
    signkx = 0 ;
  }
  else if (K0*h+K1 > 0.)
  {
    cx = cos(kxL) ;
    sx = sin(kxL) ;
    sxovkx = sx/kx ;
    c2x = cos(kx2L) ;
    s2x = sin(kx2L) ;
    s2xov2kx = s2x / 2 / kx ;
    signkx = 1 ;
  }
  else /* (K0*h+K1 < 0.) */
  {
    cx = cosh(kxL) ;
    sx = sinh(kxL) ;
    sxovkx = sx/kx ;
    c2x = cosh(kx2L) ;
    s2x = sinh(kx2L) ;
    s2xov2kx = s2x / 2 / kx ;
    signkx = -1 ;
  }
  
  
  
  if (ky == 0.)
  {
    cy = 1 ;
    sy = 0 ;
    syovky = L ;
    c2y = 1 ;
    s2y = 0 ;
    s2yov2ky = L ;
    signky = 0 ;
  }
  else if (K1 < 0.)
  {
    cy = cos(kyL) ;
    sy = sin(kyL) ;
    syovky = sy/ky ;
    c2y = cos(ky2L) ;
    s2y = sin(ky2L) ;
    s2yov2ky = s2y / 2 / ky ;
    signky = -1 ;
  }
  else /* (K1 > 0.) */
  {
    cy = cosh(kyL) ;
    sy = sinh(kyL) ;
    syovky = sy/ky ;
    c2y = cosh(ky2L) ;
    s2y = sinh(ky2L) ;
    s2yov2ky = s2y / 2 / ky ;
    signky = 1 ;
  }
  
  
  /* zero the terms which are always zero in the R-matrix */
  
  R[0][2] = 0. ;
  R[0][3] = 0. ;
  R[0][4] = 0. ;
  
  R[1][2] = 0. ;
  R[1][3] = 0. ;
  R[1][4] = 0. ;
  
  R[2][0] = 0. ;
  R[2][1] = 0. ;
  R[2][4] = 0. ;
  R[2][5] = 0. ;
  
  R[3][0] = 0. ;
  R[3][1] = 0. ;
  R[3][4] = 0. ;
  R[3][5] = 0. ;
  
  R[4][2] = 0. ;
  R[4][3] = 0. ;
  
  R[5][0] = 0. ;
  R[5][1] = 0. ;
  R[5][2] = 0. ;
  R[5][3] = 0. ;
  R[5][4] = 0. ;
  
  /* R_55 and R_66 are always unity */
  
  R[4][4] = 1. ;
  R[5][5] = 1. ;
  
  /* compute the 4x4 transverse map first */
  
  R[0][0] = cx ;
  R[0][1] = sxovkx ;
  R[1][0] = -signkx*kx*sx ;
  R[1][1] = cx ;
  
  
  R[2][2] = cy ;
  R[2][3] = syovky ;
  R[3][2] = signky*ky*sy ;
  R[3][3] = cy ;
  
  /* longitudinal R-matrix terms:  if T5xx is absent, these come out as traditional
   * R-matrix terms, otherwise they will be true deflections, thanks to the clever
   * definition of h_K0 in the two different circumstances */
  
  R[4][0] = h*sxovkx ;
  R[1][5] = h_K0*sxovkx ;
  if (kx != 0.)
  {
    R[4][1] = signkx*h*(1-cx)/kx/kx ;
    R[4][5] = signkx*h_K0*h*(kxL-sx)/kx/kx/kx ;
    
    R[0][5] = signkx*h_K0*(1-cx)/kx/kx ;
  }
  else /* kx == 0. */
  {
    R[4][1] = h*L*L/2 ;
    R[4][5] = h_K0*h*L*L*L/6 ;
    
    R[0][5] = h_K0*L*L/2 ;
  }
  
  /* if this calculation was not for tracking, we're done */
  
  if (T5xx==NULL)
    goto egress ;
  
  /* otherwise:  calculate the T5xx terms, and momentum-dependent corrections for the
   * R_5j terms */
  
  R[4][0] += h_K0*(s2xov2kx-L)/2 ;
  T5xx[0] = signkx*kx*(kx*L-s2x/2)/4 ;		/* T_511 */
  T5xx[1] = (c2x-1)/4 ;						/* T_512 */
  T5xx[2] = (L+s2xov2kx)/4 ;					/* T_522 */
  T5xx[3] = signky*ky*(s2y/2-ky*L)/4 ;		/* T_533 */
  T5xx[4] = (c2y-1)/4 ;						/* T_534 */
  T5xx[5] = (L+s2yov2ky)/4 ;					/* T_544 */
  
  if (kx != 0.)
  {
    R[4][1] += signkx*h_K0*(1-c2x)/4/kx/kx ;
    R[4][5] += signkx*h_K0*h_K0*(2*kxL-s2x)/8/kx/kx/kx ;
  }
  else /* kx == 0 */
  {
    R[4][1] += h_K0*L*L/2 ;
    R[4][5] += h_K0*h_K0*L*L*L/6 ;
  }
  
  // Calculate T-matrix terms for transverse dimensions
  // Using TRANSPORT convention, therfore these are 2X values stated for MAD
  // (use SUM_i,j for j>i)
  if (TijkTransv != NULL)
  {
    h2 = h*h ;
    hov2 = h / 2 ;
    kx2 = h*h + K1 ;
    J1 = (L-sx)/kx2 ;
    dx = L * L / 2 ;
    for (ind_i=0;ind_i<4;ind_i++)
      for (ind_j=0;ind_j<10;ind_j++)
        TijkTransv[ind_i][ind_j] = 0. ;
    TijkTransv[0][0] = -hov2 * kx2 * sx*sx ; // T111
    TijkTransv[0][1] =  h * sx * cx  ; // T112
    TijkTransv[0][4] =  h * cx * dx  ; // T122
    TijkTransv[0][9] = -h * dx ; // T144
    TijkTransv[1][4] = -hov2 * sx  ; // T222
    TijkTransv[1][9] =  TijkTransv[1][4] ; // T244
    TijkTransv[2][3] =  h * sx * cy  ; // T314
    TijkTransv[2][6] =  h * dx * cy  ; // T324
    Tij6[0]  = h2*sx*sx ;                             /* T116 */
    Tij6[1]  = h2*(sx*dx+cx*J1)/2 - (sx+L*cx)/2 ;     /* T126 */
    Tij6[2]  = h2*h*sx*J1 - h*L*sx ;                  /* T166 */
    Tij6[3]  = 0. ;                                   /* T216 */
    Tij6[4]  = 0. ;                                   /* T226 */
    Tij6[5]  = 0. ;                                   /* T266 */
    Tij6[6]  = 0. ;                                   /* T336 */
    Tij6[7]  = h2*J1*cy - (sy+L*cy)/2 ;               /* T346 */
    Tij6[8]  = 0. ;                                   /* T436 */
    Tij6[9]  = 0. ;                                   /* T446 */
    Tij6[10] = 0 ;                                    /* T516 */
    Tij6[11] = 0 ;                                    /* T526 */
    Tij6[12] = 0 ;                                    /* T566 */
  }
  
  egress :
    
    return ;
    
}

/*==================================================================*/

/* Calculate synchrotron radiation parameters for a particle passing
 * through a magnetic field.  I borrow liberally from the DIMAD
 * expressions, though I cast them in terms of Lucretia-like variables
 * here.  Since the DIMAD expressions were clearly only correct in
 * the ultra-relativistic limit, this function cavalierly treats
 * total energy in GeV and momentum in GeV/c as equivalent. */

/* NB:  the critical energy and mean # of photons, required returns;
 * if umean or urms is passed as NULL,
 * then there will be no attempt to return the mean or RMS energy loss
 * through the element. */

void CalculateSRPars( double P, double BL, double L,
        double* uc, double* nphot,
        double* umean, double* urms )
{
  double B = BL / L ;
  double P2 = P * P ;
  /* critical energy */
  
  *uc = 6.6501369e-7 * P2 * B ;
  
  /* mean # of photons emitted */
  
  *nphot = 6.1793319 * BL ;
  
  /* mean energy loss per particle */
  
  if ( umean != NULL )
    *umean = 1.2654e-6 * P2 * B * BL ;
  
  /* RMS energy loss per particle */
  
  if ( urms != NULL )
    *urms = sqrt(1.1133e-12 * P2 * P2 * B * B * BL) ;
  
  
  /*egress:*/
  
  return ;
  
}

/*==================================================================*/

/* Compute a Poisson-distributed random number with a given mean
 * value.  This function has been converted the Numerical Recipes
 * in C version to Matlab m-file and back again, and makes use of
 * the Matlab/Octave gammaln function as well as the Matlab/Octave
 * rand function (via the GetRanFlatVec and GammaLog wrapper
 * functions. */
#ifdef __CUDACC__
__device__ int poidev_gpu( double xm, curandState_t *rState )
{
  double sq, alxm, g ;
  double em, y, t ;
  int success=0 ;
  
  
  sq = sqrt(2*xm) ;
  alxm = log(xm) ;
  g = xm*alxm - GammaLog_gpu(xm+1.) ;
  
  while (success !=1)
  {
    em = -1. ;
    while (em < 0)
    {
      y = tan(  PI * RanFlatVecPtr_gpu(rState)   ) ;
      em = sq*y+xm ;
    }
    em = floor(em) ;
    t = 0.9*(1+y*y)*exp(em*alxm-GammaLog_gpu(em+1)-g) ;
    if (RanFlatVecPtr_gpu(rState) <= t)
      success = 1 ;
  }
  
  return (int)em ;
  
}
#endif

int poidev( double xm )
{
  double sq, alxm, g ;
  double em, y, t ;
  int success=0 ;
  
  
  sq = sqrt(2*xm) ;
  alxm = log(xm) ;
  g = xm*alxm - GammaLog(xm+1.) ;
  
  while (success !=1)
  {
    em = -1. ;
    while (em < 0)
    {
      y = tan(  PI * *( RanFlatVecPtr(1) )   ) ;
      em = sq*y+xm ;
    }
    em = floor(em) ;
    t = 0.9*(1+y*y)*exp(em*alxm-GammaLog(em+1)-g) ;
    if (*RanFlatVecPtr(1) <= t)
      success = 1 ;
  }
  
  return (int)em ;
  
}

/*==================================================================*/

/* Random generation of an SR photon, with energy normalized to the
 * critical energy, via Wolski's method */
#ifdef __CUDACC__
__device__ double SRSpectrumAW_gpu( curandState_t *rState )
{
  double r = RanFlatVecPtr_gpu(rState) ;
  return 0.57 * pow(r,-1./3) * pow((1-r),PI) ;
}
#endif
double SRSpectrumAW( )
{
  double r = *RanFlatVecPtr(1) ;
  return 0.57 * pow(r,-1./3) * pow((1-r),PI) ;
}

/*==================================================================*/

/* Random generation of an SR photon, with energy normalized to the
 * critical energy, via Burkhardt's method */
#ifdef __CUDACC__
__device__ double SRSpectrumHB_gpu( curandState_t *rState )
{
  double a1,a2,c1,xlow,ratio;
  double appr,exact,result;
  
  double xmin = 0. ;
  double sum1ap, sum2ap ;
  
  xlow=1.;
  
  /* initialize constants used in the approximate expressions
   * for SYNRAD   (integral over the modified Bessel function K5/3) */
  
  a1=SynRadC(1.e-38)/pow(1.e-38,-2./3.); /* = 2**2/3 GAMMA(2/3) */
  a2=SynRadC(xlow)/exp(-xlow);
  c1=pow(xmin,1./3.);
  
  /* calculate the integrals of the approximate expressions */
  
  sum1ap=3.*a1*(1.-pow(xmin,1./3.)); /* integral xmin --> 1  */
  sum2ap=a2*exp(-1.);                /* integral 1 --> infin */
  ratio=sum1ap/(sum1ap+sum2ap);
  
  /* Init done, now generate */
  
  do {
    if(RanFlatVecPtr_gpu(rState)<ratio) /* use low energy approximation */
    {
      result=c1+(1.-c1)*RanFlatVecPtr_gpu(rState);
      result*=result*result;  /* take to 3rd power; */
      exact=SynRadC(result);
      appr=a1*pow(result,-2./3.);
    }
    else                        /* use high energy approximation */
    {
      result=xlow-log(RanFlatVecPtr_gpu(rState));
      exact=SynRadC(result);
      appr=a2*exp(-result);
    }
  }
  while(exact<appr*RanFlatVecPtr_gpu(rState)); /* reject in proportion of approx */
  return result;                         /* result now exact spectrum with unity weight */
}
#endif
double SRSpectrumHB( )
{
  static int DoInit=1;
  static double a1,a2,c1,xlow,ratio;
  double appr,exact,result;
  
  if(DoInit == 1)
  {
    double xmin = 0. ;
    double sum1ap, sum2ap ;
    
    DoInit=0;
    xlow=1.;
    
    /* initialize constants used in the approximate expressions
     * for SYNRAD   (integral over the modified Bessel function K5/3) */
    
    a1=SynRadC(1.e-38)/pow(1.e-38,-2./3.); /* = 2**2/3 GAMMA(2/3) */
    a2=SynRadC(xlow)/exp(-xlow);
    c1=pow(xmin,1./3.);
    
    /* calculate the integrals of the approximate expressions */
    
    sum1ap=3.*a1*(1.-pow(xmin,1./3.)); /* integral xmin --> 1  */
    sum2ap=a2*exp(-1.);                /* integral 1 --> infin */
    ratio=sum1ap/(sum1ap+sum2ap);
    
  }
  
  /* Init done, now generate */
  do {
    if(*RanFlatVecPtr(1)<ratio) /* use low energy approximation */
    {
      result=c1+(1.-c1)*(*RanFlatVecPtr(1));
      result*=result*result;  /* take to 3rd power; */
      exact=SynRadC(result);
      appr=a1*pow(result,-2./3.);
    }
    else                        /* use high energy approximation */
    {
      result=xlow-log(*RanFlatVecPtr(1));
      exact=SynRadC(result);
      appr=a2*exp(-result);
    }
  }
  while(exact<appr*(*RanFlatVecPtr(1))); /* reject in proportion of approx */
  return result;                         /* result now exact spectrum with unity weight */
}

/*==================================================================*/

/*	returns function value SynRadC   photon spectrum dn/dx
 * (integral of modified 1/3 order Bessel function)
 * principal: Chebyshev series see H.H.Umstaetter CERN/PS/SM/81-13 10-3-1981
 * see also my LEP Note 632 of 12/1990
 * converted to C++, H.Burkhardt 21-4-1996    */

double SynRadC(double x)
{
  double synrad=0.;
  if((x>0.)&&(x<800.)) { /* otherwise result synrad remains 0 */
    if(x<6.) {
      double a,b,z;
      double p;
      double q;
      double y;
      double const twothird=2./3.;
      z=x*x/16.-2.;
      b=          .00000000000000000012;
      a=z*b  +    .00000000000000000460;
      b=z*a-b+    .00000000000000031738;
      a=z*b-a+    .00000000000002004426;
      b=z*a-b+    .00000000000111455474;
      a=z*b-a+    .00000000005407460944;
      b=z*a-b+    .00000000226722011790;
      a=z*b-a+    .00000008125130371644;
      b=z*a-b+    .00000245751373955212;
      a=z*b-a+    .00006181256113829740;
      b=z*a-b+    .00127066381953661690;
      a=z*b-a+    .02091216799114667278;
      b=z*a-b+    .26880346058164526514;
      a=z*b-a+   2.61902183794862213818;
      b=z*a-b+  18.65250896865416256398;
      a=z*b-a+  92.95232665922707542088;
      b=z*a-b+ 308.15919413131586030542;
      a=z*b-a+ 644.86979658236221700714;
      p=.5*z*a-b+  414.56543648832546975110;
      a=          .00000000000000000004;
      b=z*a+      .00000000000000000289;
      a=z*b-a+    .00000000000000019786;
      b=z*a-b+    .00000000000001196168;
      a=z*b-a+    .00000000000063427729;
      b=z*a-b+    .00000000002923635681;
      a=z*b-a+    .00000000115951672806;
      b=z*a-b+    .00000003910314748244;
      a=z*b-a+    .00000110599584794379;
      b=z*a-b+    .00002581451439721298;
      a=z*b-a+    .00048768692916240683;
      b=z*a-b+    .00728456195503504923;
      a=z*b-a+    .08357935463720537773;
      b=z*a-b+    .71031361199218887514;
      a=z*b-a+   4.26780261265492264837;
      b=z*a-b+  17.05540785795221885751;
      a=z*b-a+  41.83903486779678800040;
      q=.5*z*a-b+28.41787374362784178164;
      y=pow(x,twothird);
      synrad=(p/y-q*y-1.)*1.81379936423421784215530788143;
    }
    else {                  /* 6 < x < 174 */
      double a,b,z;
      double p;
      double const pihalf=PI/2.;
      z=20./x-2.;
      a=      .00000000000000000001;
      b=z*a  -.00000000000000000002;
      a=z*b-a+.00000000000000000006;
      b=z*a-b-.00000000000000000020;
      a=z*b-a+.00000000000000000066;
      b=z*a-b-.00000000000000000216;
      a=z*b-a+.00000000000000000721;
      b=z*a-b-.00000000000000002443;
      a=z*b-a+.00000000000000008441;
      b=z*a-b-.00000000000000029752;
      a=z*b-a+.00000000000000107116;
      b=z*a-b-.00000000000000394564;
      a=z*b-a+.00000000000001489474;
      b=z*a-b-.00000000000005773537;
      a=z*b-a+.00000000000023030657;
      b=z*a-b-.00000000000094784973;
      a=z*b-a+.00000000000403683207;
      b=z*a-b-.00000000001785432348;
      a=z*b-a+.00000000008235329314;
      b=z*a-b-.00000000039817923621;
      a=z*b-a+.00000000203088939238;
      b=z*a-b-.00000001101482369622;
      a=z*b-a+.00000006418902302372;
      b=z*a-b-.00000040756144386809;
      a=z*b-a+.00000287536465397527;
      b=z*a-b-.00002321251614543524;
      a=z*b-a+.00022505317277986004;
      b=z*a-b-.00287636803664026799;
      a=z*b-a+.06239591359332750793;
      p=.5*z*a-b    +1.06552390798340693166;
      synrad=p*sqrt(pihalf/x)/exp(x);
    }
  }
  return synrad;
}

/*==================================================================*/

/* Compute momentum loss for particle passing through a magnetic
 * field, taking into account desired options etc. For now, this
 * function takes the ultra-relativistic limit where dP = dE. */

/* RET:    momentum change in GeV/c (note positive dP = energy loss!).
 * ABORT:  never.
 * FAIL:   if P or L <= 0.  Calling routine is responsible for making
 * sure this never happens. */
#ifdef __CUDACC__
__device__ double ComputeSRMomentumLoss_gpu( double P, double BL, double L, int MethodFlag, curandState_t *rState )
{
  double uc, npmean, du ;
  int nphot, ncount ;
  
  /* initialize the normalized energy loss to zero */
  
  du = 0 ;
  npmean = 0 ;
  
  /* calculate the critical energy and the number of photons */
  
  CalculateSRPars( P, BL, L, &uc, &npmean, NULL, NULL ) ;
  
  /* Tracking flag set to zero:  no SR simulation */
  
  if ( MethodFlag == SR_None )
    goto egress ;
  
  /* otherwise get some parameters */
  
  if ( MethodFlag == SR_Mean )
  {
    du = 0.307933 * npmean ;
    goto egress ;
  }
  
  nphot = poidev_gpu(npmean, rState) ;
  for (ncount=0 ; ncount<nphot ; ncount++)
  {
    if (MethodFlag == SR_AW)
      du += SRSpectrumAW_gpu( rState ) ;
    else
      du += SRSpectrumHB_gpu( rState ) ;
  }
  
  /* remove uc normalization */
  
  egress:
    
    return du * uc ;
    
}
#endif
double ComputeSRMomentumLoss( double P, double BL, double L, int MethodFlag )
{
  double uc, npmean, du ;
  int nphot, ncount ;
  
  /* initialize the normalized energy loss to zero */
  
  du = 0 ;
  npmean = 0 ;
  
  /* calculate the critical energy and the number of photons */
  
  CalculateSRPars( P, BL, L, &uc, &npmean, NULL, NULL ) ;
  
  /* Tracking flag set to zero:  no SR simulation */
  
  if ( MethodFlag == SR_None )
    goto egress ;
  
  /* otherwise get some parameters */
  
  if ( MethodFlag == SR_Mean )
  {
    du = 0.307933 * npmean ;
    goto egress ;
  }
  
  nphot = poidev(npmean) ;
  for (ncount=0 ; ncount<nphot ; ncount++)
  {
    if (MethodFlag == SR_AW)
      du += SRSpectrumAW( ) ;
    else
      du += SRSpectrumHB( ) ;
  }
  
  /* remove uc normalization */
  
  egress:
    
    return du * uc ;
    
}

/*==================================================================*/

/* Compute the transfer map for a coordinate change.  Note that the
 * coordinate change involves both zeroth-order and linear terms,
 * which this function returns. Based heavily on the MAD8 coordinate
 * transformation routine trdsp1.
 *
 * Ret:   6-vector of coordinate changes and R-matrix.  Also, an
 * integer status which indicates failure if a nonphysical
 * coordinate rotation is requested.
 * Abort: never.
 * Fail:  never.  */

int GetCoordMap( double offset[6], double dx[6], Rmat R )
{
  double w11, w12, w13 ;
  double w21, w22, w23 ;
  double w31, w32, w33 ;
  double s2, w33i ;
  double costheta, sintheta ;
  double cosphi,  sinphi    ;
  double cospsi,  sinpsi    ;
  int status = 1 ;
  
  /* compute the trig functions */
  
  costheta = cos(offset[1]) ;
  sintheta = sin(offset[1]) ;
  cosphi   = cos(offset[3]) ;
  sinphi   = sin(offset[3]) ;
  cospsi   = cos(offset[5]) ;
  sinpsi   = sin(offset[5]) ;
  
  /* compute the w-matrix terms */
  
  w11 =  costheta * cospsi - sintheta * sinphi * sinpsi ;
  w12 = -costheta * sinpsi - sintheta * sinphi * cospsi ;
  w13 =  sintheta * cosphi ;
  
  w21 = cosphi * sinpsi ;
  w22 = cosphi * cospsi ;
  w23 = sinphi ;
  
  w31 = -sintheta * cospsi - costheta * sinphi * sinpsi ;
  w32 =  sintheta * sinpsi - costheta * sinphi * cospsi ;
  w33 =  costheta * cosphi ;
  
  /* physically unrealizable transformation is indicated by w33
   * less than or equal to zero */
  
  if (w33 <= 0.)
  {
    status = 0 ;
    goto egress ;
  }
  w33i = 1./w33 ;
  
  s2 = w13 * offset[0] + w23 * offset[2] + w33 * offset[4] ;
  s2 *= w33i ;
  
  /* construct the R-matrix of the rotation */
  
  R[0][0] = w22*w33i ;
  R[0][1] = R[0][0]*s2 ;
  R[0][2] = -w12*w33i ;
  R[0][3] = R[0][2]*s2 ;
  R[0][4] = 0 ;
  R[0][5] = 0 ;
  
  R[1][0] = 0 ;
  R[1][1] = w11 ;
  R[1][2] = 0 ;
  R[1][3] = w21 ;
  R[1][4] = 0 ;
  R[1][5] = 0 ;
  
  R[2][0] = -w21*w33i ;
  R[2][1] = R[2][0]*s2 ;
  R[2][2] = w11*w33i ;
  R[2][3] = R[2][2]*s2 ;
  R[2][4] = 0 ;
  R[2][5] = 0 ;
  
  R[3][0] = 0 ;
  R[3][1] = w12 ;
  R[3][2] = 0 ;
  R[3][3] = w22 ;
  R[3][4] = 0 ;
  R[3][5] = 0 ;
  
  R[4][0] = -w13*w33i ;
  R[4][1] = R[4][0]*s2 ;
  R[4][2] = -w23*w33i ;
  R[4][3] = R[4][2]*s2 ;
  R[4][4] = 1 ;
  R[4][5] = 0 ;
  
  R[5][0] = 0 ;
  R[5][1] = 0 ;
  R[5][2] = 0 ;
  R[5][3] = 0 ;
  R[5][4] = 0 ;
  R[5][5] = 1 ;
  
  /* construct the dx vector */
  
  dx[0] = -offset[0] ;
  dx[1] = w31 ;
  dx[2] = -offset[2] ;
  dx[3] = w32 ;
  dx[4] = -offset[4] ;
  dx[5] = 0 ;
  
  /* set status and return */
  
  egress:
    
    return status ;
    
}

/*==================================================================*/

/* Get RMS of required beam dimension
 *
 * Ret:   RMS quantity
 * Abort: never.
 * Fail:  never.  */

double GetRMSCoord( struct Bunch* ThisBunch, int dim )
{
  int ic ;
  double sumdim=0 ;
  double sumq=0 ;
  double avedim ;
  for ( ic=0; ic<ThisBunch->nray; ic++ ) {
    if (ThisBunch->stop[ic] == 0) {
      sumdim += ThisBunch->x[6*ic+dim] * ThisBunch->Q[ic] ;
      sumq += ThisBunch->Q[ic] ;
    }
  }
  avedim = sumdim / sumq  ;
  sumdim=0;
  sumq=0;
  for ( ic=0; ic<ThisBunch->nray; ic++ ) {
    sumdim += pow((ThisBunch->x[6*ic+dim] - avedim),2) * ThisBunch->Q[ic] ;
    sumq += ThisBunch->Q[ic] ;
  }
  return sumdim / sumq ;
}

/*==================================================================*/

/* Get Mean of required beam dimension
 *
 * Ret:   RMS quantity
 * Abort: never.
 * Fail:  never.  */

double GetMeanCoord( struct Bunch* ThisBunch, int dim )
{
  int ic ;
  double sumdim=0 ;
  double sumq=0 ;
  for ( ic=0; ic<ThisBunch->nray; ic++ ) {
    if (ThisBunch->stop[ic]==0) {
      sumdim += ThisBunch->x[6*ic+dim] * ThisBunch->Q[ic] ;
      sumq += ThisBunch->Q[ic] ;
    }
  }
  return sumdim / sumq ;
}
