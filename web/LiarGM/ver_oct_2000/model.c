/* model.f -- translated by f2c (version 19980516).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"

/* Common Block Declarations */

struct {
    doublereal ss[300], xx[300], yy[300];
    integer nelem;
} line_;

#define line_1 line_

struct {
    integer idum;
} randum_;

#define randum_1 randum_

struct {
    doublereal told;
} timeold_;

#define timeold_1 timeold_

struct {
    doublereal am[62500]	/* was [250][250] */, wh[250], kh[250], sw[
	    250], cw[250], dskx[62500]	/* was [250][250] */, dckx[62500]	
	    /* was [250][250] */, dsky[62500]	/* was [250][250] */, dcky[
	    62500]	/* was [250][250] */;
} harmonics_;

#define harmonics_1 harmonics_

struct {
    doublereal ams[250], khs[250], dskxs[250], dckxs[250], dskys[250], dckys[
	    250];
} harmonics_syst__;

#define harmonics_syst__1 harmonics_syst__

struct {
    doublereal a, b, f1, a1, d1, v1, f2, a2, d2, v2, f3, a3, d3, v3, tmax, 
	    tmin, smax, smin;
    integer np;
    doublereal q1, rk1, rkk1;
} earth_;

#define earth_1 earth_

struct {
    doublereal g, kmin, kmax, wmin, wmax;
    integer nk, nw;
    doublereal kmins, kmaxs;
} integ_;

#define integ_1 integ_

struct {
    doublereal difftmax;
} maxtimediff_;

#define maxtimediff_1 maxtimediff_

struct {
    integer inewparam, inewparams;
} filejustread_;

#define filejustread_1 filejustread_

/* Table of constant values */

static integer c__9 = 9;
static integer c__1 = 1;
static integer c__3 = 3;
static integer c__5 = 5;
static doublereal c_b491 = 21.143304620203601;
static doublereal c_b492 = 2.7e4;

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Main program */ MAIN__()
{
    /* Builtin functions */
    integer s_wsle(), do_lio(), e_wsle(), s_rsle(), e_rsle();
    /* Subroutine */ int s_stop();

    /* Local variables */
    extern /* Subroutine */ int prepare_systhar__();
    static doublereal time;
    extern /* Subroutine */ int misalign_();
    static doublereal dummy;
    extern /* Subroutine */ int tests_(), prepare_harmonics__(), 
	    read_pwk_param__(), read_positions__();
    extern doublereal ran1_();

    /* Fortran I/O blocks */
    static cilist io___1 = { 0, 6, 0, 0, 0 };
    static cilist io___2 = { 0, 5, 0, 0, 0 };
    static cilist io___4 = { 0, 6, 0, 0, 0 };
    static cilist io___5 = { 0, 6, 0, 0, 0 };
    static cilist io___6 = { 0, 6, 0, 0, 0 };
    static cilist io___7 = { 0, 6, 0, 0, 0 };
    static cilist io___8 = { 0, 6, 0, 0, 0 };


/* ----------------------------------------------------------------------- */
/*                                  MODEL */
/*                         version October 1999 */
/*                       Seryi@SLAC.Stanford.EDU */
/* ----------------------------------------------------------------------- */

/*  conversion to C:   f2c -r8 model.f    (-r8 promote real to double) */

/* ----------------------------------------------------------------------- */
/*        Computes horizontal x(t,s) and vertical y(t,s) */
/*        position of the ground at a given time t and in a given */
/*        longitudinal position s, assuming that at time t=0 we had */
/*        x(0,s)=0 and y(0,s)=0.  The values x(t,s) and y(t,s) will */
/*        be computed using the same power spectrum P(w,k), however */
/*        they are independent. Parameters of approximation of */
/*        the P(w,k) (PWK) can be chosen to model quiet or noisy */
/*        place. */
/*        Units are seconds and meters. */

/* ----------------------------------------------------------------------- */
/*                   The program needs the next files */
/*       "model.data"  (if it does not exist, it will create it) */
/*        for explanations of parameters see Report DAPNIA/SEA 95-04 */
/*        (should also appear in Phys.Rew.E, in 1 April 1997 issue) */

/*        Also needs one column file "positions.data" that contains */
/*        longitudinal positions s of points where we will want to */
/*        find x(t,s) and y(t,s). Number of lines in this file is */
/*        therefore the numbers of elements. In FORTRAN version */
/*        this number is limited. */

/* ----------------------------------------------------------------------- */
/*                      How it works. */
/*      We assumed that we know power spectrum of ground motion P(w,k). */
/*      Then we assume that using only finite number of */
/*      harmonics we can model ground motion in some limited range of */
/*      t and s. As one of inputs we have these ranges: minimum and */
/*      maximum time Tmin and Tmax and minimum and maximum distance */
/*      Smin and Smax. We will define then the range of important */
/*      frequencies wmin to wmax and wave-numbers kmin to kmax. */
/*      Our harmonics we will distribute equidistantly in logariphmic */
/*      sense, that is, for example, k_{i+1}/k_{i} is fixed. */
/*      Number of steps defined Np from the input file, for */
/*      example Np=50. A single harmonic characterized by */
/*      its amplitude am_{ij}, frequency w_{i}, wave number k_{j} */
/*      and phase phi_{ij}. Total number of harmonics is Np*Np. */
/*      The amplitudes of harmonics are defined once at the beginning */
/*      from integral over surface (w_{i+1}:w_{i})(k_{j+1}:k_{j}) */
/*      on P(w,k). Phases phi_{ij} are also defined once at the */
/*      beginning by random choice. The resulting x(t,s) will be */
/*      given by double sums: */
/*      x(t,s) = 0.5 sum_{i}^{Np} sum_{j}^{Np} am_{ij} * sin(w_{i} t) */
/*                                                * sin(k_{j} s + phi_{ij}) */
/*             + 0.5 sum_{i}^{Np} sum_{j}^{Np} am_{ij} * (cos(w_{i} t)-1) */
/*                                                * sin(k_{j} s + psi_{ij}) */
/*      This choise of formula ensure x(t,s) = 0 at t=0. */
/*      The last sinus is presented in the program in other form */
/*      sin(k_{j} s + phi_{ij}) = sin(k_{j} s) cos(phi_{ij}) */
/*                                     + cos(k_{j} s) sin(phi_{ij}) */
/*      So, we store not phases phi_{ij}, but cos(phi_{ij}) and */
/*      sin(phi_{ij}) */
/*      The same for y(t,s) but with different phases phi_{ij}. */

/* ----------------------------------------------------------------------- */
/*                   Changes since the previous versions */
/* 12.07.96 */
/*  1. A bug fixed in XY_PWK. We should use both sin(wt) and cos(wt)-1 */
/*     and not only sin(wt) as it was before */

/* 26.06.96 */
/*  1. A bug is fixed in the PWK subroutine. The variable Lo was */
/*     forgotten to declare real. */

/* 12.03.96 */
/*  1. Calculation of sinus or cosine function from big */
/*          arguments has been changed, presicion loss errors */
/*          should not happen any more */
/*  2. Many comments are added (maybe still not enough) */
/*  3. Harmonics generation has been changed to increase speed */
/*     If parameters of P(w,k) are the same, then at second and others */
/*     calls of PREPARE_HARMONICS subroutine only phases will */
/*     be refreshed (amplitudes will be the same, no need to integrate */
/*     P(w,k) again) */

/* Oct 99 */
/*  1. Would like to implement systematic ground motion as */
/*     "x(t,s)" = "x given by P(w,k)" + time* "given by Q(k)" */
/*     where Q(k) is the power spectrum of the systematic */
/*     comnponent */
/* ----------------------------------------------------------------------- */
/* What is not done yet: */
/* Number of steps of integration in PREPARE_HARMONICS */
/* is not optimized */
/* ----------------------------------------------------------------------- */
/* Maximum number of elements is limited by NELMX. */
/* This defect can be avoided in C version. */

/* arrays for longitudinal, horizontal and vertical position, and */
/* number of elements */

/* if random generator called with negative idum, the generator */
/* will be initialized (idum then should not be touched) */
/* this initialization is not necessary, in fact, but we put */
/* it to have possibility to get different seeds from the start */
/* It is maybe better to read idum from input file? */

    s_wsle(&io___1);
    do_lio(&c__9, &c__1, "Input positive integer to make seed=", (ftnlen)36);
    e_wsle();
    s_rsle(&io___2);
    do_lio(&c__3, &c__1, (char *)&randum_1.idum, (ftnlen)sizeof(integer));
    e_rsle();
/*      idum= -7987413 */
    randum_1.idum = -randum_1.idum;
    dummy = ran1_(&randum_1.idum);
/* read parameters of PWK */
    read_pwk_param__();
    s_wsle(&io___4);
    do_lio(&c__9, &c__1, "   ", (ftnlen)3);
    e_wsle();
    s_wsle(&io___5);
    do_lio(&c__9, &c__1, " We still print some information to standard output"
	    , (ftnlen)51);
    e_wsle();
    s_wsle(&io___6);
    do_lio(&c__9, &c__1, " it can be suppressed when not needed ", (ftnlen)38)
	    ;
    e_wsle();
/* read longitudinal position of elements and count their number Nelem */
    read_positions__();
/* calculate frequencies, wave numbers, amplitudes and phases of harmonics */
    prepare_harmonics__();
    s_wsle(&io___7);
    do_lio(&c__9, &c__1, "before syst prep harm", (ftnlen)21);
    e_wsle();
    prepare_systhar__();
    s_wsle(&io___8);
    do_lio(&c__9, &c__1, "after syst prep", (ftnlen)15);
    e_wsle();
/* the main call to calculate position of elements at a given time */

    misalign_(&time);
/*     start of tests, should be commented when not needed */
/*     (consume processor time and also changes arrays in the */
/*     common/line/, namely it put Nelem=4 and array of position */
/*     as ss/0. , Smin , sqrt(Smax*Smin) , Smax/ */

    tests_();
/*      call TESTS2 */
/*      call TESTS3 */
    s_stop("", (ftnlen)0);
} /* MAIN__ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int misalign_(t)
doublereal *t;
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer n;
    static doublereal s, x, y;
    extern /* Subroutine */ int xy_pwk__();

/* calculates x and y positions of all Nelem elements at */
/* a given time t */
    i__1 = line_1.nelem;
    for (n = 1; n <= i__1; ++n) {
	s = line_1.ss[n - 1];
	xy_pwk__(t, &s, &x, &y);
	line_1.xx[n - 1] = x;
	line_1.yy[n - 1] = y;
    }
    return 0;
} /* misalign_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int read_positions__()
{
    /* System generated locals */
    integer i__1;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer f_open(), s_rsle(), do_lio(), e_rsle(), s_wsle(), e_wsle(), 
	    f_clos();

    /* Local variables */
    static integer i__;

    /* Fortran I/O blocks */
    static cilist io___15 = { 0, 92, 1, 0, 0 };
    static cilist io___16 = { 0, 6, 0, 0, 0 };


/* reads positions of elements from the file positions.data, */
/* counts total number of elements */
    o__1.oerr = 1;
    o__1.ounit = 92;
    o__1.ofnmlen = 14;
    o__1.ofnm = "positions.data";
    o__1.orl = 0;
    o__1.osta = "old";
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    i__1 = f_open(&o__1);
    if (i__1 != 0) {
	goto L999;
    }
    i__ = 1;
L1:
    i__1 = s_rsle(&io___15);
    if (i__1 != 0) {
	goto L900;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&line_1.ss[i__ - 1], (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L900;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L900;
    }
    ++i__;
    if (i__ > 300) {
	goto L999;
    }
    goto L1;
L900:
    line_1.nelem = i__ - 1;
    goto L100;
L999:
    s_wsle(&io___16);
    do_lio(&c__9, &c__1, " Error open \"positions.data\" or too many elements"
	    , (ftnlen)49);
    e_wsle();
L100:
    cl__1.cerr = 0;
    cl__1.cunit = 92;
    cl__1.csta = 0;
    f_clos(&cl__1);
    return 0;
} /* read_positions__ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int xy_pwk__(t, s, x, y)
doublereal *t, *s, *x, *y;
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double sin(), cos();

    /* Local variables */
    static doublereal qcos, qsin;
    static integer i__, j;
    static doublereal qdcos, qdsin, sinkx, sinky, ck[250], sk[250];

/* it computes x and y positions of a single element that has */
/* longitudinal coordinate s, at a given time t (positive) */

/* arrays of: amplitude  am, frequency (omega) wh, wave number kh */
/* arrays to store values sin(w_{i} t) and cos(w_{i} t) */
/* we will use only sinus, but cosine we need to calculate sinus */
/* for the new time t using values saved at time told */


/* told is the time t of previous use of this subroutine */
    if (*t != timeold_1.told) {

/* we will calculate sin(w_{i} t) only if time has been changed since */
/* previous call, otherwise will used stored in the array sw values. */
/* it increases speed because this subroutine called Nelem times */
/* with the same t */

	i__1 = earth_1.np;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    qdsin = sin((*t - timeold_1.told) * harmonics_1.wh[i__ - 1]);
	    qdcos = cos((*t - timeold_1.told) * harmonics_1.wh[i__ - 1]);
	    qsin = harmonics_1.sw[i__ - 1] * qdcos + harmonics_1.cw[i__ - 1] *
		     qdsin;
	    qcos = harmonics_1.cw[i__ - 1] * qdcos - harmonics_1.sw[i__ - 1] *
		     qdsin;
	    harmonics_1.sw[i__ - 1] = qsin;
	    harmonics_1.cw[i__ - 1] = qcos;
	}
    }
    timeold_1.told = *t;
/* we calculate sin(k_{j} s) at each step. This is stupid and can */
/* be avoided for the cost of two arrays (NH*NELMX). But this array can be */
/* very big, in our cases (50*300) = (15000) */
/* What is better? */

    i__1 = earth_1.np;
    for (j = 1; j <= i__1; ++j) {
	sk[j - 1] = sin(*s * harmonics_1.kh[j - 1]);
	ck[j - 1] = cos(*s * harmonics_1.kh[j - 1]);
    }
/* clear variables, start of double sums */
    *x = 0.;
    *y = 0.;
    i__1 = earth_1.np;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = earth_1.np;
	for (j = 1; j <= i__2; ++j) {
	    sinkx = sk[j - 1] * harmonics_1.dckx[i__ + j * 250 - 251] + ck[j 
		    - 1] * harmonics_1.dskx[i__ + j * 250 - 251];
	    sinky = sk[j - 1] * harmonics_1.dcky[i__ + j * 250 - 251] + ck[j 
		    - 1] * harmonics_1.dsky[i__ + j * 250 - 251];
/*           x=x + am(i,j) * sw(i) * sinkx  ! this was a bug, fixed 12.07.96 */
/*           y=y + am(i,j) * sw(i) * sinky */
	    *x += harmonics_1.am[i__ + j * 250 - 251] * (harmonics_1.sw[i__ - 
		    1] * sinkx + (harmonics_1.cw[i__ - 1] - 1.) * sinky) * .5;
	    *y += harmonics_1.am[i__ + j * 250 - 251] * (harmonics_1.sw[i__ - 
		    1] * sinky + (harmonics_1.cw[i__ - 1] - 1.) * sinkx) * .5;
	}
    }
/* add systematic components */
/* we calculate sin(k_{j} s) at each step. This is stupid and can */
/* be avoided for the cost of two arrays (NH*NELMX). But this array can be */
/* very big, in our cases (50*300) = (15000) */
/* What is better? */

    i__1 = earth_1.np;
    for (j = 1; j <= i__1; ++j) {
	sk[j - 1] = sin(*s * harmonics_syst__1.khs[j - 1]);
	ck[j - 1] = cos(*s * harmonics_syst__1.khs[j - 1]);
    }
    i__1 = earth_1.np;
    for (j = 1; j <= i__1; ++j) {
	sinkx = sk[j - 1] * harmonics_syst__1.dckxs[j - 1] + ck[j - 1] * 
		harmonics_syst__1.dskxs[j - 1];
	sinky = sk[j - 1] * harmonics_syst__1.dckys[j - 1] + ck[j - 1] * 
		harmonics_syst__1.dskys[j - 1];
/*           x=x + am(i,j) * sw(i) * sinkx  ! this was a bug, fixed 12.07.96 */
/*           y=y + am(i,j) * sw(i) * sinky */
	*x += harmonics_syst__1.ams[j - 1] * *t * sinkx;
	*y += harmonics_syst__1.ams[j - 1] * *t * sinky;
    }
    return 0;
} /* xy_pwk__ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int prepare_harmonics__()
{
    /* Initialized data */

    static integer ndiv = 30;
    static doublereal pi = 3.14159265358979;

    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    integer s_wsle(), do_lio(), e_wsle();
    double pow_dd(), pow_di(), sqrt(), sin(), cos();

    /* Local variables */
    static doublereal sfin;
    extern /* Subroutine */ int ppwk_();
    static doublereal kmin1[4]	/* was [2][2] */, kmax1[4]	/* was [2][2] 
	    */, wmin1[4]	/* was [2][2] */, wmax1[4]	/* was [2][2] 
	    */;
    static integer i__, j;
    static doublereal k, s, w, phase, a0, stest[2], ttest[2], s1, ka, kb, dk;
    static integer ii, jj;
    static doublereal ds, wa, dw;
    static integer is, it;
    static doublereal wb, s1k, s1w, dkk, dww;
    extern doublereal pwk_();
    static doublereal rrr;
    extern doublereal ran1_();

    /* Fortran I/O blocks */
    static cilist io___29 = { 0, 6, 0, 0, 0 };
    static cilist io___30 = { 0, 6, 0, 0, 0 };
    static cilist io___51 = { 0, 6, 0, 0, 0 };
    static cilist io___52 = { 0, 6, 0, 0, 0 };
    static cilist io___53 = { 0, 6, 0, 0, 0 };
    static cilist io___54 = { 0, 6, 0, 0, 0 };
    static cilist io___55 = { 0, 6, 0, 0, 0 };
    static cilist io___56 = { 0, 6, 0, 0, 0 };
    static cilist io___57 = { 0, 6, 0, 0, 0 };
    static cilist io___58 = { 0, 6, 0, 0, 0 };
    static cilist io___59 = { 0, 6, 0, 0, 0 };
    static cilist io___60 = { 0, 6, 0, 0, 0 };
    static cilist io___71 = { 0, 6, 0, 0, 0 };
    static cilist io___72 = { 0, 6, 0, 0, 0 };
    static cilist io___73 = { 0, 6, 0, 0, 0 };


/* calculate frequencies, wave numbers, amplitudes and phases of harmonics */

/* Maximum number of harmonics is limited by NH*NH */
/* inewparam =0 will say to PREPARE_HARMONICS that the file */
/* with P(w,k) parameters has been just read thus complete */
/* harmonics generation should be performed */
/* if inewparam =1 it means that the harmonics has been already */
/* once calculated, this is just new seed and it is enough to */
/* generate only new phases, w and k */

/* will just regenerate phases, w and k if inewparam =1 */
    if (filejustread_1.inewparam == 1) {
	goto L1000;
    }
/* test that requested number of harmonics is smaller than array size */
    if (earth_1.np > 250) {
	earth_1.np = 250;
    }
    s_wsle(&io___29);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___30);
    do_lio(&c__9, &c__1, " Finding range of important w and k", (ftnlen)35);
    e_wsle();
/* we will estimate range of important w and k by rough */
/* preliminary integration */
/* of P(w,k) over enough big (hopefully) range of w and k */
    integ_1.nw = earth_1.np;
    integ_1.nk = earth_1.np;
/* we define this wide initial range of w and k and beleive that */
/* the important range is inside */

    integ_1.wmin = 1. / earth_1.tmax / 1e3;
    integ_1.wmax = 1. / earth_1.tmin * 1e3;
    integ_1.kmin = 1e-5;
    integ_1.kmax = 1e5;
/* ratio k_{i+1}/k_{i} is constant */

    d__1 = integ_1.kmax / integ_1.kmin;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
    d__1 = integ_1.wmax / integ_1.wmin;
    d__2 = 1. / (integ_1.nw - 1);
    dw = pow_dd(&d__1, &d__2);
/* to estimate range of important w and k we will calculate four */
/* values, namely the mean value < [x(t,s1)-x(t,s2)]^2 > for different */
/* combination of variables, namely: */
/* t=Tmin, s1-s2=Smin */
/* t=Tmin, s1-s2=Smax */
/* t=Tmax, s1-s2=Smin */
/* t=Tmax, s1-s2=Smax */
    ttest[0] = earth_1.tmin;
    ttest[1] = earth_1.tmax;
    stest[0] = earth_1.smin;
    stest[1] = earth_1.smax;
/* double loop to check all these four cases */
    for (is = 1; is <= 2; ++is) {
	for (it = 1; it <= 2; ++it) {
/* the value < [x(t,s1)-x(t,s2)]^2 > is the double integral on */
/* P(w,k) 2 (1-cos(wt)) 2 (1-cos(k (s1-s2))) dw/(2pi) dk/(2pi) = F(w,k) */

/* to find the important range of w we calculate first the value */
/* sfin = Int^{wmax}_{wmin} Int^{kmax}_{kmin} F(w,k) dw dk */
/* and then we calculate first the function */
/* s1(w)= Int^{w}_{wmin} Int^{kmax}_{kmin} F(w,k) dw dk */
/* then the ratio s1(w)/sfin will be equal 0 at wmin and 1 at wmax */
/* the region where this function changes rapidly from 0 to 1 */
/* gives main contribution to the integral. */
/* we define the range of important w as the points where */
/* s1(w)/sfin cross the level 0.01 and 0.99 for wmin and wmax */
/* correspondingly */

/* to find the range of k we do the same but with s2(k)/sfin where */
/* s2(k)= Int^{k}_{kmin} Int^{wmax}_{wmin} F(w,k) dw dk */

	    sfin = 0.;
	    i__1 = integ_1.nw;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__ - 1;
		w = integ_1.wmin * pow_di(&dw, &i__2);
		i__2 = integ_1.nk;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j - 1;
		    k = integ_1.kmin * pow_di(&dk, &i__3);
		    ds = (w * dw - w) * (k * dk - k);
		    ppwk_(&stest[is - 1], &ttest[it - 1], &k, &w, &rrr);
		    sfin += ds * rrr;
		}
	    }
	    if (sfin == 0.) {
		goto L500;
	    }
	    wmin1[is + (it << 1) - 3] = 0.;
	    wmax1[is + (it << 1) - 3] = 0.;
	    s1 = 0.;
	    i__1 = integ_1.nw;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__ - 1;
		w = integ_1.wmin * pow_di(&dw, &i__2);
		i__2 = integ_1.nk;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j - 1;
		    k = integ_1.kmin * pow_di(&dk, &i__3);
		    ds = (w * dw - w) * (k * dk - k);
		    ppwk_(&stest[is - 1], &ttest[it - 1], &k, &w, &rrr);
		    s1 += ds * rrr;
		}
		s1w = s1 / sfin;
		if (wmin1[is + (it << 1) - 3] == 0. && s1w > .01) {
		    wmin1[is + (it << 1) - 3] = w / dw;
		}
		if (wmax1[is + (it << 1) - 3] == 0. && s1w > .99) {
		    wmax1[is + (it << 1) - 3] = w * dw;
		}
	    }
	    kmin1[is + (it << 1) - 3] = 0.;
	    kmax1[is + (it << 1) - 3] = 0.;
	    s1 = 0.;
	    i__1 = integ_1.nk;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__ - 1;
		k = integ_1.kmin * pow_di(&dk, &i__2);
		i__2 = integ_1.nw;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j - 1;
		    w = integ_1.wmin * pow_di(&dw, &i__3);
		    ds = (w * dw - w) * (k * dk - k);
		    ppwk_(&stest[is - 1], &ttest[it - 1], &k, &w, &rrr);
		    s1 += ds * rrr;
		}
		s1k = s1 / sfin;
		if (kmin1[is + (it << 1) - 3] == 0. && s1k > .01) {
		    kmin1[is + (it << 1) - 3] = k / dk;
		}
		if (kmax1[is + (it << 1) - 3] == 0. && s1k > .99) {
		    kmax1[is + (it << 1) - 3] = k * dk;
		}
	    }
	}
    }
/* we have found the important ranges for all of four cases, */
/* now we find the range that cover these four */

    integ_1.kmin = kmin1[0];
    integ_1.kmax = kmax1[0];
    integ_1.wmin = wmin1[0];
    integ_1.wmax = wmax1[0];
    for (is = 1; is <= 2; ++is) {
	for (it = 1; it <= 2; ++it) {
/* Computing MIN */
	    d__1 = kmin1[is + (it << 1) - 3];
	    integ_1.kmin = min(d__1,integ_1.kmin);
/* Computing MIN */
	    d__1 = wmin1[is + (it << 1) - 3];
	    integ_1.wmin = min(d__1,integ_1.wmin);
/* Computing MAX */
	    d__1 = kmax1[is + (it << 1) - 3];
	    integ_1.kmax = max(d__1,integ_1.kmax);
/* Computing MAX */
	    d__1 = wmax1[is + (it << 1) - 3];
	    integ_1.wmax = max(d__1,integ_1.wmax);
	}
    }
L500:
    d__1 = integ_1.kmax / integ_1.kmin;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
    d__1 = integ_1.wmax / integ_1.wmin;
    d__2 = 1. / (integ_1.nw - 1);
    dw = pow_dd(&d__1, &d__2);
    integ_1.wmax /= dw;
    integ_1.kmax /= dk;
    s_wsle(&io___51);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___52);
    do_lio(&c__9, &c__1, " Range of important k and w:", (ftnlen)28);
    e_wsle();
    s_wsle(&io___53);
    do_lio(&c__9, &c__1, " k from ", (ftnlen)8);
    do_lio(&c__5, &c__1, (char *)&integ_1.kmin, (ftnlen)sizeof(doublereal));
    do_lio(&c__9, &c__1, " to ", (ftnlen)4);
    do_lio(&c__5, &c__1, (char *)&integ_1.kmax, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___54);
    do_lio(&c__9, &c__1, " w from ", (ftnlen)8);
    do_lio(&c__5, &c__1, (char *)&integ_1.wmin, (ftnlen)sizeof(doublereal));
    do_lio(&c__9, &c__1, " to ", (ftnlen)4);
    do_lio(&c__5, &c__1, (char *)&integ_1.wmax, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___55);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
/* the range of important k and w has been found */
/* now we start to find amplitude of each harmonic by */
/* integration of P(w,k) */
    d__1 = integ_1.kmax / integ_1.kmin;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
    d__1 = integ_1.wmax / integ_1.wmin;
    d__2 = 1. / (integ_1.nw - 1);
    dw = pow_dd(&d__1, &d__2);
/* estimate maximum value of t-told for which the subroutine */
/* XY_PWK will still give correct values and PLOSS error will */
/* not happen */

    maxtimediff_1.difftmax = 1e8 / integ_1.wmax / dw;
    s_wsle(&io___56);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___57);
    do_lio(&c__9, &c__1, " The maximum allowable time difference t-told ", (
	    ftnlen)46);
    e_wsle();
    s_wsle(&io___58);
    do_lio(&c__9, &c__1, " for calls to XY_PWK subroutine is about ", (ftnlen)
	    41);
    do_lio(&c__5, &c__1, (char *)&maxtimediff_1.difftmax, (ftnlen)sizeof(
	    doublereal));
    e_wsle();
    s_wsle(&io___59);
    do_lio(&c__9, &c__1, " otherwise presision loss errors will happen", (
	    ftnlen)44);
    e_wsle();
    s_wsle(&io___60);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
/* integrate P(w,k) to find amplitude */
/* each cell will be split additionnaly by Ndiv*Ndiv parts */
/* start integration */
    i__1 = integ_1.nw;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__ - 1;
	wa = integ_1.wmin * pow_di(&dw, &i__2);
	i__2 = integ_1.nk;
	for (j = 1; j <= i__2; ++j) {
	    i__3 = j - 1;
	    ka = integ_1.kmin * pow_di(&dk, &i__3);
	    wb = wa * dw;
	    kb = ka * dk;
/* the integral of P(w,k) will be stored to s */
	    s = 0.;
	    dww = (wb - wa) / ndiv;
	    dkk = (kb - ka) / ndiv;
	    ds = dww * dkk;
	    i__3 = ndiv;
	    for (ii = 1; ii <= i__3; ++ii) {
		w = wa + dww * ii;
		i__4 = ndiv;
		for (jj = 1; jj <= i__4; ++jj) {
		    k = ka + dkk * jj;
		    s += ds * pwk_(&w, &k);
		}
	    }
/* the amplitude of ij harmonic is ready */
	    a0 = 2. / pi * sqrt(s);
/* it can be negative (in fact it is not needed because anyway we will choose */
/* random phase, but let it be) */
	    if (ran1_(&randum_1.idum) > .5) {
		a0 = -a0;
	    }
/* the amplitude */
	    harmonics_1.am[i__ + j * 250 - 251] = a0;
	}
    }
    s_wsle(&io___71);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___72);
    do_lio(&c__9, &c__1, " Harmonics generation finished", (ftnlen)30);
    e_wsle();
    s_wsle(&io___73);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
/* here the phases, w and k will be generated or just refreshed */

L1000:
    d__1 = integ_1.kmax / integ_1.kmin;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
    d__1 = integ_1.wmax / integ_1.wmin;
    d__2 = 1. / (integ_1.nw - 1);
    dw = pow_dd(&d__1, &d__2);
    d__1 = 1. / ndiv;
    dww = pow_dd(&dw, &d__1);
    d__1 = 1. / ndiv;
    dkk = pow_dd(&dk, &d__1);
/* store frequency */
    i__1 = integ_1.nw;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__ - 1;
	wa = integ_1.wmin * pow_di(&dw, &i__2);
	wb = wa * dw;
/* we take w between wa and wb (which was the interval of PWK integration) */
/* with uniform distribution like here, so in principle, after many */
/* seeds all frequencies will be checked. */
/* this choice of w, in fact, result in about 50% inaccuracy */
/* for <dx^2> for small t and big l, better results can be obtained */
/* if we will put w=averaged mean weighted values, but it seems */
/* it is not acceptable to have fixed w (and especially k) because */
/* lattice (or supports) may have resonance properties and */
/* all w and k should be present in the spectrum of signal (at least */
/* after big number of seeds). */

	harmonics_1.wh[i__ - 1] = wa + ran1_(&randum_1.idum) * (wb - wa);
    }
/* and store wave number */
    i__1 = integ_1.nk;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	ka = integ_1.kmin * pow_di(&dk, &i__2);
	kb = ka * dk;
/* we do for k the same as for w */

	harmonics_1.kh[j - 1] = ka + ran1_(&randum_1.idum) * (kb - ka);
    }
    i__1 = integ_1.nw;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = integ_1.nk;
	for (j = 1; j <= i__2; ++j) {
/* generate random phase ij for horizontal motion */
	    phase = pi * 2. * ran1_(&randum_1.idum);
/* and store sin and cos of this phase */
	    harmonics_1.dskx[i__ + j * 250 - 251] = sin(phase);
	    harmonics_1.dckx[i__ + j * 250 - 251] = cos(phase);
/* generate random phase ij for vertical motion */
	    phase = pi * 2. * ran1_(&randum_1.idum);
	    harmonics_1.dsky[i__ + j * 250 - 251] = sin(phase);
	    harmonics_1.dcky[i__ + j * 250 - 251] = cos(phase);
	}
    }
/*      write(6,*)' Harmonics phases, w and k made or refreshed' */
/* L2000: */
/* initial values of told , sinus and cosin. Remember that t.ge.0 */
    timeold_1.told = 0.;
    i__1 = earth_1.np;
    for (i__ = 1; i__ <= i__1; ++i__) {
	harmonics_1.sw[i__ - 1] = 0.;
/* it is sin(0*wh(i)) */
	harmonics_1.cw[i__ - 1] = 1.;
/*       cos(0*wh(i)) */
    }
/* this is to remember that harmonics have been generated */
    filejustread_1.inewparam = 1;
/* L100: */
    return 0;
} /* prepare_harmonics__ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int prepare_systhar__()
{
    /* Initialized data */

    static integer ndiv = 30;
    static doublereal pi = 3.14159265358979;

    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    integer s_wsle(), do_lio(), e_wsle();
    double pow_dd(), pow_di(), sqrt(), sin(), cos();

    /* Local variables */
    static doublereal sfin;
    extern /* Subroutine */ int qpsk_();
    static doublereal kmin1[2], kmax1[2];
    static integer i__, j;
    static doublereal k, s, phase, a0, stest[2], s1, ka, kb, dk;
    static integer jj;
    static doublereal ds;
    static integer is;
    static doublereal s1k, dkk;
    extern doublereal qpk_();
    static doublereal rrr;
    extern doublereal ran1_();

    /* Fortran I/O blocks */
    static cilist io___77 = { 0, 6, 0, 0, 0 };
    static cilist io___78 = { 0, 6, 0, 0, 0 };
    static cilist io___92 = { 0, 6, 0, 0, 0 };
    static cilist io___93 = { 0, 6, 0, 0, 0 };
    static cilist io___94 = { 0, 6, 0, 0, 0 };
    static cilist io___95 = { 0, 6, 0, 0, 0 };
    static cilist io___102 = { 0, 6, 0, 0, 0 };
    static cilist io___103 = { 0, 6, 0, 0, 0 };
    static cilist io___104 = { 0, 6, 0, 0, 0 };


/*   calculate wave numbers, amplitudes and phases of harmonics */
/*   responsible for systematic in time motion */
/*   Maximum number of harmonics is limited by NH */
/* inewparam =0 will say to PREPARE_SYST_HARMONICS that the file */
/* with P(w,k) parameters has been just read thus complete */
/* harmonics generation should be performed */
/* if inewparam =1 it means that the harmonics has been already */
/* once calculated, this is just new seed and it is enough to */
/* generate only new phases, w and k */

/* will just regenerate phases and k if inewparams =1 */
    if (filejustread_1.inewparams != 0) {
	goto L1000;
    }
/* test that requested number of harmonics is smaller than array size */
    if (earth_1.np > 250) {
	earth_1.np = 250;
    }
    s_wsle(&io___77);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___78);
    do_lio(&c__9, &c__1, " Finding range of important k for systematic motion"
	    , (ftnlen)51);
    e_wsle();
/* we will estimate range of important k by rough */
/* preliminary integration */
/* of Q(k) over enough big (hopefully) range of k */
    integ_1.nk = earth_1.np;
/* we define this wide initial range k and beleive that */
/* the important range is inside */

    integ_1.kmins = 1e-5;
    integ_1.kmaxs = 1e5;
/* ratio k_{i+1}/k_{i} is constant */

    d__1 = integ_1.kmaxs / integ_1.kmins;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
/* to estimate range of important k we will calculate two */
/* values, namely the mean value < [x(t,s1)-x(t,s2)]^2 > for different */
/* combination of variables, namely */
/* s1-s2=Smin */
/* s1-s2=Smax */
    stest[0] = earth_1.smin;
    stest[1] = earth_1.smax;
/* double loop to check all these four cases */
    for (is = 1; is <= 2; ++is) {
/* the value < [x(s1)-x(s2)]^2 > is the double integral on */
/* Q(k) 2 (1-cos(k (s1-s2))) dk/(2pi) = F(k) */

/* to find the important range of k we calculate first the value */
/* sfin = Int^{kmax}_{kmin} F(k) dk */
/* and then we calculate first the function */
/* s1(k)= Int^{k}_{kmin} F(k) dk */
/* then the ratio s1(k)/sfin will be equal 0 at kmin and 1 at kmax */
/* the region where this function changes rapidly from 0 to 1 */
/* gives main contribution to the integral. */
/* we define the range of important k as the points where */
/* s1(k)/sfin cross the level 0.01 and 0.99 for kmin and kmax */
/* correspondingly */


	sfin = 0.;
	i__1 = integ_1.nk;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j - 1;
	    k = integ_1.kmins * pow_di(&dk, &i__2);
	    ds = k * dk - k;
	    qpsk_(&stest[is - 1], &k, &rrr);
	    sfin += ds * rrr;
	}
	if (sfin == 0.) {
	    goto L500;
	}
	kmin1[is - 1] = 0.;
	kmax1[is - 1] = 0.;
	s1 = 0.;
	i__1 = integ_1.nk;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ - 1;
	    k = integ_1.kmins * pow_di(&dk, &i__2);
	    ds = k * dk - k;
	    qpsk_(&stest[is - 1], &k, &rrr);
	    s1 += ds * rrr;
	    s1k = s1 / sfin;
	    if (kmin1[is - 1] == 0. && s1k > .01) {
		kmin1[is - 1] = k / dk;
	    }
	    if (kmax1[is - 1] == 0. && s1k > .99) {
		kmax1[is - 1] = k * dk;
	    }
	}
    }
/* we have found the important ranges for all of two cases, */
/* now we find the range that cover these two */

    integ_1.kmins = min(kmin1[1],kmin1[0]);
    integ_1.kmaxs = max(kmax1[1],kmax1[0]);
L500:
    d__1 = integ_1.kmaxs / integ_1.kmins;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
    integ_1.kmaxs /= dk;
    s_wsle(&io___92);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___93);
    do_lio(&c__9, &c__1, " Range of important k for systematic motion:", (
	    ftnlen)44);
    e_wsle();
    s_wsle(&io___94);
    do_lio(&c__9, &c__1, " k from ", (ftnlen)8);
    do_lio(&c__5, &c__1, (char *)&integ_1.kmins, (ftnlen)sizeof(doublereal));
    do_lio(&c__9, &c__1, " to ", (ftnlen)4);
    do_lio(&c__5, &c__1, (char *)&integ_1.kmaxs, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___95);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
/* the range of important k has been found */
/* now we start to find amplitude of each harmonic by */
/* integration of Q(k) */
    d__1 = integ_1.kmaxs / integ_1.kmins;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
/* integrate Q(k) to find amplitude */
/* each cell will be split additionnaly by Ndiv*Ndiv parts */
/* start integration */
    i__1 = integ_1.nk;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	ka = integ_1.kmins * pow_di(&dk, &i__2);
	kb = ka * dk;
/* the integral of Q(k) will be stored to s */
	s = 0.;
	dkk = (kb - ka) / ndiv;
	ds = dkk;
	i__2 = ndiv;
	for (jj = 1; jj <= i__2; ++jj) {
	    k = ka + dkk * jj;
	    s += ds * qpk_(&k);
	}
/* the amplitude of ij harmonic is ready */
	a0 = 2. / pi * sqrt(s);
/* it can be negative (in fact it is not needed because anyway we will choose */
/* random phase, but let it be) */
	if (ran1_(&randum_1.idum) > .5) {
	    a0 = -a0;
	}
/* the amplitude */
	harmonics_syst__1.ams[j - 1] = a0;
    }
    s_wsle(&io___102);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___103);
    do_lio(&c__9, &c__1, " Harmonics generation finished for systematic", (
	    ftnlen)45);
    e_wsle();
    s_wsle(&io___104);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
/* here the phases, k will be generated or just refreshed */

L1000:
    d__1 = integ_1.kmaxs / integ_1.kmins;
    d__2 = 1. / (integ_1.nk - 1);
    dk = pow_dd(&d__1, &d__2);
    d__1 = 1. / ndiv;
    dkk = pow_dd(&dk, &d__1);
/* store wave number */
    i__1 = integ_1.nk;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	ka = integ_1.kmins * pow_di(&dk, &i__2);
	kb = ka * dk;
	harmonics_syst__1.khs[j - 1] = ka + ran1_(&randum_1.idum) * (kb - ka);
    }
/* will not regenerate phases if inewparams =2 */
    if (filejustread_1.inewparams == 2) {
	goto L2000;
    }
/* we take k between ka and kb (which was the interval of QPK integration) */
/* with uniform distribution like here, so in principle, after many */
/* seeds all frequencies will be checked. */
/* this choice of k, in fact, result in about some inaccuracy */
/* for <dx^2> but it seems */
/* it is not acceptable to have fixed k because */
/* lattice (or supports) may have resonance properties and */
/* all k should be present in the spectrum of signal (at least */
/* after big number of seeds). */

    i__1 = integ_1.nk;
    for (j = 1; j <= i__1; ++j) {
/* generate random phase j for horizontal motion */
	phase = pi * 2. * ran1_(&randum_1.idum);
/* and store sin and cos of this phase */
	harmonics_syst__1.dskxs[j - 1] = sin(phase);
	harmonics_syst__1.dckxs[j - 1] = cos(phase);
/* generate random phase j for vertical motion */
	phase = pi * 2. * ran1_(&randum_1.idum);
	harmonics_syst__1.dskys[j - 1] = sin(phase);
	harmonics_syst__1.dckys[j - 1] = cos(phase);
    }
/*      write(6,*)' Harmonics phases and k made or refreshed for systematic' */
L2000:
/* initial values of told , sinus and cosin. Remember that t.ge.0 */
    timeold_1.told = 0.;
/* this is to remember that also systematic harmonics have been generated */
    filejustread_1.inewparams = 1;
/* L100: */
    return 0;
} /* prepare_systhar__ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int ppwk_(s, t, k, w, rrr)
doublereal *s, *t, *k, *w, *rrr;
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    extern doublereal fu_(), pwk_();

/* it put to RRR the function */
/* P(w,k) 2 (1-cos(wt)) 2(1-cos(ks)) 2/(2pi) 2/(2pi) */
/* the coefficient "2" in "2/(2pi)" is due to the fact that */
/* we define P(w,k) so that w and k can be positive or negative, */
/* but we will integrate only on positive values */

    d__1 = *w * *t;
    d__2 = *k * *s;
    *rrr = pwk_(w, k) * 2. * fu_(&d__1) * 2. * fu_(&d__2) * .1013211;
/* 2*(1.-cos(w*T)) */
/* 2*(1.-cos(k*L)) */
/* 2*2/6.28/6.28 */
    return 0;
} /* ppwk_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal pwk_(w, k)
doublereal *w, *k;
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double exp();

    /* Local variables */
    static doublereal lo;
    extern doublereal fu_(), fmicro_();
    static doublereal pn1, pn2, pn3, vv1, vv2, vv3, vvs;

/* gives P(w,k) using parameters of the model */
/* for the model explanation see somewhere else */

/* 		 PWK of the "corrected ATL law": */
    if (earth_1.a != 0. && earth_1.b != 0.) {
/* Computing 2nd power */
	d__1 = *w;
	lo = earth_1.b / earth_1.a / (d__1 * d__1);
/* Computing 2nd power */
	d__1 = *w * *k;
	d__2 = lo * *k;
	ret_val = earth_1.a / (d__1 * d__1) * fu_(&d__2);
/* that is (1.-cos(Lo*k)) */
    } else {
	ret_val = 0.;
    }
/*               And wave contribution, three peaks */
    vv1 = earth_1.v1;
    vv2 = earth_1.v2;
    vv3 = earth_1.v3;
    if (earth_1.v1 < 0. || earth_1.v2 < 0. || earth_1.v3 < 0.) {
	vvs = exp(-(*w) / 12.5) * 1900. + 450.;
    }
/* if v < 0 then the SLAC formula is used */
    if (earth_1.v1 < 0.) {
	vv1 = vvs;
    }
    pn1 = fmicro_(w, k, &vv1, &earth_1.f1, &earth_1.d1, &earth_1.a1);
    if (earth_1.v2 < 0.) {
	vv2 = vvs;
    }
    pn2 = fmicro_(w, k, &vv2, &earth_1.f2, &earth_1.d2, &earth_1.a2);
    if (earth_1.v3 < 0.) {
	vv3 = vvs;
    }
    pn3 = fmicro_(w, k, &vv3, &earth_1.f3, &earth_1.d3, &earth_1.a3);
    ret_val = ret_val + pn1 + pn2 + pn3;
    return ret_val;
} /* pwk_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal fu_(x)
doublereal *x;
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Builtin functions */
    double sin();

/* gives (1-cos(x)) */

/* if x is big we replace (1-cos(x)) by its mean value 1 */

    if (*x > 1e8) {
	ret_val = 1.;
	return ret_val;
    }
/* Computing 2nd power */
    d__1 = sin(*x / 2.);
    ret_val = d__1 * d__1 * 2.;
/* 			it equals to (1-cos(x)) */
    return ret_val;
} /* fu_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal fmicro_(w, k, velm, fmic, dmic, amic)
doublereal *w, *k, *velm, *fmic, *dmic, *amic;
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double sqrt();

    /* Local variables */
    extern doublereal cmic_();
    static doublereal km;

/* gives distribution of amplitudes on k and w due to waves */
/* with phase velosity velm */

    km = *w / *velm;
    ret_val = 0.;
    if (*k < km) {
/* this shape of distribution on k assumes that the waves travell */
/* in the plane (i.e. on our surface) and distribution of */
/* directions is homogenious */

/* Computing 2nd power */
	d__1 = *k;
/* Computing 2nd power */
	d__2 = km;
	ret_val = 2. / km / sqrt(1. - d__1 * d__1 / (d__2 * d__2));
/* distribution on w */
	ret_val *= cmic_(w, fmic, dmic, amic);
    }
    return ret_val;
} /* fmicro_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal cmic_(w, fmic, dmic, amic)
doublereal *w, *fmic, *dmic, *amic;
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Local variables */
    static doublereal f, p0, p1;

/* gives distribution of amplitudes on w due to waves */
    f = *w / 2. / 3.1415926;
    p0 = *amic;
/* Computing 4th power */
    d__1 = (f - *fmic) / *fmic * *dmic, d__1 *= d__1;
    p1 = 1. / (d__1 * d__1 + 1.);
    ret_val = p0 * p1;
/* L900: */
    return ret_val;
} /* cmic_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int qpsk_(s, k, rrr)
doublereal *s, *k, *rrr;
{
    /* System generated locals */
    doublereal d__1;

    /* Local variables */
    extern doublereal fu_(), qpk_();

/* it put to RRR the function */
/* Q(k) 2(1-cos(ks)) 2/(2pi) */
/* the coefficient "2" in "2/(2pi)" is due to the fact that */
/* we define Q(k) so that k can be positive or negative, */
/* but we will integrate only on positive values */

    d__1 = *k * *s;
    *rrr = qpk_(k) * 2. * fu_(&d__1) * .318;
/* 2*(1.-cos(k*L)) */
/* 2/6.28 */
    return 0;
} /* qpsk_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal qpk_(k)
doublereal *k;
{
    /* System generated locals */
    doublereal ret_val, d__1;

/* gives Q(k) using parameters of the model */
/* for the model explanation see somewhere else */

/* 	function cmic(w,fmic,dmic,amic) */
/* gives distribution of amplitudes on w due to waves */
/* 	f=w/2./3.1415926 */
/* !	QPK=Q1/( ((k-rk1)/rk1*rkk1)**4 +1.) */
/* !	qqq=(k/rk1)**4 */
/* !	QPK=QPK*qqq /( qqq +1.) */
/* Computing 2nd power */
    d__1 = (*k / 6.2832 - earth_1.rk1) / earth_1.rk1 * earth_1.rkk1;
    ret_val = earth_1.q1 / (d__1 * d__1 + 1.);
    return ret_val;
} /* qpk_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal ran1_(idum)
integer *idum;
{
    /* Initialized data */

    static integer iv[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	    0,0,0,0,0,0,0,0 };
    static integer iy = 0;

    /* System generated locals */
    integer i__1;
    doublereal ret_val, d__1;

    /* Local variables */
    static integer j, k;

/* ----------------------------------------------------------------------- */
/*            Copied from "Numerical Recipes", page 271 */
/* ----------------------------------------------------------------------- */
/*  "Minimal" random number generator of Park and Miller with Bays-Durham */
/*   shuffle and added safeguards. Returns a uniform random deviate */
/*   between 0.0 and 1.0 (exclusive of the endpoint values). */
/*   Call with "idum" a negative number to initialize; thereafter, */
/*   do not alter "idum" between successive deviates in a sequence. */
/*   "rnmx" should approximate the largest floating value that is */
/*   less than 1. */
    if (*idum <= 0 || iy == 0) {
/* Computing MAX */
	i__1 = -(*idum);
	*idum = max(i__1,1);
	for (j = 40; j >= 1; --j) {
	    k = *idum / 127773;
	    *idum = (*idum - k * 127773) * 16807 - k * 2836;
	    if (*idum < 0) {
		*idum += 2147483647;
	    }
	    if (j <= 32) {
		iv[j - 1] = *idum;
	    }
	}
	iy = iv[0];
    }
    k = *idum / 127773;
    *idum = (*idum - k * 127773) * 16807 - k * 2836;
    if (*idum < 0) {
	*idum += 2147483647;
    }
    j = iy / 67108864 + 1;
    iy = iv[j - 1];
    iv[j - 1] = *idum;
/* Computing MIN */
    d__1 = iy * 4.6566128752457969e-10;
    ret_val = min(d__1,.99999987999999995);
    return ret_val;
} /* ran1_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int read_pwk_param__()
{
    /* Format strings */
    static char fmt_920[] = "(a,1pe12.5)";
    static char fmt_921[] = "(a,i6)";

    /* System generated locals */
    integer i__1;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer f_open(), s_rsle(), do_lio(), e_rsle(), f_clos(), s_wsle(), 
	    e_wsle();
    /* Subroutine */ int s_copy();
    integer s_wsfe(), do_fio(), e_wsfe();

    /* Local variables */
    static doublereal dt;
    static char chardum[50];

    /* Fortran I/O blocks */
    static cilist io___122 = { 1, 92, 1, 0, 0 };
    static cilist io___124 = { 1, 92, 1, 0, 0 };
    static cilist io___125 = { 1, 92, 1, 0, 0 };
    static cilist io___126 = { 1, 92, 1, 0, 0 };
    static cilist io___127 = { 1, 92, 1, 0, 0 };
    static cilist io___128 = { 1, 92, 1, 0, 0 };
    static cilist io___129 = { 1, 92, 1, 0, 0 };
    static cilist io___130 = { 1, 92, 1, 0, 0 };
    static cilist io___131 = { 1, 92, 1, 0, 0 };
    static cilist io___132 = { 1, 92, 1, 0, 0 };
    static cilist io___133 = { 1, 92, 1, 0, 0 };
    static cilist io___134 = { 1, 92, 1, 0, 0 };
    static cilist io___135 = { 1, 92, 1, 0, 0 };
    static cilist io___136 = { 1, 92, 1, 0, 0 };
    static cilist io___137 = { 1, 92, 1, 0, 0 };
    static cilist io___138 = { 1, 92, 1, 0, 0 };
    static cilist io___139 = { 1, 92, 1, 0, 0 };
    static cilist io___140 = { 1, 92, 1, 0, 0 };
    static cilist io___141 = { 1, 92, 1, 0, 0 };
    static cilist io___142 = { 1, 92, 1, 0, 0 };
    static cilist io___143 = { 1, 92, 1, 0, 0 };
    static cilist io___144 = { 1, 92, 1, 0, 0 };
    static cilist io___145 = { 0, 6, 0, 0, 0 };
    static cilist io___146 = { 0, 6, 0, 0, 0 };
    static cilist io___147 = { 0, 6, 0, 0, 0 };
    static cilist io___148 = { 0, 6, 0, 0, 0 };
    static cilist io___149 = { 0, 6, 0, 0, 0 };
    static cilist io___151 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___152 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___153 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___154 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___155 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___156 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___157 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___158 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___159 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___160 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___161 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___162 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___163 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___164 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___165 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___166 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___167 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___168 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___169 = { 0, 92, 0, fmt_921, 0 };
    static cilist io___170 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___171 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___172 = { 0, 92, 0, fmt_920, 0 };
    static cilist io___173 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___174 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___175 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___176 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___177 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___178 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___179 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___180 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___181 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___182 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___183 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___184 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___185 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___186 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___187 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___188 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___189 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___190 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___191 = { 0, 6, 0, fmt_921, 0 };
    static cilist io___192 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___193 = { 0, 6, 0, fmt_920, 0 };
    static cilist io___194 = { 0, 6, 0, fmt_920, 0 };


/* read parameters of P(w,k) from the file and put to the common block */
/* if there is no input file, a version that correspond to very noisy */
/* conditions such as in the HERA tunnel will be created */
    filejustread_1.inewparam = 0;
    o__1.oerr = 1;
    o__1.ounit = 92;
    o__1.ofnmlen = 10;
    o__1.ofnm = "model.data";
    o__1.orl = 0;
    o__1.osta = "old";
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    i__1 = f_open(&o__1);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___122);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.a, (ftnlen)sizeof(doublereal)
	    );
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___124);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.b, (ftnlen)sizeof(doublereal)
	    );
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___125);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.f1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___126);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.a1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___127);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.d1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___128);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.v1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___129);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.f2, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___130);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.a2, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___131);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.d2, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___132);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.v2, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___133);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.f3, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___134);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.a3, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___135);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.d3, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___136);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.v3, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___137);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.tmin, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___138);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.tmax, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___139);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.smin, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___140);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.smax, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___141);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__3, &c__1, (char *)&earth_1.np, (ftnlen)sizeof(integer));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___142);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.q1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___143);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.rk1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = s_rsle(&io___144);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__9, &c__1, chardum, (ftnlen)50);
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = do_lio(&c__5, &c__1, (char *)&earth_1.rkk1, (ftnlen)sizeof(
	    doublereal));
    if (i__1 != 0) {
	goto L999;
    }
    i__1 = e_rsle();
    if (i__1 != 0) {
	goto L999;
    }
    cl__1.cerr = 0;
    cl__1.cunit = 92;
    cl__1.csta = 0;
    f_clos(&cl__1);
    goto L900;
L999:
    cl__1.cerr = 0;
    cl__1.cunit = 92;
    cl__1.csta = 0;
    f_clos(&cl__1);
    s_wsle(&io___145);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___146);
    do_lio(&c__9, &c__1, " Error reading file \"model.data\", a new", (ftnlen)
	    39);
    e_wsle();
    s_wsle(&io___147);
    do_lio(&c__9, &c__1, " version of the file has been created and used", (
	    ftnlen)46);
    e_wsle();
    s_wsle(&io___148);
    do_lio(&c__9, &c__1, " (conditions such as HERA tunnel+SLAC systematics)",
	     (ftnlen)50);
    e_wsle();
    s_wsle(&io___149);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    earth_1.a = 1e-16;
    earth_1.b = 1e-15;
    earth_1.f1 = .14;
    earth_1.a1 = 9.9999999999999994e-12;
    earth_1.d1 = 5.;
    earth_1.v1 = 1e3;
    earth_1.f2 = 2.5;
    earth_1.a2 = 1.0000000000000001e-15;
    earth_1.d2 = 1.5;
    earth_1.v2 = 400.;
    earth_1.f3 = 50.;
    earth_1.a3 = 9.9999999999999998e-20;
    earth_1.d3 = 1.5;
    earth_1.v3 = 400.;
    earth_1.tmin = .001;
    earth_1.tmax = 1e4;
    earth_1.smin = 1.;
    earth_1.smax = 1e3;
    earth_1.np = 50;
    dt = .02;
    earth_1.q1 = 1e-21;
    earth_1.rk1 = .1;
    earth_1.rkk1 = 1.;
    o__1.oerr = 0;
    o__1.ounit = 92;
    o__1.ofnmlen = 10;
    o__1.ofnm = "model.data";
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    s_copy(chardum, "'Parameter A of the ATL law,     A [m**2/m/s]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___151);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Parameter B of the PWK,         B [m**2/s**3]  '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___152);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.b, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Frequency of 1-st peak in PWK,  f1 [Hz]        '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___153);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.f1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Amplitude of 1-st peak in PWK,  a1 [m**2/Hz]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___154);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of 1-st peak in PWK,      d1 [1]         '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___155);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.d1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Velocity of 1-st peak in PWK,   v1 [m/s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___156);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.v1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Frequency of 2-nd peak in PWK,  f2 [Hz]        '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___157);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.f2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Amplitude of 2-nd peak in PWK,  a2 [m**2/Hz]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___158);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of 2-nd peak in PWK,      d2 [1]         '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___159);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.d2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Velocity of 2-nd peak in PWK,   v2 [m/s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___160);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.v2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Frequency of 3-rd peak in PWK,  f3 [Hz]        '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___161);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.f3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Amplitude of 3-rd peak in PWK,  a3 [m**2/Hz]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___162);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of 3-rd peak in PWK,      d3 [1]         '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___163);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.d3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Velocity of 3-rd peak in PWK,   v3 [m/s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___164);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.v3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Minimum time,                   Tmin [s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___165);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.tmin, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Maximum time,                   Tmax [s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___166);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.tmax, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Minimum distance,               Smin [m]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___167);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.smin, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Maximum distance,               Smax [m]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___168);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.smax, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Number of w or k harmonics,     Np   [1]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___169);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.np, (ftnlen)sizeof(integer));
    e_wsfe();
    s_copy(chardum, "'Ampl. of peak in systematic.P,Q1  [m**3*Hz**2] '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___170);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.q1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Wavenumber of peak in syst.P,  rk1 [1/m]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___171);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.rk1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of peak in system.P,      rkk1 [1]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___172);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.rkk1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    cl__1.cerr = 0;
    cl__1.cunit = 92;
    cl__1.csta = 0;
    f_clos(&cl__1);
L900:
    s_copy(chardum, "'Parameter A of the ATL law,     A [m**2/m/s]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___173);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Parameter B of the PWK,         B [m**2/s**3]  '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___174);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.b, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Frequency of 1-st peak in PWK,  f1 [Hz]        '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___175);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.f1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Amplitude of 1-st peak in PWK,  a1 [m**2/Hz]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___176);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of 1-st peak in PWK,      d1 [1]         '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___177);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.d1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Velocity of 1-st peak in PWK,   v1 [m/s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___178);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.v1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Frequency of 2-nd peak in PWK,  f2 [Hz]        '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___179);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.f2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Amplitude of 2-nd peak in PWK,  a2 [m**2/Hz]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___180);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of 2-nd peak in PWK,      d2 [1]         '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___181);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.d2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Velocity of 2-nd peak in PWK,   v2 [m/s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___182);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.v2, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Frequency of 3-rd peak in PWK,  f3 [Hz]        '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___183);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.f3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Amplitude of 3-rd peak in PWK,  a3 [m**2/Hz]   '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___184);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.a3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of 3-rd peak in PWK,      d3 [1]         '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___185);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.d3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Velocity of 3-rd peak in PWK,   v3 [m/s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___186);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.v3, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Minimum time,                   Tmin [s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___187);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.tmin, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Maximum time,                   Tmax [s]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___188);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.tmax, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Minimum distance,               Smin [m]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___189);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.smin, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Maximum distance,               Smax [m]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___190);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.smax, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Number of w or k harmonics,     Np   [1]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___191);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.np, (ftnlen)sizeof(integer));
    e_wsfe();
    s_copy(chardum, "'Ampl. of peak in systematic.P,Q1  [m**3*Hz**2] '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___192);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.q1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Wavenumber of peak in syst.P,  rk1 [1/m]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___193);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.rk1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    s_copy(chardum, "'Width of peak in system.P,      rkk1 [1]       '", (
	    ftnlen)50, (ftnlen)49);
    s_wsfe(&io___194);
    do_fio(&c__1, chardum, (ftnlen)50);
    do_fio(&c__1, (char *)&earth_1.rkk1, (ftnlen)sizeof(doublereal));
    e_wsfe();
    return 0;
} /* read_pwk_param__ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
doublereal x2pwka_(a, b, t, rl)
doublereal *a, *b, *t, *rl;
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Builtin functions */
    double sqrt();

    /* Local variables */
    static doublereal q, t0, pi, tt0;

/* gives approximation of <[x(t,s+rl)-x(t,s)]^2> for pure ATL P(w,k) */
/* needed only for the TESTS subroutine */

    ret_val = 0.;
    if (*t == 0. || *rl == 0. || *a == 0. || *b == 0.) {
	goto L900;
    }
    pi = 3.14159265358979;
    t0 = sqrt(*a * *rl / *b);
    tt0 = *t / t0;
    q = 1. / (tt0 + 1. / tt0 + 1.) + 1.;
/* Computing 2nd power */
    d__1 = tt0;
    ret_val = *a * t0 * *rl * (d__1 * d__1) * 2. / pi / (2. / pi * tt0 + 1.) *
	     q;
L900:
    return ret_val;
} /* x2pwka_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int tests_()
{
    /* Format strings */
    static char fmt_100[] = "(5e13.5)";
    static char fmt_150[] = "(8e13.5)";

    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1, d__2;

    /* Builtin functions */
    integer s_wsle(), do_lio(), e_wsle();
    double sqrt();
    integer s_rsle(), e_rsle(), s_wsfe(), do_fio(), e_wsfe();
    double pow_dd(), pow_di();

    /* Local variables */
    extern /* Subroutine */ int prepare_systhar__();
    static doublereal time;
    extern /* Subroutine */ int misalign_();
    static doublereal tmaxhere;
    static integer k, n;
    static doublereal s, t, x, y;
    static integer iseed, nseed;
    static doublereal dtmaxhere;
    extern doublereal x2pwka_();
    extern /* Subroutine */ int prepare_harmonics__();
    static doublereal dt, dx[60]	/* was [3][20] */, dy[60]	/* 
	    was [3][20] */;
    static integer it, nt;
    extern /* Subroutine */ int xy_pwk__();
    static doublereal xma[7], xmi[7];

    /* Fortran I/O blocks */
    static cilist io___199 = { 0, 6, 0, 0, 0 };
    static cilist io___200 = { 0, 6, 0, 0, 0 };
    static cilist io___201 = { 0, 6, 0, 0, 0 };
    static cilist io___202 = { 0, 6, 0, 0, 0 };
    static cilist io___203 = { 0, 6, 0, 0, 0 };
    static cilist io___204 = { 0, 6, 0, 0, 0 };
    static cilist io___205 = { 0, 6, 0, 0, 0 };
    static cilist io___208 = { 0, 6, 0, 0, 0 };
    static cilist io___209 = { 0, 5, 0, 0, 0 };
    static cilist io___211 = { 0, 6, 0, 0, 0 };
    static cilist io___212 = { 0, 6, 0, 0, 0 };
    static cilist io___213 = { 0, 6, 0, 0, 0 };
    static cilist io___214 = { 0, 6, 0, 0, 0 };
    static cilist io___215 = { 0, 6, 0, 0, 0 };
    static cilist io___217 = { 0, 8, 0, fmt_100, 0 };
    static cilist io___218 = { 0, 9, 0, fmt_100, 0 };
    static cilist io___219 = { 0, 6, 0, 0, 0 };
    static cilist io___220 = { 0, 6, 0, 0, 0 };
    static cilist io___221 = { 0, 6, 0, 0, 0 };
    static cilist io___222 = { 0, 6, 0, 0, 0 };
    static cilist io___223 = { 0, 6, 0, 0, 0 };
    static cilist io___224 = { 0, 6, 0, 0, 0 };
    static cilist io___226 = { 0, 6, 0, 0, 0 };
    static cilist io___227 = { 0, 6, 0, 0, 0 };
    static cilist io___236 = { 0, 13, 0, fmt_150, 0 };
    static cilist io___237 = { 0, 14, 0, fmt_150, 0 };
    static cilist io___239 = { 0, 6, 0, 0, 0 };
    static cilist io___240 = { 0, 6, 0, 0, 0 };
    static cilist io___241 = { 0, 6, 0, 0, 0 };
    static cilist io___242 = { 0, 6, 0, 0, 0 };
    static cilist io___243 = { 0, 6, 0, 0, 0 };
    static cilist io___244 = { 0, 6, 0, 0, 0 };
    static cilist io___245 = { 0, 6, 0, 0, 0 };
    static cilist io___246 = { 0, 6, 0, 0, 0 };
    static cilist io___247 = { 0, 6, 0, 0, 0 };
    static cilist io___248 = { 0, 6, 0, 0, 0 };
    static cilist io___249 = { 0, 6, 0, 0, 0 };
    static cilist io___250 = { 0, 6, 0, 0, 0 };
    static cilist io___255 = { 0, 10, 0, fmt_100, 0 };
    static cilist io___256 = { 0, 11, 0, fmt_100, 0 };
    static cilist io___257 = { 0, 12, 0, fmt_100, 0 };
    static cilist io___258 = { 0, 6, 0, 0, 0 };
    static cilist io___259 = { 0, 6, 0, 0, 0 };
    static cilist io___260 = { 0, 6, 0, 0, 0 };
    static cilist io___261 = { 0, 6, 0, 0, 0 };


/* do some tests using generated harmonics */

    s_wsle(&io___199);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___200);
    do_lio(&c__9, &c__1, " Start of TESTS", (ftnlen)15);
    e_wsle();
    line_1.nelem = 4;
    line_1.ss[0] = 0.;
    line_1.ss[1] = earth_1.smin;
    line_1.ss[2] = sqrt(earth_1.smax * earth_1.smin);
    line_1.ss[3] = earth_1.smax;
    s_wsle(&io___201);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    s_wsle(&io___202);
    do_lio(&c__9, &c__1, " write to ftn08 t, x(0), x(s1), x(s2), x(s3)", (
	    ftnlen)44);
    e_wsle();
    s_wsle(&io___203);
    do_lio(&c__9, &c__1, " write to ftn09 t, y(0), y(s1), y(s2), y(s3)", (
	    ftnlen)44);
    e_wsle();
    s_wsle(&io___204);
    do_lio(&c__9, &c__1, " where s1, s2, s3 =", (ftnlen)19);
    do_lio(&c__5, &c__1, (char *)&earth_1.smin, (ftnlen)sizeof(doublereal));
    d__1 = sqrt(earth_1.smax * earth_1.smin);
    do_lio(&c__5, &c__1, (char *)&d__1, (ftnlen)sizeof(doublereal));
    do_lio(&c__5, &c__1, (char *)&earth_1.smax, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___205);
    do_lio(&c__9, &c__1, " Nt is number of points on time", (ftnlen)31);
    e_wsle();
    nt = 1000;
    tmaxhere = earth_1.tmax;
    s_wsle(&io___208);
    do_lio(&c__9, &c__1, "Input Nt=", (ftnlen)9);
    e_wsle();
    s_rsle(&io___209);
    do_lio(&c__3, &c__1, (char *)&nt, (ftnlen)sizeof(integer));
    e_rsle();
    dtmaxhere = tmaxhere / nt;
    s_wsle(&io___211);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    s_wsle(&io___212);
    do_lio(&c__9, &c__1, " In tests dtmax=", (ftnlen)16);
    do_lio(&c__5, &c__1, (char *)&dtmaxhere, (ftnlen)sizeof(doublereal));
    e_wsle();
    if (dtmaxhere > maxtimediff_1.difftmax) {
	tmaxhere = maxtimediff_1.difftmax * nt;
	s_wsle(&io___213);
	do_lio(&c__9, &c__1, "  ", (ftnlen)2);
	e_wsle();
	s_wsle(&io___214);
	do_lio(&c__9, &c__1, "TESTS: too big Tmax, localy changed to ", (
		ftnlen)39);
	do_lio(&c__5, &c__1, (char *)&tmaxhere, (ftnlen)sizeof(doublereal));
	e_wsle();
	s_wsle(&io___215);
	do_lio(&c__9, &c__1, " (or one can change number of steps in TESTS)", 
		(ftnlen)45);
	e_wsle();
    }
    d__1 = tmaxhere;
    d__2 = tmaxhere / nt;
    for (time = 0.; d__2 < 0 ? time >= d__1 : time <= d__1; time += d__2) {
	misalign_(&time);
	s_wsfe(&io___217);
	do_fio(&c__1, (char *)&time, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.xx[0], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.xx[1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.xx[2], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.xx[3], (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_wsfe(&io___218);
	do_fio(&c__1, (char *)&time, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.yy[0], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.yy[1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.yy[2], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&line_1.yy[3], (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    s_wsle(&io___219);
    do_lio(&c__9, &c__1, " finished ", (ftnlen)10);
    e_wsle();
/* **** */
    s_wsle(&io___220);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    s_wsle(&io___221);
    do_lio(&c__9, &c__1, " write to ftn13 s, x(t1), x(t2) ... x(t5)", (ftnlen)
	    41);
    e_wsle();
    s_wsle(&io___222);
    do_lio(&c__9, &c__1, " write to ftn14 s, y(t1), y(t2) ... y(t5)", (ftnlen)
	    41);
    e_wsle();
    s_wsle(&io___223);
    do_lio(&c__9, &c__1, " where s in meters", (ftnlen)18);
    e_wsle();
    s_wsle(&io___224);
    do_lio(&c__9, &c__1, " max time ", (ftnlen)10);
    do_lio(&c__5, &c__1, (char *)&tmaxhere, (ftnlen)sizeof(doublereal));
    e_wsle();
    dt = 28571428.571428571;
/* Tmaxhere/7		!5 */
    s_wsle(&io___226);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    s_wsle(&io___227);
    do_lio(&c__9, &c__1, " In files 13,14 dt=", (ftnlen)19);
    do_lio(&c__5, &c__1, (char *)&dt, (ftnlen)sizeof(doublereal));
    e_wsle();

    for (it = 1; it <= 7; ++it) {
/* 5 */
	t = dt * it;
	xy_pwk__(&t, &c_b491, &x, &y);
	xmi[it - 1] = x;
	xy_pwk__(&t, &c_b492, &x, &y);
	xma[it - 1] = x;
    }
    for (n = 1; n <= 1277; ++n) {
/*       s=3000./300*n	! length of the SLC linac is 2 miles ... */
/* 300 */
	s = n * 21.143304620203601;
/* length of the LEP is 27 km ... */
	for (it = 1; it <= 7; ++it) {
/* 5 */
	    t = dt * it;
	    xy_pwk__(&t, &s, &x, &y);
	    line_1.xx[it - 1] = x;
/* (it) used not as was supposed to, but OK */
	    line_1.yy[it - 1] = y;
	}
	s_wsfe(&io___236);
	do_fio(&c__1, (char *)&s, (ftnlen)sizeof(doublereal));
	for (it = 1; it <= 7; ++it) {
	    d__2 = line_1.xx[it - 1] - xmi[it - 1] - (xma[it - 1] - xmi[it - 
		    1]) / 1276. * (n - 1);
	    do_fio(&c__1, (char *)&d__2, (ftnlen)sizeof(doublereal));
	}
	e_wsfe();
/* 5) */
	s_wsfe(&io___237);
	do_fio(&c__1, (char *)&s, (ftnlen)sizeof(doublereal));
	for (it = 1; it <= 7; ++it) {
	    do_fio(&c__1, (char *)&line_1.yy[it - 1], (ftnlen)sizeof(
		    doublereal));
	}
	e_wsfe();
/* 5) */
    }

    nt = 20;
    nseed = 50;
    tmaxhere = earth_1.tmax;
L5000:
    d__2 = tmaxhere / earth_1.tmin;
    d__1 = 1. / (nt - 1);
    dt = pow_dd(&d__2, &d__1);
    i__1 = nt - 1;
    i__2 = nt - 2;
    dtmaxhere = earth_1.tmin * pow_di(&dt, &i__1) - earth_1.tmin * pow_di(&dt,
	     &i__2);
    s_wsle(&io___239);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    s_wsle(&io___240);
    do_lio(&c__9, &c__1, " In tests dtmax=", (ftnlen)16);
    do_lio(&c__5, &c__1, (char *)&dtmaxhere, (ftnlen)sizeof(doublereal));
    e_wsle();
    if (dtmaxhere > maxtimediff_1.difftmax) {
	tmaxhere = maxtimediff_1.difftmax / (1. - 1. / dt);
	s_wsle(&io___241);
	do_lio(&c__9, &c__1, "  ", (ftnlen)2);
	e_wsle();
	s_wsle(&io___242);
	do_lio(&c__9, &c__1, "TESTS: too big Tmax, localy changed to ", (
		ftnlen)39);
	do_lio(&c__5, &c__1, (char *)&tmaxhere, (ftnlen)sizeof(doublereal));
	e_wsle();
	s_wsle(&io___243);
	do_lio(&c__9, &c__1, " (or one can change array size in TESTS)", (
		ftnlen)40);
	e_wsle();
	goto L5000;
    }
/* here we will calculate <[x(t,s+rl)-x(t,s)]^2> for different */
/* t and sl, with Nseed number of averaging. Each time we should */
/* generate new harmonics thus it is time consuming */
/* the approximate value of <[x(t,s+rl)-x(t,s)]^2> for the */
/* "corrected ATL" will be also calculated and if our model */
/* has no wave contribution (amplitudes of all three peaks are zero) */
/* then these calculated and analytical values should be close */
/* it allows to see that program works well or not */
/* for example it can allow to estimate number of harmonics */
/* that we really need to describe ground motion with */
/* desired accuracy */

    s_wsle(&io___244);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    s_wsle(&io___245);
    do_lio(&c__9, &c__1, " write t,rms_x, rms_y, rms_quiet_pwk ", (ftnlen)37);
    e_wsle();
    s_wsle(&io___246);
    do_lio(&c__9, &c__1, " to ftn10 for ds=", (ftnlen)17);
    do_lio(&c__5, &c__1, (char *)&earth_1.smin, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___247);
    do_lio(&c__9, &c__1, " to ftn11 for ds=", (ftnlen)17);
    d__2 = sqrt(earth_1.smax * earth_1.smin);
    do_lio(&c__5, &c__1, (char *)&d__2, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___248);
    do_lio(&c__9, &c__1, " to ftn12 for ds=", (ftnlen)17);
    do_lio(&c__5, &c__1, (char *)&earth_1.smax, (ftnlen)sizeof(doublereal));
    e_wsle();
    s_wsle(&io___249);
    do_lio(&c__9, &c__1, " number of seeds=", (ftnlen)17);
    do_lio(&c__3, &c__1, (char *)&nseed, (ftnlen)sizeof(integer));
    e_wsle();
    s_wsle(&io___250);
    do_lio(&c__9, &c__1, "  ", (ftnlen)2);
    e_wsle();
    i__1 = nseed;
    for (iseed = 1; iseed <= i__1; ++iseed) {
/*      write(6,*)' iseed=',iseed */
	prepare_harmonics__();
	prepare_systhar__();
	i__2 = nt;
	for (n = 1; n <= i__2; ++n) {
	    i__3 = n - 1;
	    t = earth_1.tmin * pow_di(&dt, &i__3);
	    misalign_(&t);
	    for (k = 1; k <= 3; ++k) {
/* Computing 2nd power */
		d__2 = line_1.xx[0] - line_1.xx[k];
		dx[k + n * 3 - 4] += d__2 * d__2;
/* Computing 2nd power */
		d__2 = line_1.yy[0] - line_1.yy[k];
		dy[k + n * 3 - 4] += d__2 * d__2;
	    }
	}
    }
    i__1 = nt;
    for (n = 1; n <= i__1; ++n) {
	i__2 = n - 1;
	t = earth_1.tmin * pow_di(&dt, &i__2);
	for (k = 1; k <= 3; ++k) {
	    dx[k + n * 3 - 4] = sqrt(dx[k + n * 3 - 4] / nseed);
	    dy[k + n * 3 - 4] = sqrt(dy[k + n * 3 - 4] / nseed);
	}
	s_wsfe(&io___255);
	do_fio(&c__1, (char *)&t, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&dx[n * 3 - 3], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&dy[n * 3 - 3], (ftnlen)sizeof(doublereal));
	d__2 = sqrt(x2pwka_(&earth_1.a, &earth_1.b, &t, &earth_1.smin));
	do_fio(&c__1, (char *)&d__2, (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_wsfe(&io___256);
	do_fio(&c__1, (char *)&t, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&dx[n * 3 - 2], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&dy[n * 3 - 2], (ftnlen)sizeof(doublereal));
	d__1 = sqrt(earth_1.smin * earth_1.smax);
	d__2 = sqrt(x2pwka_(&earth_1.a, &earth_1.b, &t, &d__1));
	do_fio(&c__1, (char *)&d__2, (ftnlen)sizeof(doublereal));
	e_wsfe();
	s_wsfe(&io___257);
	do_fio(&c__1, (char *)&t, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&dx[n * 3 - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&dy[n * 3 - 1], (ftnlen)sizeof(doublereal));
	d__2 = sqrt(x2pwka_(&earth_1.a, &earth_1.b, &t, &earth_1.smax));
	do_fio(&c__1, (char *)&d__2, (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    s_wsle(&io___258);
    do_lio(&c__9, &c__1, " TESTS finished.  ", (ftnlen)18);
    e_wsle();
    s_wsle(&io___259);
    do_lio(&c__9, &c__1, "  Check files ftn08, ftn09, ftn10, ftn11, ftn12 ", (
	    ftnlen)48);
    e_wsle();
    s_wsle(&io___260);
    do_lio(&c__9, &c__1, " In C version the names are ", (ftnlen)28);
    e_wsle();
    s_wsle(&io___261);
    do_lio(&c__9, &c__1, " fort.8, fort.9, fort.10, fort.11, fort.12 ", (
	    ftnlen)43);
    e_wsle();
    return 0;
} /* tests_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int tests2_()
{
    /* Format strings */
    static char fmt_101[] = "(20a)";
    static char fmt_100[] = "(5e13.5)";
    static char fmt_150[] = "(6e13.5)";

    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer s_wsle(), do_lio(), e_wsle(), f_open(), s_rsfe(), do_fio(), 
	    e_rsfe(), s_rsle(), e_rsle(), f_clos(), s_wsfe(), e_wsfe();

    /* Local variables */
    extern /* Subroutine */ int prepare_systhar__();
    static integer npoi;
    static doublereal sumd;
    static integer nseedmax;
    static doublereal sumd_min__;
    static integer i__, j;
    static doublereal scale, x;
    static integer iseed;
    static doublereal y, optam[250], optkh[250], optcx[250], optcy[250], 
	    sydym, optsx[250], optsy[250], zd[300], yd[300], ym[300];
    extern /* Subroutine */ int xy_pwk__();
    static char aaa[20];
    static doublereal opt_scl__, sym2;

    /* Fortran I/O blocks */
    static cilist io___262 = { 0, 6, 0, 0, 0 };
    static cilist io___263 = { 0, 6, 0, 0, 0 };
    static cilist io___264 = { 0, 1, 0, fmt_101, 0 };
    static cilist io___266 = { 0, 1, 0, fmt_101, 0 };
    static cilist io___269 = { 0, 1, 0, 0, 0 };
    static cilist io___272 = { 0, 6, 0, 0, 0 };
    static cilist io___275 = { 0, 6, 0, 0, 0 };
    static cilist io___276 = { 0, 5, 0, 0, 0 };
    static cilist io___281 = { 0, 6, 0, 0, 0 };
    static cilist io___294 = { 0, 6, 0, 0, 0 };
    static cilist io___295 = { 0, 6, 0, 0, 0 };
    static cilist io___296 = { 0, 1, 0, fmt_100, 0 };
    static cilist io___297 = { 0, 1, 0, 0, 0 };
    static cilist io___298 = { 0, 1, 0, fmt_150, 0 };
    static cilist io___299 = { 0, 6, 0, 0, 0 };
    static cilist io___300 = { 0, 6, 0, 0, 0 };
    static cilist io___301 = { 0, 6, 0, 0, 0 };
    static cilist io___302 = { 0, 1, 0, 0, 0 };
    static cilist io___303 = { 0, 1, 0, fmt_150, 0 };
    static cilist io___304 = { 0, 6, 0, 0, 0 };


/* do some tests using generated harmonics */

    s_wsle(&io___262);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___263);
    do_lio(&c__9, &c__1, " Start of TESTS2", (ftnlen)16);
    e_wsle();
    o__1.oerr = 0;
    o__1.ounit = 1;
    o__1.ofnmlen = 11;
    o__1.ofnm = "slc_83v.dat";
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    s_rsfe(&io___264);
    do_fio(&c__1, aaa, (ftnlen)20);
    e_rsfe();
    s_rsfe(&io___266);
    do_fio(&c__1, aaa, (ftnlen)20);
    e_rsfe();
    npoi = 276;
    i__1 = npoi;
    for (i__ = 1; i__ <= i__1; ++i__) {
	s_rsle(&io___269);
	do_lio(&c__5, &c__1, (char *)&zd[i__ - 1], (ftnlen)sizeof(doublereal))
		;
	do_lio(&c__5, &c__1, (char *)&yd[i__ - 1], (ftnlen)sizeof(doublereal))
		;
	e_rsle();
	yd[i__ - 1] /= 1e3;
    }
    cl__1.cerr = 0;
    cl__1.cunit = 1;
    cl__1.csta = 0;
    f_clos(&cl__1);
    s_wsle(&io___272);
    do_lio(&c__9, &c__1, " File read", (ftnlen)10);
    e_wsle();
    sumd_min__ = 0.;
    nseedmax = 5000;
    s_wsle(&io___275);
    do_lio(&c__9, &c__1, "input Nseedmax=", (ftnlen)15);
    e_wsle();
    s_rsle(&io___276);
    do_lio(&c__3, &c__1, (char *)&nseedmax, (ftnlen)sizeof(integer));
    e_rsle();
    i__1 = nseedmax;
    for (iseed = 1; iseed <= i__1; ++iseed) {
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    xy_pwk__(&earth_1.tmax, &zd[i__ - 1], &x, &y);
	    ym[i__ - 1] = y;
	}
	s_wsle(&io___281);
	do_lio(&c__9, &c__1, " model ym generated", (ftnlen)19);
	e_wsle();
	sym2 = 0.;
	sydym = 0.;
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing 2nd power */
	    d__1 = ym[i__ - 1];
	    sym2 += d__1 * d__1;
	    sydym += ym[i__ - 1] * yd[i__ - 1];
	}
	scale = sydym / sym2;
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ym[i__ - 1] *= scale;
	}
/*      write(6,*)' model scaled by *',scale */
	sumd = 0.;
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing 2nd power */
	    d__1 = ym[i__ - 1] - yd[i__ - 1];
	    sumd += d__1 * d__1;
	}
	sumd /= npoi;
	if (sumd_min__ != 0.) {
	    if (sumd < sumd_min__) {
		sumd_min__ = sumd;
		opt_scl__ = scale;
		i__2 = earth_1.np;
		for (j = 1; j <= i__2; ++j) {
		    optsx[j - 1] = harmonics_syst__1.dskxs[j - 1];
		    optcx[j - 1] = harmonics_syst__1.dckxs[j - 1];
		    optsy[j - 1] = harmonics_syst__1.dskys[j - 1];
		    optcy[j - 1] = harmonics_syst__1.dckys[j - 1];
		    optam[j - 1] = harmonics_syst__1.ams[j - 1];
		    optkh[j - 1] = harmonics_syst__1.khs[j - 1];
		}
		s_wsle(&io___294);
		do_lio(&c__9, &c__1, "Best  <dy**2>=", (ftnlen)14);
		do_lio(&c__5, &c__1, (char *)&sumd, (ftnlen)sizeof(doublereal)
			);
		do_lio(&c__9, &c__1, " min<dy**2>=", (ftnlen)12);
		do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(
			doublereal));
		e_wsle();
		s_wsle(&io___295);
		do_lio(&c__9, &c__1, "   at iseed=", (ftnlen)12);
		do_lio(&c__3, &c__1, (char *)&iseed, (ftnlen)sizeof(integer));
		do_lio(&c__9, &c__1, " scale=", (ftnlen)7);
		do_lio(&c__5, &c__1, (char *)&scale, (ftnlen)sizeof(
			doublereal));
		e_wsle();
		o__1.oerr = 0;
		o__1.ounit = 1;
		o__1.ofnmlen = 10;
		o__1.ofnm = "testme.dat";
		o__1.orl = 0;
		o__1.osta = 0;
		o__1.oacc = 0;
		o__1.ofm = 0;
		o__1.oblnk = 0;
		f_open(&o__1);
		i__2 = npoi;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    s_wsfe(&io___296);
		    do_fio(&c__1, (char *)&zd[i__ - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&yd[i__ - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&ym[i__ - 1], (ftnlen)sizeof(
			    doublereal));
		    e_wsfe();
		}
		cl__1.cerr = 0;
		cl__1.cunit = 1;
		cl__1.csta = 0;
		f_clos(&cl__1);
		o__1.oerr = 0;
		o__1.ounit = 1;
		o__1.ofnmlen = 7;
		o__1.ofnm = "opt.dat";
		o__1.orl = 0;
		o__1.osta = 0;
		o__1.oacc = 0;
		o__1.ofm = 0;
		o__1.oblnk = 0;
		f_open(&o__1);
		s_wsle(&io___297);
		do_lio(&c__9, &c__1, "# scale=", (ftnlen)8);
		do_lio(&c__5, &c__1, (char *)&opt_scl__, (ftnlen)sizeof(
			doublereal));
		do_lio(&c__9, &c__1, " sum_min=", (ftnlen)9);
		do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(
			doublereal));
		do_lio(&c__9, &c__1, " seed=", (ftnlen)6);
		do_lio(&c__3, &c__1, (char *)&iseed, (ftnlen)sizeof(integer));
		e_wsle();
		i__2 = earth_1.np;
		for (j = 1; j <= i__2; ++j) {
		    s_wsfe(&io___298);
		    do_fio(&c__1, (char *)&optsx[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optcx[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optsy[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optcy[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optam[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optkh[j - 1], (ftnlen)sizeof(
			    doublereal));
		    e_wsfe();
		}
		cl__1.cerr = 0;
		cl__1.cunit = 1;
		cl__1.csta = 0;
		f_clos(&cl__1);
	    }
	} else {
	    sumd_min__ = sumd;
	}
	if (iseed / 100 * 100 == iseed) {
	    s_wsle(&io___299);
	    do_lio(&c__9, &c__1, " <dy**2>=", (ftnlen)9);
	    do_lio(&c__5, &c__1, (char *)&sumd, (ftnlen)sizeof(doublereal));
	    do_lio(&c__9, &c__1, " min<dy**2>=", (ftnlen)12);
	    do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(
		    doublereal));
	    e_wsle();
	    s_wsle(&io___300);
	    do_lio(&c__9, &c__1, " iseed=", (ftnlen)7);
	    do_lio(&c__3, &c__1, (char *)&iseed, (ftnlen)sizeof(integer));
	    e_wsle();
	}
/*      call PREPARE_HARMONICS */
	prepare_systhar__();
/*      write(6,*)' back to beginning, iseed=',iseed */
    }
/* loop of iseed */
    s_wsle(&io___301);
    do_lio(&c__9, &c__1, " TESTS2 finished.  ", (ftnlen)19);
    e_wsle();
    o__1.oerr = 0;
    o__1.ounit = 1;
    o__1.ofnmlen = 7;
    o__1.ofnm = "opt.dat";
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    s_wsle(&io___302);
    do_lio(&c__9, &c__1, "# scale=", (ftnlen)8);
    do_lio(&c__5, &c__1, (char *)&opt_scl__, (ftnlen)sizeof(doublereal));
    do_lio(&c__9, &c__1, " sum_min=", (ftnlen)9);
    do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(doublereal));
    e_wsle();
    i__1 = earth_1.np;
    for (j = 1; j <= i__1; ++j) {
	s_wsfe(&io___303);
	do_fio(&c__1, (char *)&optsx[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optcx[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optsy[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optcy[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optam[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optkh[j - 1], (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    cl__1.cerr = 0;
    cl__1.cunit = 1;
    cl__1.csta = 0;
    f_clos(&cl__1);
    s_wsle(&io___304);
    do_lio(&c__9, &c__1, " TESTS2 finished, see file opt.dat for phases ", (
	    ftnlen)46);
    e_wsle();
    return 0;
} /* tests2_ */

/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* ####################################################################### */
/* 23456789*123456789*123456789*123456789*123456789*123456789*123456789*12 */
/* ####################################################################### */
/* Subroutine */ int tests3_()
{
    /* Format strings */
    static char fmt_101[] = "(20a)";
    static char fmt_100[] = "(5e13.5)";
    static char fmt_150[] = "(6e13.5)";

    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1;
    olist o__1;
    cllist cl__1;

    /* Builtin functions */
    integer s_wsle(), do_lio(), e_wsle(), f_open(), s_rsfe(), do_fio(), 
	    e_rsfe(), s_rsle(), e_rsle(), f_clos();
    double sin(), cos(), atan2();
    integer s_wsfe(), e_wsfe();

    /* Local variables */
    static doublereal s_cc__;
    extern /* Subroutine */ int prepare_systhar__();
    static doublereal s_yc__;
    static integer npoi;
    static doublereal s_ss__, sumd, s_ys__;
    static integer nseedmax;
    static doublereal sumd_min__;
    static integer i__, j;
    static doublereal scale, x;
    static integer iseed;
    static doublereal y, cosin, optam[250], optkh[250], optcx[250], optcy[250]
	    ;
    static integer j0;
    static doublereal sinus, sydym, optsx[250], optsy[250], zd[300], yd[300], 
	    ym[300];
    extern /* Subroutine */ int xy_pwk__();
    static char aaa[20];
    static doublereal opt_scl__, phase_j0__, sym2;

    /* Fortran I/O blocks */
    static cilist io___305 = { 0, 6, 0, 0, 0 };
    static cilist io___306 = { 0, 6, 0, 0, 0 };
    static cilist io___307 = { 0, 1, 0, fmt_101, 0 };
    static cilist io___309 = { 0, 1, 0, fmt_101, 0 };
    static cilist io___312 = { 0, 1, 0, 0, 0 };
    static cilist io___315 = { 0, 6, 0, 0, 0 };
    static cilist io___317 = { 0, 6, 0, 0, 0 };
    static cilist io___318 = { 0, 5, 0, 0, 0 };
    static cilist io___329 = { 0, 6, 0, 0, 0 };
    static cilist io___333 = { 0, 6, 0, 0, 0 };
    static cilist io___346 = { 0, 6, 0, 0, 0 };
    static cilist io___347 = { 0, 6, 0, 0, 0 };
    static cilist io___348 = { 0, 1, 0, fmt_100, 0 };
    static cilist io___349 = { 0, 1, 0, 0, 0 };
    static cilist io___350 = { 0, 1, 0, fmt_150, 0 };
    static cilist io___351 = { 0, 6, 0, 0, 0 };
    static cilist io___352 = { 0, 6, 0, 0, 0 };
    static cilist io___353 = { 0, 6, 0, 0, 0 };
    static cilist io___354 = { 0, 1, 0, 0, 0 };
    static cilist io___355 = { 0, 1, 0, fmt_150, 0 };
    static cilist io___356 = { 0, 6, 0, 0, 0 };


/* do some tests using generated harmonics */

    s_wsle(&io___305);
    do_lio(&c__9, &c__1, " ", (ftnlen)1);
    e_wsle();
    s_wsle(&io___306);
    do_lio(&c__9, &c__1, " Start of TESTS3", (ftnlen)16);
    e_wsle();
    o__1.oerr = 0;
    o__1.ounit = 1;
    o__1.ofnmlen = 11;
    o__1.ofnm = "slc_83v.dat";
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    s_rsfe(&io___307);
    do_fio(&c__1, aaa, (ftnlen)20);
    e_rsfe();
    s_rsfe(&io___309);
    do_fio(&c__1, aaa, (ftnlen)20);
    e_rsfe();
    npoi = 276;
    i__1 = npoi;
    for (i__ = 1; i__ <= i__1; ++i__) {
	s_rsle(&io___312);
	do_lio(&c__5, &c__1, (char *)&zd[i__ - 1], (ftnlen)sizeof(doublereal))
		;
	do_lio(&c__5, &c__1, (char *)&yd[i__ - 1], (ftnlen)sizeof(doublereal))
		;
	e_rsle();
	yd[i__ - 1] /= 1e3;
    }
    cl__1.cerr = 0;
    cl__1.cunit = 1;
    cl__1.csta = 0;
    f_clos(&cl__1);
    s_wsle(&io___315);
    do_lio(&c__9, &c__1, " File read", (ftnlen)10);
    e_wsle();
/* ======================================================= */
/* add systematic components */
/* we calculate sin(k_{j} s) at each step. This is stupid and can */
/* be avoided for the cost of two arrays (NH*NELMX). But this array can be */
/* very big, in our cases (50*300) = (15000) */
/* What is better? */

/*           do j=1,Np */
/*            sk(j)=sin(s*khs(j)) */
/*            ck(j)=cos(s*khs(j)) */
/*           end do */

/*         do j=1,Np */
/*           sinkx=sk(j)*dckxs(j)+ck(j)*dskxs(j) */
/*           sinky=sk(j)*dckys(j)+ck(j)*dskys(j) */
/*           x=x + ams(j) * t * sinkx */
/*           y=y + ams(j) * t * sinky */
/*         end do */

/* ======================================================== */
    nseedmax = 5000;
    s_wsle(&io___317);
    do_lio(&c__9, &c__1, "input Nseedmax=", (ftnlen)15);
    e_wsle();
    s_rsle(&io___318);
    do_lio(&c__3, &c__1, (char *)&nseedmax, (ftnlen)sizeof(integer));
    e_rsle();
    sumd_min__ = 0.;
    i__1 = nseedmax;
    for (iseed = 1; iseed <= i__1; ++iseed) {
/* find phase and amplitude a la fouriert */
	i__2 = earth_1.np;
	for (j0 = 1; j0 <= i__2; ++j0) {
	    s_ys__ = 0.;
	    s_yc__ = 0.;
	    s_ss__ = 0.;
	    s_cc__ = 0.;
	    i__3 = npoi;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		sinus = sin(zd[i__ - 1] * harmonics_syst__1.khs[j0 - 1]);
		cosin = cos(zd[i__ - 1] * harmonics_syst__1.khs[j0 - 1]);
		s_ys__ += yd[i__ - 1] * sinus;
		s_yc__ += yd[i__ - 1] * cosin;
/* Computing 2nd power */
		d__1 = sinus;
		s_ss__ += d__1 * d__1;
/* Computing 2nd power */
		d__1 = cosin;
		s_cc__ += d__1 * d__1;
	    }
	    phase_j0__ = atan2(s_yc__ * s_ss__, s_ys__ * s_cc__);
	    harmonics_syst__1.dckxs[j0 - 1] = cos(phase_j0__);
	    harmonics_syst__1.dskxs[j0 - 1] = sin(phase_j0__);
	    if (harmonics_syst__1.dckxs[j0 - 1] != 0.) {
		harmonics_syst__1.ams[j0 - 1] = s_ys__ / (s_ss__ * 
			harmonics_syst__1.dckxs[j0 - 1]) / earth_1.tmax;
	    } else {
		harmonics_syst__1.ams[j0 - 1] = s_yc__ / (s_cc__ * 
			harmonics_syst__1.dskxs[j0 - 1]) / earth_1.tmax;
	    }
	}
	s_wsle(&io___329);
	do_lio(&c__9, &c__1, "phases and amplitudes are redefined", (ftnlen)
		35);
	e_wsle();
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    xy_pwk__(&earth_1.tmax, &zd[i__ - 1], &x, &y);
	    ym[i__ - 1] = y;
	}
	s_wsle(&io___333);
	do_lio(&c__9, &c__1, " model ym generated with redefined phases and \
ampl.", (ftnlen)51);
	e_wsle();
	sym2 = 0.;
	sydym = 0.;
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing 2nd power */
	    d__1 = ym[i__ - 1];
	    sym2 += d__1 * d__1;
	    sydym += ym[i__ - 1] * yd[i__ - 1];
	}
	scale = sydym / sym2;
/*      do i=1,npoi */
/*       ym(i)=ym(i)*scale */
/*      end do */
/*      write(6,*)' model scaled by *',scale */
	sumd = 0.;
	i__2 = npoi;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing 2nd power */
	    d__1 = ym[i__ - 1] - yd[i__ - 1];
	    sumd += d__1 * d__1;
	}
	sumd /= npoi;
	if (sumd_min__ != 0.) {
	    if (sumd < sumd_min__) {
		sumd_min__ = sumd;
		opt_scl__ = scale;
		i__2 = earth_1.np;
		for (j = 1; j <= i__2; ++j) {
		    optsx[j - 1] = harmonics_syst__1.dskxs[j - 1];
		    optcx[j - 1] = harmonics_syst__1.dckxs[j - 1];
		    optsy[j - 1] = harmonics_syst__1.dskys[j - 1];
		    optcy[j - 1] = harmonics_syst__1.dckys[j - 1];
		    optam[j - 1] = harmonics_syst__1.ams[j - 1];
		    optkh[j - 1] = harmonics_syst__1.khs[j - 1];
		}
		s_wsle(&io___346);
		do_lio(&c__9, &c__1, "Best  <dy**2>=", (ftnlen)14);
		do_lio(&c__5, &c__1, (char *)&sumd, (ftnlen)sizeof(doublereal)
			);
		do_lio(&c__9, &c__1, " min<dy**2>=", (ftnlen)12);
		do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(
			doublereal));
		e_wsle();
		s_wsle(&io___347);
		do_lio(&c__9, &c__1, "   at iseed=", (ftnlen)12);
		do_lio(&c__3, &c__1, (char *)&iseed, (ftnlen)sizeof(integer));
		do_lio(&c__9, &c__1, " scale=", (ftnlen)7);
		do_lio(&c__5, &c__1, (char *)&scale, (ftnlen)sizeof(
			doublereal));
		e_wsle();
		o__1.oerr = 0;
		o__1.ounit = 1;
		o__1.ofnmlen = 10;
		o__1.ofnm = "testme.dat";
		o__1.orl = 0;
		o__1.osta = 0;
		o__1.oacc = 0;
		o__1.ofm = 0;
		o__1.oblnk = 0;
		f_open(&o__1);
		i__2 = npoi;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    s_wsfe(&io___348);
		    do_fio(&c__1, (char *)&zd[i__ - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&yd[i__ - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&ym[i__ - 1], (ftnlen)sizeof(
			    doublereal));
		    e_wsfe();
		}
		cl__1.cerr = 0;
		cl__1.cunit = 1;
		cl__1.csta = 0;
		f_clos(&cl__1);
		o__1.oerr = 0;
		o__1.ounit = 1;
		o__1.ofnmlen = 7;
		o__1.ofnm = "opt.dat";
		o__1.orl = 0;
		o__1.osta = 0;
		o__1.oacc = 0;
		o__1.ofm = 0;
		o__1.oblnk = 0;
		f_open(&o__1);
		s_wsle(&io___349);
		do_lio(&c__9, &c__1, "# scale=", (ftnlen)8);
		do_lio(&c__5, &c__1, (char *)&opt_scl__, (ftnlen)sizeof(
			doublereal));
		do_lio(&c__9, &c__1, " sum_min=", (ftnlen)9);
		do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(
			doublereal));
		do_lio(&c__9, &c__1, " seed=", (ftnlen)6);
		do_lio(&c__3, &c__1, (char *)&iseed, (ftnlen)sizeof(integer));
		e_wsle();
		i__2 = earth_1.np;
		for (j = 1; j <= i__2; ++j) {
		    s_wsfe(&io___350);
		    do_fio(&c__1, (char *)&optsx[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optcx[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optsy[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optcy[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optam[j - 1], (ftnlen)sizeof(
			    doublereal));
		    do_fio(&c__1, (char *)&optkh[j - 1], (ftnlen)sizeof(
			    doublereal));
		    e_wsfe();
		}
		cl__1.cerr = 0;
		cl__1.cunit = 1;
		cl__1.csta = 0;
		f_clos(&cl__1);
	    }
	} else {
	    sumd_min__ = sumd;
	}
	if (iseed / 100 * 100 == iseed) {
	    s_wsle(&io___351);
	    do_lio(&c__9, &c__1, " <dy**2>=", (ftnlen)9);
	    do_lio(&c__5, &c__1, (char *)&sumd, (ftnlen)sizeof(doublereal));
	    do_lio(&c__9, &c__1, " min<dy**2>=", (ftnlen)12);
	    do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(
		    doublereal));
	    e_wsle();
	    s_wsle(&io___352);
	    do_lio(&c__9, &c__1, " iseed=", (ftnlen)7);
	    do_lio(&c__3, &c__1, (char *)&iseed, (ftnlen)sizeof(integer));
	    e_wsle();
	}
/*      call PREPARE_HARMONICS */
/* will only regenerate k and not phases */
	filejustread_1.inewparams = 2;
	prepare_systhar__();
/*      write(6,*)' back to beginning, iseed=',iseed */
    }
/* loop of iseed */
    s_wsle(&io___353);
    do_lio(&c__9, &c__1, " TESTS3 finished.  ", (ftnlen)19);
    e_wsle();
    o__1.oerr = 0;
    o__1.ounit = 1;
    o__1.ofnmlen = 7;
    o__1.ofnm = "opt.dat";
    o__1.orl = 0;
    o__1.osta = 0;
    o__1.oacc = 0;
    o__1.ofm = 0;
    o__1.oblnk = 0;
    f_open(&o__1);
    s_wsle(&io___354);
    do_lio(&c__9, &c__1, "# scale=", (ftnlen)8);
    do_lio(&c__5, &c__1, (char *)&opt_scl__, (ftnlen)sizeof(doublereal));
    do_lio(&c__9, &c__1, " sum_min=", (ftnlen)9);
    do_lio(&c__5, &c__1, (char *)&sumd_min__, (ftnlen)sizeof(doublereal));
    e_wsle();
    i__1 = earth_1.np;
    for (j = 1; j <= i__1; ++j) {
	s_wsfe(&io___355);
	do_fio(&c__1, (char *)&optsx[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optcx[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optsy[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optcy[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optam[j - 1], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&optkh[j - 1], (ftnlen)sizeof(doublereal));
	e_wsfe();
    }
    cl__1.cerr = 0;
    cl__1.cunit = 1;
    cl__1.csta = 0;
    f_clos(&cl__1);
    s_wsle(&io___356);
    do_lio(&c__9, &c__1, " TESTS3 finished, see file opt.dat for phases ", (
	    ftnlen)46);
    e_wsle();
    return 0;
} /* tests3_ */

/* Main program alias */ int model_ () { MAIN__ (); }
