*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
*
*-----------------------------------------------------------------------
*                     Ground motion model @ LIAR
*                            Andrei Seryi
*                       Seryi@SLAC.Stanford.EDU                   
*                        Revision Feb 3, 2000
c Mod:
c       Apr-28-2002 A.S. added NxTFxMult_OF_SUPPORT (and y) in GM_TECHNICAL_NOISE
c           this will define if tech noise should be multiplied by transfer function 
c       Dec 28, 2001  A.S.
c           Added module GM_S_POSITION
c
*       Nov 29, 2000  L. Hendrickson
*           Add SAVE to ALLOCATABLE per F. Ostiguy.
C
************************************************************************
*
*                              modules
*
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
*
      MODULE GM_PARAMETERS
c      IMPLICIT NONE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE

* read parameters of P(w,k) from the file and put to the common block
* if there is no input file, a version that correspond to very noisy
* conditions such as in the HERA tunnel will be created

c      TYPE GM_CORRECTED_ATL
	  REAL*8 :: A = 5.00000E-19
	  REAL*8 :: B = 1.00000E-18
c	  END TYPE

c	  TYPE GM_PURE_ATL
c	  REAL*8 A
c	  END TYPE

c	  TYPE GM_WAVE
	  REAL*8 :: F1 = 1.40000E-01
	  REAL*8 :: A1 = 1.00000E-11
	  REAL*8 :: D1 = 5.00000E+00
	  REAL*8 :: V1 = -1.0000E+03
	  REAL*8 :: F2 = 2.50000E+00
	  REAL*8 :: A2 = 1.00000E-15
	  REAL*8 :: D2 = 1.50000E+00
	  REAL*8 :: V2 = -4.0000E+02
	  REAL*8 :: F3 = 5.00000E+01
	  REAL*8 :: A3 = 1.00000E-19
	  REAL*8 :: D3 = 1.50000E+00
	  REAL*8 :: V3 = -4.0000E+02
c	  END TYPE

c	  TYPE GM_SYSTEMATIC
	  REAL*8 :: Q1 = 2.70000E-02
	  REAL*8 :: RK1 = 2.00000E-03
	  REAL*8 :: RKK1 = 1.00000E+00
c	  END TYPE

	  REAL*8 :: TMAX = 3.00000E+06
	  REAL*8 :: TMIN = 1.00000E-02
	  REAL*8 :: SMAX = 1.50000E+04
	  REAL*8 :: SMIN = 1.00000E+00
	  INTEGER*4 :: NP =     50

        INTEGER*4 :: iwhat_syst = 1
        REAL*8 :: tau_syst = 30.0000E+00*365.*24.*3600.
        REAL*8 :: tgap_syst = 2.00000E+00*365.*24.*3600.

c A.S. 01/28/02
	  LOGICAL*4 :: TF_EXIST = .false.
	  LOGICAL*4 :: TECH_NOISE_EXIST = .false.


c	TYPE (GM_CORRECTED_ATL) our_atl
c	TYPE (GM_WAVE) our_waves(3)
c	TYPE (GM_SYSTEMATIC) our_syst

      END MODULE GM_PARAMETERS

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	MODULE GM_HARMONICS
	USE GM_PARAMETERS
        USE CONTROL_MOD
c      IMPLICIT NONE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE

cINTEGER*4, ALLOCATABLE :: matrix(:,:)REAL, ALLOCATABLE    :: vector(:)...
cALLOCATE(matrix(3,5),vector(-2:N+2))


      real*8, ALLOCATABLE :: am(:,:),wh(:),kh(:)
      real*8, ALLOCATABLE :: sw(:),cw(:)
      real*8, ALLOCATABLE :: swdt(:),cwdt(:)
      real*8, ALLOCATABLE :: dskx(:,:),dckx(:,:)
      real*8, ALLOCATABLE :: dsky(:,:),dcky(:,:)
	logical*4 HARMONICS_ALLOCATED
	data HARMONICS_ALLOCATED/.false./

      CONTAINS

      SUBROUTINE ALLOCATE_HARMONCS (NP,IERR)

      INTEGER*4 NP,IERR, I, IER
c	write(deflun,*)'inside allocate harm, press enter'
c	read(*,*)

	IF(.not.HARMONICS_ALLOCATED) THEN
	IERR=0
	ALLOCATE(am(NP,NP),wh(NP),kh(NP),STAT=IER)
	IERR=IERR+IER
	ALLOCATE(sw(NP),cw(NP),STAT=IER)
	IERR=IERR+IER
	ALLOCATE(swdt(NP),cwdt(NP),STAT=IER)
	IERR=IERR+IER
	ALLOCATE(dskx(NP,NP),dckx(NP,NP),dsky(NP,NP),
     $					 	dcky(NP,NP),STAT=IER)
	IERR=IERR+IER
	END IF

      IF (IERR.EQ.0) THEN
	HARMONICS_ALLOCATED=.true.
	ELSE
      WRITE(deflun,*) 'ERR>  Could not allocate GM Harmonics!'
      WRITE(deflun,*) 'STAT=', IERR
	HARMONICS_ALLOCATED=.false.
            END IF
       END SUBROUTINE

	END MODULE GM_HARMONICS

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	MODULE GM_HARMONICS_SYST
	USE GM_PARAMETERS
        USE CONTROL_MOD

c      IMPLICIT NONE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE      

	real*8, ALLOCATABLE ::  ams(:),khs(:)
      real*8, ALLOCATABLE ::  dskxs(:),dckxs(:),dskys(:),dckys(:)
      real*8, ALLOCATABLE ::  sk(:),ck(:)
	logical*4 HARMONICS_SYST_ALLOCATED
	data HARMONICS_SYST_ALLOCATED/.false./

      CONTAINS

      SUBROUTINE ALLOCATE_HARMONCS_SYST (NP,IERR)
      INTEGER*4 NP,IERR, I, IER

	IF(.not.HARMONICS_SYST_ALLOCATED) THEN
	IERR=0
	ALLOCATE(ams(NP),khs(NP),STAT=IER)
	IERR=IERR+IER
	ALLOCATE(dskxs(NP),dckxs(NP),dskys(NP),
     $					 	dckys(NP),STAT=IER)
	IERR=IERR+IER
	ALLOCATE(sk(NP),ck(NP),STAT=IER)
	IERR=IERR+IER
	END IF

      IF (IERR.EQ.0) THEN
	HARMONICS_SYST_ALLOCATED=.true.
	ELSE
      WRITE(deflun,*) 'ERR>  Could not allocate GM Harmonics Syst.!'
      WRITE(deflun,*) 'STAT=', IERR
	HARMONICS_SYST_ALLOCATED=.false.
            END IF
       END SUBROUTINE

	END MODULE GM_HARMONICS_SYST

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################


      MODULE GM_HARM_PREPARE
c      IMPLICIT NONE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE
      INTEGER*4 Nk,Nw
      real*8 G,kmin,kmax
	real*8 :: wmin=0.
	real*8 :: wmax=0.
	real*8 kmins,kmaxs
	real*8 told
c	real*8 difftmax
	real*8 dtold
	data told/0.0/,dtold/0.0/
      END MODULE GM_HARM_PREPARE
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################


      MODULE GM_S_POSITION
c      IMPLICIT NONE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE
      real*8 SBEG
	data SBEG/0.0/
	logical*4 FLIPS
	data FLIPS/.false./
      END MODULE GM_S_POSITION

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      MODULE TEST_GM_LINE
c      IMPLICIT NONE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE
      parameter (NELMX=300)
      real*8 ss(NELMX),xx(NELMX),yy(NELMX)
	INTEGER*4 Nelem
      END MODULE TEST_GM_LINE

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      MODULE GM_RANDOM_GEN
c      IMPLICIT NONE
       USE CONTROL_MOD
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE
	INTEGER*4 idum
      END MODULE GM_RANDOM_GEN

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	MODULE GM_ABSOLUTE_TIME
	USE GM_PARAMETERS
        USE CONTROL_MOD
      IMPLICIT NONE
c	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE

      real*8 GM_ABS_TIME
	data GM_ABS_TIME/0.0/

      CONTAINS
      SUBROUTINE SET_GM_ABSOLUTE_TIME (TIME,IERR)
	REAL*8 TIME
      INTEGER*4 IERR

	IF(TIME.ge.0.0) THEN
	IERR=0
	GM_ABS_TIME=TIME
	ELSE
	IERR=1
	END IF

      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Absolute TIME must be mositive!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
      END SUBROUTINE


      SUBROUTINE INCREASE_GM_ABSOLUTE_TIME (DTIME,IERR)
        USE CONTROL_MOD
	REAL*8 DTIME
      INTEGER*4 IERR

	IF(DTIME.gt.0.0) THEN
	IERR=0
	GM_ABS_TIME=GM_ABS_TIME+DTIME
	ELSE
	IERR=1
	END IF

      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  DTIME is expected to be >0!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
      END SUBROUTINE

	END MODULE GM_ABSOLUTE_TIME

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	MODULE GM_TRANSFUNCTION
	USE GM_PARAMETERS
        USE CONTROL_MOD
      USE LATTICE_MOD

	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE
c
        TYPE GM_TO_ELT_TF  
	   SEQUENCE
          REAL*8       F0_TF
          REAL*8       Q_TF
          REAL*8 , POINTER :: TF_RE(:)
          REAL*8 , POINTER :: TF_IM(:)
	    LOGICAL*4    tf_defined_by_file
	    INTEGER*4    ID_TF
        END TYPE

      allocatable :: GMTF(:)
        TYPE (GM_TO_ELT_TF) GMTF

      REAL*8, ALLOCATABLE, TARGET :: TF_RE_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: TF_IM_ALL(:,:)

	INTEGER*4, ALLOCATABLE :: ID_TFx_OF_SUPPORT(:)
	INTEGER*4, ALLOCATABLE :: ID_TFy_OF_SUPPORT(:)
	REAL*8,    ALLOCATABLE :: DS_AT_GROUND_OF_SUPPORT(:)


	INTEGER*4 NUMBER_OF_GMTF

      CONTAINS

      SUBROUTINE ALLOCATE_GM_TF(NGMTF_ALLOC)
	USE GM_PARAMETERS
      USE CONTROL_MOD
      INTEGER*4 NGMTF_ALLOC, IERR, I

	TF_EXIST= .true.

      IF (ALLOCATED(TF_RE_ALL)) DEALLOCATE (TF_RE_ALL)
      ALLOCATE (TF_RE_ALL(NP,NGMTF_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate TF_RE_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(TF_IM_ALL)) DEALLOCATE (TF_IM_ALL)
      ALLOCATE (TF_IM_ALL(NP,NGMTF_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate TF_IM_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF


      IF (ALLOCATED(GMTF)) DEALLOCATE (GMTF)
      ALLOCATE (GMTF(NGMTF_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate GMTF!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

          DO I = 1, NGMTF_ALLOC
            GMTF(I)%tf_defined_by_file     = .false.
            GMTF(I)%F0_TF          = 0.
            GMTF(I)%Q_TF           = 0.
            GMTF(I)%TF_RE       => TF_RE_ALL(1:NP,I)
            GMTF(I)%TF_IM       => TF_IM_ALL(1:NP,I)
	      GMTF(I)%ID_TF          = 0
          END DO

	IF (ALLOCATED(ID_TFx_OF_SUPPORT)) DEALLOCATE (ID_TFx_OF_SUPPORT)
      ALLOCATE (ID_TFx_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate ID_TFx_OF_SUPPORT !'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

	IF (ALLOCATED(ID_TFy_OF_SUPPORT)) DEALLOCATE (ID_TFy_OF_SUPPORT)
      ALLOCATE (ID_TFy_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate ID_TFy_OF_SUPPORT !'
      WRITE(deflun,*) 'STAT=', IERR
            END IF


	IF (ALLOCATED(DS_AT_GROUND_OF_SUPPORT)) 
     >                DEALLOCATE (DS_AT_GROUND_OF_SUPPORT)
      ALLOCATE (DS_AT_GROUND_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*)'ERR> Could not allocate DS_AT_GROUND_OF_SUPPORT !'
      WRITE(deflun,*)'STAT=', IERR
            END IF

      DO I = 1, NUM%SUPPORT
            ID_TFx_OF_SUPPORT(I) = 0
            ID_TFy_OF_SUPPORT(I) = 0
	      DS_AT_GROUND_OF_SUPPORT(I) = 0.0
	END DO

      END SUBROUTINE

	END MODULE GM_TRANSFUNCTION

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

	MODULE GM_TECHNICAL_NOISE
	USE GM_PARAMETERS
        USE CONTROL_MOD
      USE LATTICE_MOD

	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
      SAVE
c
        TYPE GM_TECH_NOISE  
	   SEQUENCE
	    REAL*8  F1 
	    REAL*8  A1 
	    REAL*8  D1 
	    REAL*8  F2 
	    REAL*8  A2 
	    REAL*8  D2 
	    REAL*8  F3 
	    REAL*8  A3 
	    REAL*8  D3 
          REAL*8 , POINTER :: AM_N(:)
          REAL*8 , POINTER :: WH_N(:)
          REAL*8 , POINTER :: SW_N(:)
          REAL*8 , POINTER :: CW_N(:)
          REAL*8 , POINTER :: SWDT_N(:)
          REAL*8 , POINTER :: CWDT_N(:)
          REAL*8 , POINTER :: DC_N(:)
          REAL*8 , POINTER :: DS_N(:)
	    INTEGER*4    ID_NOISE
	    INTEGER*4    DUMMY_ARRAY_ALIGNER
        END TYPE

      allocatable :: TCHN(:)
        TYPE (GM_TECH_NOISE) TCHN

      REAL*8, ALLOCATABLE, TARGET :: AM_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: WH_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: SW_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: CW_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: SWDT_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: CWDT_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: DC_N_ALL(:,:)
      REAL*8, ALLOCATABLE, TARGET :: DS_N_ALL(:,:)


	INTEGER*4, ALLOCATABLE :: ID_NOISEx_OF_SUPPORT(:)
	INTEGER*4, ALLOCATABLE :: ID_NOISEy_OF_SUPPORT(:)
C A.S. Apr-28-2002 : parameter which defines if noise 
C should be multiplied by TF or not
	INTEGER*4, ALLOCATABLE :: NxTFxMult_OF_SUPPORT(:)
	INTEGER*4, ALLOCATABLE :: NyTFyMult_OF_SUPPORT(:)


	INTEGER*4 NUMBER_OF_TECHNOISE

      CONTAINS

      SUBROUTINE ALLOCATE_GM_TECH_NOISE(NNOISE_ALLOC)
	USE GM_PARAMETERS
      USE CONTROL_MOD
      INTEGER*4 NNOISE_ALLOC, IERR, I

	TECH_NOISE_EXIST= .true.

      IF (ALLOCATED(AM_N_ALL)) DEALLOCATE (AM_N_ALL)
      ALLOCATE (AM_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate AM_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(WH_N_ALL)) DEALLOCATE (WH_N_ALL)
      ALLOCATE (WH_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate WH_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(SW_N_ALL)) DEALLOCATE (SW_N_ALL)
      ALLOCATE (SW_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate SW_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(CW_N_ALL)) DEALLOCATE (CW_N_ALL)
      ALLOCATE (CW_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate CW_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(SWDT_N_ALL)) DEALLOCATE (SWDT_N_ALL)
      ALLOCATE (SWDT_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate SWDT_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(CWDT_N_ALL)) DEALLOCATE (CWDT_N_ALL)
      ALLOCATE (CWDT_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate CWDT_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(DC_N_ALL)) DEALLOCATE (DC_N_ALL)
      ALLOCATE (DC_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate DC_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(DS_N_ALL)) DEALLOCATE (DS_N_ALL)
      ALLOCATE (DS_N_ALL(NP,NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate DS_N_ALL!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      IF (ALLOCATED(TCHN)) DEALLOCATE (TCHN)
      ALLOCATE (TCHN(NNOISE_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate TCHN!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

          DO I = 1, NNOISE_ALLOC
            TCHN(I)%F1     = 1.0
            TCHN(I)%F2     = 2.0
            TCHN(I)%F3     = 3.0
            TCHN(I)%A1     = 0.0
            TCHN(I)%A2     = 0.0
            TCHN(I)%A3     = 0.0
            TCHN(I)%D1     = 1.0
            TCHN(I)%D2     = 1.0
            TCHN(I)%D3     = 1.0
            TCHN(I)%AM_N       => AM_N_ALL(1:NP,I)
            TCHN(I)%WH_N       => WH_N_ALL(1:NP,I)
            TCHN(I)%SW_N       => SW_N_ALL(1:NP,I)
            TCHN(I)%CW_N       => CW_N_ALL(1:NP,I)
            TCHN(I)%SWDT_N     => SWDT_N_ALL(1:NP,I)
            TCHN(I)%CWDT_N     => CWDT_N_ALL(1:NP,I)
            TCHN(I)%DC_N       => DC_N_ALL(1:NP,I)
            TCHN(I)%DS_N       => DS_N_ALL(1:NP,I)
	      TCHN(I)%ID_NOISE   = 0
          END DO

	IF (ALLOCATED(ID_NOISEx_OF_SUPPORT)) 
     >        DEALLOCATE (ID_NOISEx_OF_SUPPORT)
      ALLOCATE (ID_NOISEx_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate ID_NOISEx_OF_SUPPORT !'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

	IF (ALLOCATED(ID_NOISEy_OF_SUPPORT)) 
     >        DEALLOCATE (ID_NOISEy_OF_SUPPORT)
      ALLOCATE (ID_NOISEy_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate ID_NOISEy_OF_SUPPORT !'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

	IF (ALLOCATED(NxTFxMult_OF_SUPPORT)) 
     >        DEALLOCATE (NxTFxMult_OF_SUPPORT)
      ALLOCATE (NxTFxMult_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate NxTFxMult_OF_SUPPORT !'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

	IF (ALLOCATED(NyTFyMult_OF_SUPPORT)) 
     >        DEALLOCATE (NyTFyMult_OF_SUPPORT)
      ALLOCATE (NyTFyMult_OF_SUPPORT(NUM%SUPPORT),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate NyTFyMult_OF_SUPPORT !'
      WRITE(deflun,*) 'STAT=', IERR
            END IF

      DO I = 1, NUM%SUPPORT
            ID_NOISEx_OF_SUPPORT(I) = 0
            ID_NOISEy_OF_SUPPORT(I) = 0
            NxTFxMult_OF_SUPPORT(I) = 0
            NyTFyMult_OF_SUPPORT(I) = 0
	END DO

      END SUBROUTINE

	END MODULE GM_TECHNICAL_NOISE

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
