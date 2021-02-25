#include "fintrf.h"
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c+                                                             +
c+    LIAR: data TYPE for lattice definitions.            +
c+                                                             +
c+                                         RA, SLAC 9/1/95     +
c+       +
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      MODULE LATTICE_MOD
C
C     MOD:
C          06-jan-2004, PT:
C             support for longer element names.
C          14-APR-2003, PT:
C             add initialization of GIRDER and APERTURE values,
C             also of REF in x/ycors.
C          28-jan-2003, PT:
C             add DTILT to MAGNET structure.
C          20-jan-2003, PT:
C             add database for collimators.
C          27-Mar-2002, Linda Hendrickson
C             Add mzslice for markers.
C          12-mar-2002, PT:
C             give GIRDER parameter to BEND magnets.
C          17-jan-2002, PT:
C             add parameters to BEND magnets.  Improve allocation
C             of BEND elements.
C          07-jan-2002, PT:
C             add KLOSS parameter to RF structures.  Change 
C             QPOINTER in BPM structure to MAG_E_POINTER,
C             a more general pointer into the ELEMENT structure
C             at the MAGNET which contains the BPM.
C             Add BPOINTER for BPMs inside bends (actually 
C             gradient magnets).
C          07-Dec-2001, Linda Hendrickson (LJH):
C             Add STRUCTYPES_TYPE structure.
C          13-Nov-2001, Linda Hendrickson (LJH):
C             Rip out crab cavity support.
C             Make a new MAGNETTYPE which replaces QUADTYPE,
C             SEXTTYPE,OCTUTYPE.  Change MULTTYPE to use MAGNETTYPE.
C          10-oct-2001, PT:
C             support for preserving TYPE string information
C             for accelerator structures.
C             support for beam pitch and yaw in BPMs.  support
C             for pitch mode calculation in QUAD_STEER.
C          09-aug-2001, PT:
C             add GIRDER INTEGER*4 field to QUAD, SEXT, OCTU,
C             MULT, and BPM types.  Redefine SUPPORTs so that
C             POINTER is an F90 pointer to an allocatable
C             array.
C          02-jul-2001, PT:
c             add EPOINTER to MARKER type.
C		 27-jun-2001, PT:
C			added logical IS_INST to MARKERTYPE to indicate
C			that the MARKER was orignially a clas of INST in
C			XSIF.  Add NUM%INST to track # of instruments.
C             Add SEXTUPOLE, OCTUPOLE, MULTIPOLE types.
C             Store XSIF type index in ELEMENT structure.
C          15-may-2001, PT:
C             added STRUC_CAPTURE allocatable array for
C             RF_SWITCHOFF/RESTORE use; add variables KLOSS_RF
C             and CHARGE_RF.  Add DO_FIT logical to QSTEER_ARGS.
C             Add variable SILENT_TWISS.
C          29-Jan-2001, LJH for Brian McCandless, for parallel liar
C             Add ELEM_GLOBAL_SIZE, ELEM_BLOCK_SIZE.
C          03-jan-2001, PT:
C             add DXP and DYP (yaw and pitch) REAL*8 arrays to
C             structure data.
C          29-nov-2000, PT:
C             add TwissInit structure and EnergyInit Real*8
C             to capture initial conditions (which are otherwise
C             lost in CALC_TWISS).  Parameters to indicate
C             whether a MODE for twiss gathering is for upstream
C             or downstream.  Add N_EXTRA_BPMS and EXTRA_MODE
C             to QSTEER_ARGS structure.  Add parameters for 
C             use with EXTRA_MODE variables.  Add WEIGHT_EXTRA
C             to QSTEER_ARGS.
C          16-nov-2000, PT:
C             ALLOCATE_QSTEER_STUFF now takes as args
C             the sizes of the things to be allocated.
C          14-nov-2000, PT:
C             added support for QALIGN_NEW intermediate step
C             information handling
C          08-nov-2000, PT:
C             added support for use of QUAD_STEER subroutine
C             (QSTEER_ARGS structure, some dynamically-allocated
C             variables for use in QUAD_STEER)
C          07-NOV-2000, PT:
C             added the following data fields:
C            %EPOINTER to QUAD, BPM, BEND, STRUC, XCOR, YCOR
C             to point back at ELEMENT structure
C            %BPOINTER to QUAD to point at BPM structure
C            %QPOINTER to BPM to point at QUAD structure
C          14-oct-1998, PT:
C             made SUPPORT and MARKER allocatable (allocated
C             at end of COUNT_TRNS).
C          09-oct-1998, PT:
C             added APERTURE field to QUADTYPE, added
C             ALLOCATABLE TWISS2 structure.  IMPLICIT NONE
C             and SAVE statements.
C
      USE DEFINITIONS_MOD
      USE XSIF_SIZE_PARS

      IMPLICIT NONE
      SAVE
c
c++  Data TYPEs for definitions of the beamline elements
c
        TYPE BEAMLINE
      SEQUENCE
          CHARACTER(ENAME_LENGTH)  NAME
          REAL*8       S
          REAL*8       DS
          REAL*8       LDRIFT
          REAL*8       ENERGY
          LOGICAL*1    IS_QUAD
          LOGICAL*1    IS_BEND
          LOGICAL*1    IS_STRUC
          LOGICAL*1    IS_BPM
          LOGICAL*1    IS_XCOR
          LOGICAL*1    IS_YCOR
          LOGICAL*1    IS_MARKER
          LOGICAL*1    IS_SEXT
          LOGICAL*1    IS_OCTU
          LOGICAL*1    IS_MULT
          LOGICAL*2    IS_COLL
          LOGICAL*4    LDUMMY2
          mwpointer    POINTER
          INTEGER*4    XSIF_TYP_INDX
        END TYPE
      allocatable :: ELEMENT(:)
        TYPE (BEAMLINE) ELEMENT
c
c++  TYPEs for element types
c
        TYPE MAGNETTYPE        ! generic things that npole magnets have
      SEQUENCE
          REAL*8       LENGTH
          REAL*8       K
          REAL*8       APERTURE
          REAL*8       DK
          REAL*8       DX
          REAL*8       DY
          REAL*8       TILT, DTILT
          mwpointer    EPOINTER ! points into ELEMENT array
          mwpointer    BPOINTER ! points into BPM array
          mwpointer    GIRDER   ! points into SUPPORT array
          INTEGER*4    ISPARE 
        END TYPE

      allocatable :: QUAD(:)
        TYPE (MAGNETTYPE) QUAD

      allocatable :: SEXT(:)
        TYPE (MAGNETTYPE) SEXT

      allocatable :: OCTU(:)
        TYPE (MAGNETTYPE) OCTU
C
C     the Multipole type stores all of the K-values and their tilt angles
C     and permits a BPM if desired.  The multipole type includes all of the
C     items from other magnettype items plus a few.
C
       TYPE MULTTYPE
      SEQUENCE
          TYPE (MAGNETTYPE) MAG
          REAL*8      KL(0:20)
          REAL*8      Tn(0:20)
          REAL*8      SCALEFAC
          REAL*8      DKL(0:20)
       END TYPE
      ALLOCATABLE :: MULT(:)
      TYPE (MULTTYPE) MULT
C
c
        TYPE BENDTYPE
      SEQUENCE
          REAL*8       LENGTH
          REAL*8       B
          REAL*8       DB
          REAL*8       TILT, DTILT
          REAL*8       EDGE1
          REAL*8       EDGE2
          REAL*8       FRINGE1
          REAL*8       FRINGE2
          REAL*8       GAP1
          REAL*8       GAP2
          REAL*8       K1, K2
          REAL*8       H1, H2
          REAL*8       DX, DY
          mwpointer    EPOINTER ! points into ELEMENT array
          mwpointer    BPOINTER ! points into BPM array
          mwpointer    END_EPOINTER ! points at last slice in ELEMENT arry
          mwpointer    GIRDER      ! pointer into SUPPORT array
        END TYPE

      allocatable :: BEND(:)
        TYPE (BENDTYPE) BEND
c
        TYPE STRUCTYPE
          REAL*8       LENGTH
          REAL*8       RFPHASE
          REAL*8       ERRPHASE
          REAL*8       GRADIENT
          REAL*8       ERRGRAD
          REAL*8       DX(NPIECE_PER_STRUC_MAX)
          REAL*8       DY(NPIECE_PER_STRUC_MAX)
          REAL*8       DXP(NPIECE_PER_STRUC_MAX)
          REAL*8       DYP(NPIECE_PER_STRUC_MAX)
          REAL*8       RFBPM_X(NRFBPM_MAX)
          REAL*8       RFBPM_Y(NRFBPM_MAX)
          REAL*8       RESOL
          REAL*8       FRF
          REAL*8       KLOSS       !loss parameter (V/C/m)
          INTEGER*4    WAKEERRTYPE !shows what type of error wake is assigned to the stuc
          INTEGER*4    GIRDER
          INTEGER*4    TYPE
          INTEGER*4    ERRTYPE
          mwpointer    EPOINTER ! points into ELEMENT array
          CHARACTER(ETYPE_LENGTH) TYPE_STRING
        END TYPE

      allocatable :: STRUC(:)
        TYPE (STRUCTYPE) STRUC

        TYPE STRUCTYPES_TYPE    ! for types of structures
          LOGICAL*4 IS_INITED ! true if we have initialized this
                                  ! structure and the STRUC TYPE indices
                                  ! (This is done by a call to STRUC_CHECK_ALL)
             ! (number of types defined is in NUM%STRUC_TYPE)
          CHARACTER(ETYPE_LENGTH) TYPE_STRING(STRUC_TYPE_MAX)
                     !  Note from LJH :   I'm allowing a legal structure
                     !   type_string to be ' '.
        END TYPE
        TYPE (STRUCTYPES_TYPE) STRUCTYPES

c
        TYPE BPMTYPE
      SEQUENCE
          REAL*8       XBEAM(NB_MAX)
          REAL*8       YBEAM(NB_MAX)
          REAL*8       XEMIT(NB_MAX)
          REAL*8       YEMIT(NB_MAX)
          REAL*8       XEMITALL
          REAL*8       YEMITALL
          REAL*8       XLUMF
          REAL*8       YLUMF
          REAL*8       XBMAG(NB_MAX)
          REAL*8       YBMAG(NB_MAX)
          REAL*8       ESPREAD
          REAL*8       ENERGY
          REAL*8       DX
          REAL*8       DY
          REAL*8       TILT
          REAL*8       RESOL
          REAL*8       XREF(MAX_NREF, 2)
          REAL*8       YREF(MAX_NREF, 2)
          REAL*8       XSIZE
          REAL*8       YSIZE
          REAL*8       XPSIZE
          REAL*8       YPSIZE
          REAL*8       ETAX
          REAL*8       ETAY
          LOGICAL*4    REFSTAT(MAX_NREF)
          LOGICAL*4    STATUS
          INTEGER*4    NREF(MAX_NREF)
          INTEGER*4    GIRDER
          REAL*8       BEAMYAW(NB_MAX)     ! dx/dz of beam
          REAL*8       BEAMPITCH(NB_MAX)   ! dy/dz of beam
          mwpointer    EPOINTER      ! points into ELEMENT array
          mwpointer    MAG_E_POINTER ! points at element which contains
                                     ! the BPM (if not free standing)
        END TYPE

      allocatable :: BPM(:)
        TYPE (BPMTYPE) BPM
c
        CHARACTER*16   REF_BPMY(MAX_NREF)
c
        TYPE XCORRTYPE
      SEQUENCE
          REAL*8       LENGTH
          REAL*8       FIELD
          REAL*8       DFIELD
          REAL*8       TILT
          REAL*8       REF
          mwpointer    EPOINTER ! points into ELEMENT array
          INTEGER*4    ISPARE   ! alignment padding
        END TYPE

      allocatable :: XCOR(:)
        TYPE (XCORRTYPE) XCOR
c
        TYPE YCORRTYPE
      SEQUENCE
          REAL*8       LENGTH
          REAL*8       FIELD
          REAL*8       DFIELD
          REAL*8       TILT
          REAL*8       REF
          mwpointer    EPOINTER ! points into ELEMENT array
          INTEGER*4    ISPARE   ! alignment padding
        END TYPE

      allocatable :: YCOR(:)
        TYPE (XCORRTYPE) YCOR
c
        TYPE MARKERTYPE
      SEQUENCE
          REAL*8       XBEAM(NB_MAX)
          REAL*8       YBEAM(NB_MAX)
          REAL*8       XANGBEAM(NB_MAX)
          REAL*8       YANGBEAM(NB_MAX)
          REAL*8       XEMITBEAM(NB_MAX)
          REAL*8       YEMITBEAM(NB_MAX)
          REAL*8       XEMITALL
          REAL*8       YEMITALL
          REAL*8       ENERGYBEAM(NB_MAX)
          REAL*8       QBEAM(NB_MAX)
          REAL*8       X(NS_MAX)
          REAL*8       Y(NS_MAX)
          REAL*8       XANG(NS_MAX)
          REAL*8       YANG(NS_MAX)
          REAL*8       XEMIT(NS_MAX)
          REAL*8       YEMIT(NS_MAX)
          REAL*8       XLUMF(NB_MAX)
          REAL*8       YLUMF(NB_MAX)
          REAL*8       ENERGY(NS_MAX)
          REAL*8       Q(NS_MAX)
          REAL*8       XX(NS_MAX)
          REAL*8       XXANG(NS_MAX)
          REAL*8       XANGXANG(NS_MAX)
          REAL*8       YY(NS_MAX)
          REAL*8       YYANG(NS_MAX)
          REAL*8       YANGYANG(NS_MAX)
          REAL*8       XSIZE
          REAL*8       YSIZE
          REAL*8       XPSIZE
          REAL*8       YPSIZE
          REAL*8       XSIZESL(NS_MAX)
          REAL*8       YSIZESL(NS_MAX)
          REAL*8       XPSIZESL(NS_MAX)
          REAL*8       YPSIZESL(NS_MAX)
          REAL*8       MZSLICE(NS_MAX)  ! slice z pos at the marker
		LOGICAL*4    IS_INST
		mwpointer	 EPOINTER      ! for alignment
        END TYPE

        TYPE (MARKERTYPE), ALLOCATABLE :: MARKER(:)

        TYPE COLLTYPE
          SEQUENCE
              REAL*8      LENGTH
              REAL*8      XGAP
              REAL*8      YGAP
              REAL*8      RADLEN
              REAL*8      DX
              REAL*8      DY
              INTEGER*4   SHAPE
              mwpointer   EPOINTER
              INTEGER*4   GIRDER
              INTEGER*4   ISPARE
        END TYPE

        INTEGER*4, PARAMETER :: ECOLL = 0
        INTEGER*4, PARAMETER :: RCOLL = 1

        TYPE (COLLTYPE), ALLOCATABLE :: COLLIMATOR(:)
c
c++  Numbers of lattice elements
c
        TYPE ELEMENTNUMBERS
          SEQUENCE
          INTEGER*4    ELEM
          INTEGER*4    QUAD
          INTEGER*4    BEND
          INTEGER*4    PIECE_PER_STRUC
          INTEGER*4    PIECE
          INTEGER*4    STRUC
          INTEGER*4    GIRDER
          INTEGER*4    STRUC_TYPE
          INTEGER*4    STRUC_WAKEERRTYPE !number of error-wakes for structures
          INTEGER*4    BPM
          INTEGER*4    XCOR
          INTEGER*4    YCOR
          INTEGER*4    MARKER
          INTEGER*4    SUPPORT
		  INTEGER*4    INST
          INTEGER*4    SEXT
          INTEGER*4    OCTU
          INTEGER*4    MULT
          INTEGER*4    COLL
C     BCM ADDED for parallel PLIAR.
C          ELEM_GLOBAL_SIZE is the total number of elements in the lattice.
C          ELEM_BLOCK_SIZE is the number of elements that are grouped
C            together into a single pipeline stage.  The meaning of the other
C            fields is slightly changed in PLIAR.  They now represent sizes
C            on the current process.  For example, ELEM is the total number
C            of elements on the current process.
          INTEGER*4    ELEM_GLOBAL_SIZE
          INTEGER*4    ELEM_BLOCK_SIZE

        END TYPE
        TYPE (ELEMENTNUMBERS) NUM
c
c++  Twiss numbers for lattice
c
        TYPE TWISSTYPE
      SEQUENCE
          REAL*8       PSIX
          REAL*8       PSIY
          REAL*8       BETAX
          REAL*8       BETAY
          REAL*8       ALPHAX
          REAL*8       ALPHAY
        END TYPE
      allocatable :: TWISS(:)
      allocatable :: TWISSP(:)
      allocatable :: TWISS2(:)

        TYPE (TWISSTYPE) TWISS
        TYPE (TWISSTYPE) TWISSP
        TYPE (TWISSTYPE) TWISS2

        TYPE (TWISSTYPE) TWISSINIT
        REAL*8           ENERGYINIT

        INTEGER*4, PARAMETER :: MODE_UPSTREAM = 0
        INTEGER*4, PARAMETER :: MODE_DOWNSTREAM = 1
c
c++  Support TYPE
c
        TYPE SUPPORTTYPE
      SEQUENCE
          REAL*8       S1
          REAL*8       S2
          LOGICAL*4    FOR_QUAD
          LOGICAL*4    FOR_STRUC
          mwpointer , POINTER :: POINTER(:)
          INTEGER*4    NATTACH
	  INTEGER*4    myspare
        END TYPE

        TYPE (SUPPORTTYPE), ALLOCATABLE :: SUPPORT(:)


        TYPE QSTEERARG
      SEQUENCE
          INTEGER*4 IERR
          INTEGER*4 IXY
          INTEGER*4 FIRST
          INTEGER*4 LAST
          INTEGER*4 N_EXTRA_BPMS
          INTEGER*4 EXTRA_MODE
          LOGICAL*4 SET_CORR
          LOGICAL*4 SET_INIT
          LOGICAL*4 RESOLUTION
          LOGICAL*4 DO_FIT
          LOGICAL*4 PITCH_MODE
          INTEGER*4 ISPARE
          REAL*8    WEIGHT_BPM
          REAL*8    WEIGHT_MOVER
          REAL*8    WEIGHT_INIT
          REAL*8    WEIGHT_EXTRA
          REAL*8    RESOL
          REAL*8    MOVER_RESOL
          REAL*8    RESNRM
        END TYPE

      INTEGER*4, PARAMETER :: EXTRA_MODE_ABS = 0
      INTEGER*4, PARAMETER :: EXTRA_MODE_DELTA = 1

      INTEGER*4, PARAMETER :: MAX_CHARMAT = 2000

      REAL*8 , ALLOCATABLE :: FIT_WEIGHT(:)
      REAL*8 , ALLOCATABLE :: RESULT_VEC(:)
      REAL*8 , ALLOCATABLE :: CONSTRAINT_VEC(:)
      REAL*8 , ALLOCATABLE :: COV_MAT(:,:)
      REAL*8 , ALLOCATABLE :: CHAR_MAT(:,:)
      REAL*8 , ALLOCATABLE :: FIT_WORK(:)

C     variables for QALIGN_NEW intermediate step analysis

      INTEGER*4 :: NUM_QALIGN_EMITTANCE = 0
      REAL*8       SUM_EMIT, SUM_EMIT2
      REAL*8    :: MAX_EMIT = 0.D0
      REAL*8    :: MIN_EMIT = 0.D0
c
c     STRUC_CAPTURE captures gradient and phase of structures when
C     turned off in RF_SWITCHOFF, so that RF_RESTORE can restore
C     them.

      REAL*8, ALLOCATABLE :: STRUC_CAPTURE(:,:)

c     these variables capture the values of KLOSS and CHARGE which
C     are set in SET_RF, RF_SWITCHOFF, or RF_RESTORE.

      REAL*8 CHARGE_RF
      LOGICAL*4 :: SILENT_TWISS = .FALSE.
c


c
      CONTAINS
      SUBROUTINE ALLOCATE_TWISS (NTWISS_ALLOC)
C
C     MOD:
C
C          09-OCT-1998, PT:
C             add allocation for TWISS2 array.
C

      USE CONTROL_MOD

      INTEGER*4 NTWISS_ALLOC, IERR, I

      IF (ALLOCATED(TWISS)) DEALLOCATE (TWISS)
      IF (ALLOCATED(TWISSP)) DEALLOCATE (TWISSP)
      IF (ALLOCATED(TWISS2)) DEALLOCATE (TWISS2)
      ALLOCATE (TWISS(NTWISS_ALLOC),TWISSP(NTWISS_ALLOC),
     $        TWISS2(NTWISS_ALLOC),STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 
     &    'ERR>  Could not allocate TWISS, TWISS2, and TWISSP!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NTWISS_ALLOC
c
            TWISS(I)%PSIX      = 0.
            TWISS(I)%BETAX     = 0.
            TWISS(I)%ALPHAX    = 0.
            TWISS(I)%PSIY      = 0.
            TWISS(I)%BETAY     = 0.
            TWISS(I)%ALPHAY    = 0.
c
            TWISSP(I)%PSIX      = 0.
            TWISSP(I)%BETAX     = 0.
            TWISSP(I)%ALPHAX    = 0.
            TWISSP(I)%PSIY      = 0.
            TWISSP(I)%BETAY     = 0.
            TWISSP(I)%ALPHAY    = 0.
c
            TWISS2(I)%PSIX      = 0.
            TWISS2(I)%BETAX     = 0.
            TWISS2(I)%ALPHAX    = 0.
            TWISS2(I)%PSIY      = 0.
            TWISS2(I)%BETAY     = 0.
            TWISS2(I)%ALPHAY    = 0.
          END DO
      END SUBROUTINE


      SUBROUTINE ALLOCATE_COLL(NUMBER_COLL)

      USE CONTROL_MOD
      USE XSIF_ELEM_PARS

      INTEGER*4 NUMBER_COLL, IERR, I

      IF (ALLOCATED(COLLIMATOR)) DEALLOCATE(COLLIMATOR)
      ALLOCATE(COLLIMATOR(NUMBER_COLL),STAT=IERR)
      IF (IERR.NE.0) THEN
        WRITE(deflun,*) 'ERR>  Could not allocate COLLIMATOR!'
        WRITE(deflun,*) 'STAT=', IERR
        GOTO 9999
      END IF
      DO I = 1,NUMBER_COLL

          COLLIMATOR(I)%LENGTH = 0.D0
          COLLIMATOR(I)%XGAP   = 0.D0
          COLLIMATOR(I)%YGAP   = 0.D0
          COLLIMATOR(I)%DX     = 0.D0
          COLLIMATOR(I)%DY     = 0.D0
          COLLIMATOR(I)%RADLEN = 0.D0
          COLLIMATOR(I)%SHAPE  = RCOLL
          COLLIMATOR(I)%EPOINTER = 0
          COLLIMATOR(I)%GIRDER = 0
          COLLIMATOR(I)%ISPARE = 0

      ENDDO
9999  RETURN
      END SUBROUTINE

      
      SUBROUTINE ALLOCATE_ELEMENT (NELEMENT_ALLOC)

      USE CONTROL_MOD

      INTEGER*4 NELEMENT_ALLOC, IERR, I

      IF (ALLOCATED(ELEMENT)) DEALLOCATE (ELEMENT)
      ALLOCATE (ELEMENT(NELEMENT_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate ELEMENT!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NELEMENT_ALLOC
c
            ELEMENT(I)%NAME        = ' '
            ELEMENT(I)%S           = 0.
            ELEMENT(I)%DS          = 0.
            ELEMENT(I)%LDRIFT      = 0.
            ELEMENT(I)%IS_QUAD     = .false.
            ELEMENT(I)%IS_BEND     = .false.
            ELEMENT(I)%IS_STRUC    = .false.
            ELEMENT(I)%IS_BPM      = .false.
            ELEMENT(I)%IS_XCOR     = .false.
            ELEMENT(I)%IS_YCOR     = .false.
            ELEMENT(I)%IS_MARKER   = .false.
            ELEMENT(I)%IS_SEXT     = .FALSE.
            ELEMENT(I)%IS_OCTU     = .FALSE.
            ELEMENT(I)%IS_MULT     = .FALSE.
            ELEMENT(I)%IS_COLL     = .FALSE.
            ELEMENT(I)%POINTER     = 0
            ELEMENT(I)%ENERGY      = 0.
          END DO
      END SUBROUTINE

      SUBROUTINE ALLOCATE_QUAD (NQUAD_ALLOC)

      USE CONTROL_MOD

      INTEGER*4 NQUAD_ALLOC, IERR, I
      IF (ALLOCATED(QUAD)) DEALLOCATE (QUAD)
      ALLOCATE (QUAD(NQUAD_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate QUAD!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NQUAD_ALLOC
            QUAD(I)%LENGTH     = 0.
            QUAD(I)%K          = 0.
            QUAD(I)%DK         = 0.
            QUAD(I)%DX         = 0.
            QUAD(I)%DY         = 0.
            QUAD(I)%TILT    = 0.
            QUAD(I)%DTILT    = 0.
            QUAD(I)%APERTURE = 1.
            QUAD(I)%EPOINTER = 0
            QUAD(I)%BPOINTER = 0
            QUAD(I)%GIRDER   = 0
          END DO
      END SUBROUTINE

      SUBROUTINE ALLOCATE_SEXT (NSEXT_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 NSEXT_ALLOC, IERR, I

      IF (ALLOCATED(SEXT)) DEALLOCATE (SEXT)
      ALLOCATE (SEXT(NSEXT_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate SEXT!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NSEXT_ALLOC
            SEXT(I)%LENGTH     = 0.
            SEXT(I)%K          = 0.
            SEXT(I)%DK         = 0.
            SEXT(I)%DX         = 0.
            SEXT(I)%DY         = 0.
            SEXT(I)%APERTURE   = 1.
            SEXT(I)%TILT    = 0.
            SEXT(I)%DTILT    = 0.
            SEXT(I)%EPOINTER = 0
            SEXT(I)%BPOINTER = 0
            SEXT(I)%GIRDER   = 0
          END DO
      END SUBROUTINE

      SUBROUTINE ALLOCATE_OCTU (NOCTU_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 NOCTU_ALLOC, IERR, I

      IF (ALLOCATED(OCTU)) DEALLOCATE (OCTU)
      ALLOCATE (OCTU(NOCTU_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate OCTU!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NOCTU_ALLOC
            OCTU(I)%LENGTH     = 0.
            OCTU(I)%K          = 0.
            OCTU(I)%DK         = 0.
            OCTU(I)%DX         = 0.
            OCTU(I)%DY         = 0.
            OCTU(I)%APERTURE   = 1.
            OCTU(I)%TILT    = 0.
            OCTU(I)%DTILT    = 0.
            OCTU(I)%EPOINTER = 0
            OCTU(I)%BPOINTER = 0
            OCTU(I)%GIRDER   = 0
          END DO
      END SUBROUTINE

      SUBROUTINE ALLOCATE_MULT (NMULT_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 NMULT_ALLOC, IERR, I, IPOLE

      IF (ALLOCATED(MULT)) DEALLOCATE (MULT)
      ALLOCATE (MULT(NMULT_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate MULT!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NMULT_ALLOC
            MULT(I)%MAG%LENGTH     = 0.
            MULT(I)%MAG%DX         = 0.
            MULT(I)%MAG%DY         = 0.
            MULT(I)%MAG%TILT    = 0.
            MULT(I)%MAG%DTILT    = 0.
            MULT(I)%MAG%APERTURE = 1.
            MULT(I)%MAG%EPOINTER = 0
            MULT(I)%MAG%BPOINTER = 0
            MULT(I)%MAG%GIRDER   = 0
            MULT(I)%SCALEFAC = 0
            DO IPOLE = 1,20
              MULT(I)%KL(IPOLE) = 0.D0
              MULT(I)%TN(IPOLE) = 0.D0
              MULT(I)%DKL(IPOLE) = 0.D0
            END DO
          END DO
      END SUBROUTINE


      SUBROUTINE ALLOCATE_BEND (NBEND_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 NBEND_ALLOC, IERR, I

      IF (ALLOCATED(BEND)) DEALLOCATE (BEND)
      ALLOCATE (BEND(NBEND_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate BEND!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NBEND_ALLOC
            BEND(I)%LENGTH     = 0.
            BEND(I)%B          = 0.
            BEND(I)%DB         = 0.
            BEND(I)%TILT       = 0.
            BEND(I)%EDGE1      = 0.
            BEND(I)%EDGE2      = 0.
            BEND(I)%GAP1       = 0.
            BEND(I)%GAP2       = 0.
            BEND(I)%FRINGE1    = 0.
            BEND(I)%FRINGE2    = 0.
            BEND(I)%K1         = 0.
            BEND(I)%K2         = 0.
            BEND(I)%H1         = 0.
            BEND(I)%H2         = 0.
            BEND(I)%EPOINTER = 0
            BEND(I)%BPOINTER = 0
            bend(i)%dx = 0.
            bend(i)%dy = 0.
            bend(i)%dtilt = 0.
            bend(i)%end_epointer = 0
            bend(i)%girder = 0
          END DO
      END SUBROUTINE

      SUBROUTINE ALLOCATE_STRUC (NSTRUC_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 NSTRUC_ALLOC, IERR, I, J

      IF (ALLOCATED(STRUC)) DEALLOCATE (STRUC)
      ALLOCATE (STRUC(NSTRUC_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate STRUC!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NSTRUC_ALLOC
            STRUC(I)%LENGTH     = 0.
            STRUC(I)%RFPHASE    = 0.
            STRUC(I)%ERRPHASE   = 0.
            STRUC(I)%GRADIENT   = 0.
            STRUC(I)%ERRGRAD    = 0.
            DO J = 1, NPIECE_PER_STRUC_MAX
              STRUC(I)%DX(J)    = 0.
              STRUC(I)%DY(J)    = 0.
            END DO
            STRUC(I)%TYPE       = 0
            STRUC(I)%ERRTYPE    = 0
            STRUC(I)%RESOL      = 0
            STRUC(I)%RFBPM_X(1) = 0.
            STRUC(I)%RFBPM_Y(1) = 0.
            STRUC(I)%RFBPM_X(2) = 0.
            STRUC(I)%RFBPM_Y(2) = 0.
      STRUC(I)%FRF        = 0.
      STRUC(I)%EPOINTER = 0
      STRUC(I)%GIRDER = 0
          END DO
      END SUBROUTINE

      SUBROUTINE ALLOCATE_BPM (NBPM_ALLOC)

      USE CONTROL_MOD

      INTEGER*4 NBPM_ALLOC, IERR, I, J

      IF (ALLOCATED(BPM)) DEALLOCATE (BPM)
      ALLOCATE (BPM(NBPM_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate BPM!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, NBPM_ALLOC
            DO J = 1, NB_MAX
              BPM(I)%XBEAM(J)   = 0.
              BPM(I)%YBEAM(J)   = 0.
              BPM(I)%XEMIT(J)   = 0.
              BPM(I)%YEMIT(J)   = 0.
              BPM(I)%XEMITALL   = 0.
              BPM(I)%YEMITALL   = 0.
              BPM(I)%XLUMF      = 0.
              BPM(I)%YLUMF      = 0.
              BPM(I)%XBMAG(J)   = 0.
              BPM(I)%YBMAG(J)   = 0.
            END DO
            BPM(I)%DX           = 0.
            BPM(I)%DY           = 0.
            BPM(I)%TILT         = 0.
            BPM(I)%RESOL        = 0.
            BPM(I)%ENERGY       = 0.
            BPM(I)%ESPREAD      = 0.
            BPM(I)%STATUS       = .true.
            BPM(I)%XSIZE        = 0.
            BPM(I)%YSIZE        = 0.
            BPM(I)%XPSIZE       = 0.
            BPM(I)%YPSIZE       = 0.
            BPM(I)%ETAX         = 0.
            BPM(I)%ETAY         = 0.
            BPM(I)%EPOINTER = 0
            BPM(I)%MAG_E_POINTER = 0
            BPM(I)%GIRDER = 0
            DO J = 1, MAX_NREF
              BPM(I)%NREF(J)    = 0.
              BPM(I)%XREF(j,1)  = 0.
              BPM(I)%XREF(j,2)  = 0.
              BPM(I)%YREF(j,1)  = 0.
              BPM(I)%YREF(j,2)  = 0.
              BPM(I)%REFSTAT(j) = .true.
              REF_BPMY(j)       = ' '
            END DO
          END DO
      END SUBROUTINE
C
      SUBROUTINE ALLOCATE_XCOR (XCOR_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 XCOR_ALLOC, IERR, I

      IF (ALLOCATED(XCOR)) DEALLOCATE (XCOR)

      ALLOCATE (XCOR(XCOR_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate XCOR!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, XCOR_ALLOC
            XCOR(I)%LENGTH    = 0.
            XCOR(I)%FIELD     = 0.
            XCOR(I)%DFIELD    = 0.
            XCOR(I)%TILT      = 0.
            XCOR(I)%REF       = 0.
            XCOR(I)%EPOINTER = 0
          END DO
      END SUBROUTINE
C
      SUBROUTINE ALLOCATE_YCOR (YCOR_ALLOC)

      USE CONTROL_MOD
      INTEGER*4 YCOR_ALLOC, IERR, I

      IF (ALLOCATED(YCOR)) DEALLOCATE (YCOR)
      ALLOCATE (YCOR(YCOR_ALLOC),
     $        STAT=IERR)
      IF (IERR.NE.0) THEN
      WRITE(deflun,*) 'ERR>  Could not allocate YCOR!'
      WRITE(deflun,*) 'STAT=', IERR
            END IF
          DO I = 1, YCOR_ALLOC
            YCOR(I)%LENGTH    = 0.
            YCOR(I)%FIELD     = 0.
            YCOR(I)%DFIELD    = 0.
            YCOR(I)%TILT      = 0.
            YCOR(I)%REF       = 0.
            YCOR(I)%EPOINTER = 0
          END DO
      END SUBROUTINE
C

	SUBROUTINE ALLOCATE_MARKER( MARKER_ALLOC )

        USE CONTROL_MOD
	INTEGER*4 MARKER_ALLOC, IERR, I, J

	IF (ALLOCATED(MARKER)) DEALLOCATE(MARKER)
	ALLOCATE (MARKER(MARKER_ALLOC),
     &		STAT=IERR)
	IF (IERR.NE.0) THEN
		WRITE(deflun,*)'ERR> Could not allocate MARKER!'
		WRITE(deflun,*)'STAT=',IERR
	ENDIF

	DO I=1,MARKER_ALLOC
		DO J=1,NS_MAX

			MARKER(I)%X(J) = 0.D0
			MARKER(I)%Y(J) = 0.D0
			MARKER(I)%XANG(J) = 0.D0
			MARKER(I)%YANG(J) = 0.D0
			MARKER(I)%XEMIT(J) = 0.D0
			MARKER(I)%YEMIT(J) = 0.D0
			MARKER(I)%ENERGY(J) = 0.D0

		ENDDO
          MARKER(I)%IS_INST = .FALSE.
	ENDDO
	END SUBROUTINE

	SUBROUTINE ALLOCATE_SUPPORT( SUPPORT_ALLOC )

        USE CONTROL_MOD
	INTEGER*4 SUPPORT_ALLOC, IERR, I, J

	IF (ALLOCATED(SUPPORT)) DEALLOCATE(SUPPORT)
	ALLOCATE (SUPPORT(SUPPORT_ALLOC),
     &		STAT=IERR)
	IF (IERR.NE.0) THEN
		WRITE(deflun,*)'ERR> Could not allocate SUPPORT!'
		WRITE(deflun,*)'STAT=',IERR
	ENDIF

	DO I=1,SUPPORT_ALLOC

		SUPPORT(I)%S1 = 0.D0
		SUPPORT(I)%S2 = 0.D0
		SUPPORT(I)%FOR_QUAD = .FALSE.
		SUPPORT(I)%FOR_STRUC = .FALSE.
		SUPPORT(I)%NATTACH = 0
	
	ENDDO

	END SUBROUTINE

      SUBROUTINE ALLOCATE_QSTEER_STUFF( NROW, NCOL, IERR )

      USE CONTROL_MOD
      IMPLICIT NONE

      INTEGER*4 NROW, NCOL, IERR

      IERR = 0

      IF ( ALLOCATED( FIT_WEIGHT ) ) THEN
          DEALLOCATE( FIT_WEIGHT,
     &                RESULT_VEC, 
     &                CONSTRAINT_VEC,
     &                COV_MAT,
     &                CHAR_MAT,
     &                FIT_WORK          )
      ENDIF

      ALLOCATE( FIT_WEIGHT( NROW ), 
     &          RESULT_VEC( NCOL ),
     &          CONSTRAINT_VEC( NROW ),
     &          COV_MAT( NROW , NCOL ),
     &          CHAR_MAT( NROW , NCOL ),
     &          FIT_WORK( NROW ), STAT = IERR       )

      IF ( IERR .NE. 0 ) THEN
          WRITE(deflun,*)'ERR> Could not allocate QSTEER stuff!'
          WRITE(deflun,*)'STAT = ',IERR
          GOTO 9999
      ENDIF

9999  RETURN

      END SUBROUTINE

      SUBROUTINE SUPPORT_PTR_ASSOC( IERRFLAG )
C
C     associates an appropriate INTEGER*4 array with each SUPPORT's
C     POINTER pointer.
C
      USE CONTROL_MOD
      IMPLICIT NONE
      SAVE

      INTEGER*4  IERRFLAG
      mwpointer, POINTER :: PTR_TARG(:)      
      INTEGER*4 SUPP_CTR

      IF (.NOT. ALLOCATED(SUPPORT) ) THEN

          WRITE(deflun,*)
     &        'ERR> SUPPORT structure not allocated in ',
     &        'SUPPORT_PTR_ASSOC!'
          IERRFLAG = -1
          GOTO 9999

      ENDIF

C     loop over supports

      DO SUPP_CTR = 1,NUM%SUPPORT

          ALLOCATE(PTR_TARG(SUPPORT(SUPP_CTR)%NATTACH+1),
     &             STAT = IERRFLAG)
          IF (IERRFLAG .NE. 0) THEN

              WRITE(deflun,*)
     &            'ERR> Unable to allocate POINTER array for ',
     &            ' SUPPORT number ',SUPP_CTR,' in ',
     &            'SUPPORT_PTR_ASSOC!'
              GOTO 9999

          ENDIF

C         perform the association; we will use the last entry in
C         POINTER to keep track of how many of the slots in POINTER
C         are in use.

          SUPPORT(SUPP_CTR)%POINTER => PTR_TARG
          NULLIFY(PTR_TARG)
          SUPPORT(SUPP_CTR)%POINTER(SUPPORT(SUPP_CTR)%NATTACH+1) = 0

      END DO      

9999  RETURN
      END SUBROUTINE

      END MODULE LATTICE_MOD
