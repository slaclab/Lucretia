      MODULE XSIF_ELEMENTS
C
C     This module contains the data structures which hold information
C     on defined beamline elements and parameters.  It is descended
C     from DIMAD_ELEMENTS.
C
C     AUTH: PT, 05-JAN-2001
C
C     MOD:
C		 23-DEC-2003, PT:
C			add logicals BETA0_FROM_USE and BEAM_FROM_USE to
C			indicate that the user has selected a beam or twiss
C			statement, and that the parser should not override.
C          15-dec-2003, PT:
C             expand element names to 16 characters for compatibility
C             with more-modern MAD syntax.
C		 05-dec-2003, PT:
C			add ERRFLG and EALIGN allocatable arrays to support
C			addition of EALIGNs to the beamline which has been
C			expanded.
C		 19-may-2003, PT:
C			transform element data structures to POINTER type to
C			facilitate dynamic allocation.
C          16-may-2003, PT:
C             add IBEAM_PTR and IBETA0_PTR to point to the selected
C             BEAM and BETA0 definitions in the element list.
C          02-mar-2001, PT:
C             rename NELM to NELM_XSIF to avoid conflict with
C             DIMAD's NELM.
C          05-jan-2001, PT:
C             new global variable LINE_EXPANDED which indicates
C             whether a USE statement has executed.
C             
C
      USE XSIF_SIZE_PARS

      IMPLICIT NONE
      SAVE

C========1=========2=========3=========4=========5=========6=========7=C

C     name, type, and label of elements from MAD input parser
C	allocate to MAXELM

      CHARACTER(ENAME_LENGTH), POINTER ::  KELEM(:)        ! name
      CHARACTER(ETYPE_LENGTH), POINTER ::  KETYP(:)        ! type
      CHARACTER(ELABL_LENGTH), POINTER ::  KELABL(:)       ! label

C     local vars for getting type and label

      CHARACTER(ETYPE_LENGTH)  KTYPE
      CHARACTER(ELABL_LENGTH) KLABL

C     element counting (?); to MAXELM

      INTEGER*4, POINTER :: IECNT(:)

C     link table; second index for ILDAT should be 6
C	first index should be MXLIST

      INTEGER*4 IUSED
      INTEGER*4, POINTER :: ILDAT(:,:)

C     element data for MAD input parser;
C	second index for IEDAT should be 3
C	first index should be MAXELM

      INTEGER*4 IELEM1,IELEM2
      INTEGER*4, POINTER :: IETYP(:)
      INTEGER*4, POINTER :: IEDAT(:,:)
      INTEGER*4, POINTER :: IELIN(:)
      LOGICAL*4, POINTER :: ELEM_LOCKED(:)

C     parameter names in MAD input parser
C	allocate to MAXPAR

      CHARACTER(PNAME_LENGTH), POINTER :: KPARM(:)

C     parameter data and parameter values from MAD input parser
C	first index is MAXPAR, 2nd index for IPDAT is 2

      INTEGER*4 IPARM1,IPARM2
      INTEGER*4, POINTER :: IPTYP(:)
      INTEGER*4, POINTER :: IPDAT(:,:)
      INTEGER*4, POINTER :: IPLIN(:)

      INTEGER*4 IPLIST
	INTEGER*4, POINTER :: IPNEXT(:)

      REAL*8, POINTER :: PDATA(:)

C     MAD version of a beamline and period definition
C	allocate to MAXPOS

      INTEGER*4, ALLOCATABLE :: ITEM(:)
      LOGICAL*4, ALLOCATABLE :: PRTFLG(:)
	LOGICAL*4, ALLOCATABLE :: ERRFLG(:)
	REAL*8,    ALLOCATABLE :: ERRPTR(:) 

      INTEGER*4 IPERI
      INTEGER*4 IACT
      INTEGER*4 NSUP
      INTEGER*4 NELM_XSIF            ! number of MAD elts in line
      INTEGER*4 NFREE
      INTEGER*4 NPOS1
      INTEGER*4 NPOS2

C     units variable

      INTEGER*4 KFLAGU

C	wakefield filenames

      TYPE :: WAKEFILE
          CHARACTER, POINTER :: FNAM_PTR(:)
      END TYPE

      TYPE (WAKEFILE), POINTER :: LWAKE_FILE( : )
      TYPE (WAKEFILE), POINTER :: TWAKE_FILE( : )

C	flags to tell RD_WAKEFILE which one to read

	INTEGER*4 LWAKE_FLAG, TWAKE_FLAG
	PARAMETER ( LWAKE_FLAG = 1, TWAKE_FLAG = 2)

C	number of allocated wakefiles of each case

	INTEGER*4 NUM_LWAKE, NUM_TWAKE

C     NLC-standard indication

      LOGICAL*4 NLC_STANDARD
	INTEGER*4 IKEYW_GLOBAL

C     indication of beamline expansion

      LOGICAL*4 :: LINE_EXPANDED = .FALSE.

      INTEGER*4 :: IBEAM_PTR = 0
      INTEGER*4 :: IBETA0_PTR = 0
	LOGICAL*4 :: BEAM_FROM_USE = .FALSE.
	LOGICAL*4 :: BETA0_FROM_USE = .FALSE.

      END MODULE XSIF_ELEMENTS
