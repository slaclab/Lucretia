      MODULE XSIF_SIZE_PARS
C
C     Module which contains sizing parameters for arrays in the XSIF
C     parser.  Descended from DIMAD_SIZE_PARS.
C
C     AUTH: PT, 05-jan-2001
C
C     MOD:
C          15-dec-2003, PT:
C             add constants to define the lengths of element and
C             parameter names.
C		 19-may-2003, PT:
C			setup for dynamic allocation of data tables:  
C			MAXPOS, MAXELM, MAXPAR, MAXLST, MXCALL, MX_WAKEFILE are
C			converted to conventional integers, initial values and
C			increment values are established.
C          15-may-2001, PT:
C             increase MAXPOS and MAXLST to 49152 (48 kiloelts) each
C             to accomodate NLC2001 500 GeV deck with 90 cm structure.
C		 20-feb-2001, PT:
C			put back MXLINE parameter, accidentally deleted in move
C			from DIMAD_ to XSIF_ modules.
C
      IMPLICIT NONE
      SAVE

C========1=========2=========3=========4=========5=========6=========7=C

      INTEGER*4 MAXPOS    ! max # of MAD positions
      INTEGER*4 MAXELM    ! max # of MAD elements
      INTEGER*4 MAXPAR    ! max # of MAD element params
      INTEGER*4 MAXLST    ! MAD's max-list parameter
      INTEGER*4 MAXERR    !
	INTEGER*4 MXCALL	! mx # of nested CALL operations
	INTEGER*4 MX_WAKEFILE ! mx # of wakefield files
	INTEGER*4 MX_WAKE_Z ! max number of wakefield z-positions
	INTEGER*4 MX_SHARECON ! max number of shared parameters
	INTEGER*4 MXLINE

C      PARAMETER ( MAXPOS = 49152 )
C      PARAMETER ( MAXELM = 32768 )
C      PARAMETER ( MAXPAR = 40000 )
C      PARAMETER ( MAXLST = 49152 )
      PARAMETER ( MAXERR = 100   )
C	PARAMETER ( MXCALL = 32    )
C	PARAMETER ( MX_WAKEFILE = 16 )
	PARAMETER ( MX_WAKE_Z = 300)
	PARAMETER ( MX_SHARECON = 8)
	PARAMETER ( MXLINE = 1000 )

	INTEGER*4, PARAMETER :: MAXELM_INIT = 32768
	INTEGER*4, PARAMETER :: MAXPAR_INIT = 32768
	INTEGER*4, PARAMETER :: MAXLST_INIT = 32768
	INTEGER*4, PARAMETER :: MXCALL_INIT = 32
	INTEGER*4, PARAMETER :: MX_WAKE_INIT = 16

	INTEGER*4, PARAMETER :: MAXELM_STEP = 32768
	INTEGER*4, PARAMETER :: MAXPAR_STEP = 32768
	INTEGER*4, PARAMETER :: MAXLST_STEP = 32768
	INTEGER*4, PARAMETER :: MXCALL_STEP = 32
	INTEGER*4, PARAMETER :: MAX_WAKE_STEP = 16

C	Tolerances -- when there is this little space left
C	over in the tables, they will be EXPANDed.

	INTEGER*4, PARAMETER :: ELM_TOL = 128
	INTEGER*4, PARAMETER :: PAR_TOL = 256
	INTEGER*4, PARAMETER :: LST_TOL = 1024

	INTEGER*4 ETYPE_LENGTH
	INTEGER*4 ELABL_LENGTH

	PARAMETER ( ETYPE_LENGTH = 16  )
	PARAMETER ( ELABL_LENGTH = 24 )

	INTEGER*4, PARAMETER :: ENAME_LENGTH = 16
	INTEGER*4, PARAMETER :: PNAME_LENGTH = 16

C     while we're at it, it's nice to have a parameter which
C     describes the longest string variable that we'll need to
C     cope with.  For now, we'll just include the element and the
C     parameter lengths, since the XSIF dictionary functions never
C     really look at the TYPE or LABEL constructs.

      INTEGER*4, PARAMETER :: MAXCHARLEN = MAX( ENAME_LENGTH,
     &                                          PNAME_LENGTH )

      END MODULE XSIF_SIZE_PARS
