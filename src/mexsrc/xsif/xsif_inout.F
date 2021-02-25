      MODULE XSIF_INOUT
C
C     Data and parameters related to I/O.  Descended from DIMAD_INOUT.
C
C     Auth: PT, 05-jan-2001
C
C     Mod:
C		 08-dec-2003, PT:
C			add logical WHITESPACE_SKIPPED so that RDNEXT can
C			signal to the calling routine whether it had to skip
C			white space to get to the next character or not.
C		 19-may-2003, PT:
C			make IO_UNIT a POINTER to an integer array.  Add
C			XSIF_BADALLOC to signal failure of memory allocation.
C			Add FATAL_ALLOC_ERROR.
C          16-may-2003, PT:
C             support for xsif open stack data structures and for
C             use of the new CALL, FILENAME=filename syntax for
C             CALL statement.  Add USE_BEAM_OR_BETA0 flag to
C             indicate whether a USE statement was followed by a 
C             beamline name or a BEAM/BETA0 name.
C          20-mar-2001, PT:
C             Move XSIF_VERSION and XSIF_VERSION_DATE initialization
C             to XSIF_HEADER, so that every time we change version
C             date we do not need to recompile everything!
C          02-MAR-2001, PT:
C             changed XSIF error-returns to parameters; added logical
C             XSIF_STOP, which indicates whether some command wants
C             the parser to exit.
C          31-jan-2001, PT:
C             added logical FATAL_READ_ERROR which is set when such
C             an error occurs; error message XSIF_FATALREAD, to signal
C             same to calling routine.
C          08-JAN-2001, PT:
C             add XSIF_PAR_NODEFINE, error value for undefined parameters
C             in the input stream.
C          05-JAN-2001, PT:
C             logical switch NLC_STD, which indicates whether
C             warnings for non-standard elts/keywds should be
C             issued.  Logical XUSE_FROM_FILE indicates that
C             a USE command was issued in the XSIF deck (rather than
C             being issued from a subroutine).
C
C     Modules:
C
      USE XSIF_SIZE_PARS

      IMPLICIT NONE
      SAVE
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C     MAD page header information

      CHARACTER*80  KTIT
      CHARACTER*16 :: XSIF_VERSION != '1.2 BETA'
      CHARACTER*11 :: XSIF_VERS_DATE != '20-Mar-2001'
      CHARACTER*11  KDATE
      CHARACTER*8   KTIME

C     logical units, pointers, counters for MAD I/O

      INTEGER*4 IDATA
      INTEGER*4 IPRNT
      INTEGER*4 IECHO
	INTEGER*4 :: ISCRN = 6
      INTEGER*4 ILINE
      INTEGER*4 ILCOM
      INTEGER*4 ICOL
      INTEGER*4 IMARK
      INTEGER*4 NWARN
      INTEGER*4 NFAIL

C     flags for I/O

      LOGICAL*4 SCAN
      LOGICAL*4 ERROR
      LOGICAL*4 SKIP
      LOGICAL*4 ENDFIL
      LOGICAL*4 INTER
      LOGICAL*4 :: DIMAT_STOP = .FALSE.

C     MAD input buffer

      CHARACTER*1  KLINE(81)
      CHARACTER*80 KTEXT
	CHARACTER*1  KLINE_ORIG(81)
	CHARACTER*80 KTEXT_ORIG

      EQUIVALENCE (KTEXT,KLINE(1))

	EQUIVALENCE (KTEXT_ORIG,KLINE_ORIG(1))

C     echo/noecho flag

      INTEGER*4 NOECHO

C     expression encoding/decoding

      INTEGER*4 LEV
      INTEGER*4 IOP(50), IVAL(50)

	INTEGER*4, POINTER :: IO_UNIT( : )
	INTEGER*4 NUM_CALL

C=====================================================================C

C     MAD input parser machine structure flags

      LOGICAL*4 INVAL
      LOGICAL*4 NEWCON
      LOGICAL*4 NEWPAR
      LOGICAL*4 PERI
      LOGICAL*4 STABX
      LOGICAL*4 STABZ
      LOGICAL*4 SYMM

      CHARACTER, POINTER :: PATH_PTR(:)
      CHARACTER, TARGET :: PATH_LCL(1) = ( / '.' / )

      CHARACTER*26 :: LOTOUP = 'abcdefghijklmnopqrstuvwxyz'                                                  
      CHARACTER*1 :: UPTOLO(26) 
	CHARACTER*26 :: UPLO = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	
	EQUIVALENCE(UPLO(1:1),UPTOLO(1))                                                 

c	error flags for XSIF_PARSE

	  INTEGER*4, parameter  ::  XSIF_PARSE_NOLINE = -191
	  INTEGER*4, parameter  ::  XSIF_PARSE_ERROR  = -193
	  INTEGER*4, parameter  ::  XSIF_PARSE_NOOPEN = -195
c     PT, 08-jan-2001
        INTEGER*4, parameter  ::  XSIF_PAR_NODEFINE = -197
C     end new block 08-jan-2001
c     PT, 31-jan-2001
        INTEGER*4, parameter  ::  XSIF_FATALREAD    = -199
	  INTEGER*4, PARAMETER  ::  XSIF_BADALLOC     = -201
        LOGICAL*4  ::  FATAL_READ_ERROR = .FALSE.
	  LOGICAL*4  ::  FATAL_ALLOC_ERROR = .FALSE.
	  LOGICAL*4  ::  XSIF_STOP = .FALSE.
C     end new block 08-jan-2001

c     NLC-related warnings flag

        LOGICAL*4 :: NLC_STD = .TRUE.
        LOGICAL*4 XUSE_FROM_FILE

C     definitions related to files OPENed or CALLed by XSIF

      TYPE XSIF_FILETYPE

          INTEGER*4 UNIT_NUMBER
          CHARACTER, POINTER :: FILE_NAME(:)
          TYPE (XSIF_FILETYPE), POINTER :: NEXT_FILE

      END TYPE XSIF_FILETYPE

      TYPE (XSIF_FILETYPE), POINTER :: XSIF_OPEN_STACK_HEAD
      TYPE (XSIF_FILETYPE), POINTER :: XSIF_OPEN_STACK_TAIL

      INTEGER*4 :: XCALL_UNITNO = 5001

      LOGICAL*4 :: USE_BEAM_OR_BETA0

	LOGICAL*4 :: WHITESPACE_SKIPPED

      END MODULE XSIF_INOUT
