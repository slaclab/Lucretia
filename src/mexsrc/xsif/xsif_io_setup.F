      INTEGER*4 FUNCTION XSIF_IO_SETUP( DECKFILE, ERRFILE, STREAMFILE,
     &                                  DECKUNIT, ERRUNIT, STREAMUNIT,
     &                                  OUTLUN, XSIFECHO, NLCWARN      )
C
C     This function sets up the I/O for the XSIF parser.
C
C     The XSIF parser requires an input file which contains the deck,
C         an error file that it sends messages to, and a stream file that
C         it echoes the input file to.  This subroutine takes the file
C         names for the deck, error, and stream, and opens them with the
C         unit numbers specified by the user.  It then uses XSIF's global
C         variable controller to set those values for the parser.  This
C         function also uses XSIFECHO and NLCWARN to tell the parser whether
C         we want the input deck echoed to the errfile, streamfile and
C         screen, and whether we want warnings about non-NLC standard stuff.
C
C     AUTH: PT, 05-JAN-2001
C
C     MOD:
C          14-MAR-2003, PT:
C             update to use XOPEN_STACK_MANAGE to open files.
C          02-MAR-2001, PT:
C             move FATAL_READ_ERROR initialization to CLEAR so that
C             all XSIF-required initializtion is centralized.
C          31-JAN-2001, PT:
C             initialize FATAL_READ_ERROR to FALSE so that if we call
C             xsif after a previous fatal read error that error is not
C             sticking around jamming up the works.
C
      USE XSIF_INOUT

      IMPLICIT NONE
      SAVE

C     argument declarations

      CHARACTER(*)    DECKFILE        ! input file with deck
      CHARACTER(*)    ERRFILE         ! output file for messages
      CHARACTER(*)    STREAMFILE      ! output file for deck echo

      INTEGER*4   DECKUNIT            ! unit number for deck
      INTEGER*4   ERRUNIT             ! unit number for messages
      INTEGER*4   STREAMUNIT          ! unit number for deck echo

      INTEGER*4   OUTLUN              ! unit number for warnings from this fn.
      LOGICAL*4   XSIFECHO            ! do or do not echo the deck
      LOGICAL*4   NLCWARN             ! do or do not issue standards warnings.

C     local declarations

      INTEGER*4   ERR_LCL             ! did something go wrong?

      CHARACTER*8 KDATE_TEMP          ! vars for time/date call
      CHARACTER*10 KTIME_TEMP
      CHARACTER*5 ZONE
      INTEGER*4 VALUES(8)
      LOGICAL*4 XSM_STAT

C     referenced functions

      LOGICAL     INTRAC
      LOGICAL*4   XOPEN_STACK_MANAGE

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

c
c     get the date and time
c
      CALL DATE_AND_TIME(KDATE_TEMP,KTIME_TEMP,ZONE,VALUES)
      CALL VMS_TIMEDATE(KDATE_TEMP,KTIME_TEMP,VALUES(2),
     &                  KDATE,KTIME                       )
      ERR_LCL = 0

      IF ( XSIFECHO ) THEN
          NOECHO = 0
      ELSE
          NOECHO = 1
      ENDIF

      IF ( NLCWARN ) THEN
          NLC_STD = .TRUE.
      ELSE
          NLC_STD = .FALSE.
      ENDIF
C
C     open the desired files to the desired unit numbers
C
C	open the XSIF file for input

c	OPEN( DECKUNIT, FILE=DECKFILE, STATUS='OLD',IOSTAT=ERR_LCL )
      XSM_STAT = XOPEN_STACK_MANAGE( DECKFILE, DECKUNIT, 'OLD' )

          IF (.NOT. XSM_STAT) THEN

C		IF ( ERR_LCL .NE. 0 ) THEN
			WRITE(OUTLUN,*)'*** ERROR *** Cannot open XSIF file:  ',
     &				DECKFILE
			ERR_LCL = XSIF_PARSE_NOOPEN
			GOTO 9999
		ENDIF

C	open the echo file for error messages

c	OPEN( ERRUNIT, FILE=ERRFILE, STATUS='REPLACE', IOSTAT=ERR_LCL )
      XSM_STAT = XOPEN_STACK_MANAGE( ERRFILE, ERRUNIT, 'REPLACE' )

C		IF ( ERR_LCL .NE. 0 ) THEN
          IF (.NOT. XSM_STAT) THEN
			WRITE(OUTLUN,*)'*** ERROR *** Cannot open XSIF file:  ',
     &				'XSIF.ERR'
			ERR_LCL = XSIF_PARSE_NOOPEN
			GOTO 9999
		ENDIF

C	open the stream file for echoing

c	OPEN( STREAMUNIT, FILE=STREAMFILE, 
C     &      STATUS='REPLACE', IOSTAT=ERR_LCL )
      XSM_STAT = XOPEN_STACK_MANAGE( STREAMFILE, STREAMUNIT, 'REPLACE' )

C		IF ( ERR_LCL .NE. 0 ) THEN
          IF (.NOT. XSM_STAT) THEN
			WRITE(OUTLUN,*)'*** ERROR *** Cannot open XSIF file:  ',
     &				'XSIF.STR'
			ERR_LCL = XSIF_PARSE_NOOPEN
			GOTO 9999
		ENDIF

C	call RDINIT, to set up I/O commands

	CALL RDINIT( DECKUNIT, STREAMUNIT, ERRUNIT )

C     misc. variable initialization

	CALL CLEAR

      INTER = INTRAC()

C     set return value and exit

9999  XSIF_IO_SETUP = ERR_LCL

      RETURN
      END


