	SUBROUTINE RD_FILENAME( FNAME, COLSTART, COLEND, ERROR1 )
C
C	member of MAD INPUT PARSER
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C	Reads a file name (up to 80 characters) from the standard
C	input.  Returns the name (FNAME) and the beginning and ending
C	columns (COLSTART and COLEND) in KLINE if successful, or 
C	non-zero IERROR1 value if unsuccessful.  The filename must be
C	enclosed by double quotes and all on one line; COLSTART and
C	COLEND are the positions of the double quotes.
C
C	N.B.:  RD_FILENAME assumes that the calling routine (or someone
C	       up the call chain) has positioned the cursor (ICOL) at
C	       the next input character via RDNEXT.
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C	MOD:
C          16-may-2003, PT:
C             permit single quotes to identify filenames.  If
C             quotes are not found or lineskip occurs, return to
C             calling procedure with error status.
C          30-mar-2001, PT:
C             suppress leading blanks to satisfy unix.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C		 19-OCT-1998, PT:
C			use KTEXT_ORIG to preserve original upper-case and
C			lower-case status.
C
C	modules:
C
	USE XSIF_INOUT
C
	IMPLICIT NONE
	SAVE
C
C	argument declarations
C
	CHARACTER(LEN=80), INTENT(OUT) :: FNAME
	INTEGER*4, INTENT(OUT) :: COLSTART, COLEND
	LOGICAL*4, INTENT(OUT) :: ERROR1
C
C	local declarations
C
	INTEGER*4 STARTLINE
      INTEGER*4 BLANK_COUNTR
      CHARACTER*4 :: KBLANK = ' '
      LOGICAL*4 DOUBLE_QUOTES
      INTEGER*4 COL_LOOP
C
C	referenced functions
C

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                 C  O  D  E                                           C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C	locate the first double quote mark

	ERROR1 = .FALSE.
	COLSTART = ICOL
	STARTLINE = ILINE

	IF ( (KLINE(ICOL) .NE. ACHAR(34)) 
     &            .AND.
     &     (KLINE(ICOL). NE. ACHAR(39)) ) THEN
		ERROR1 = .TRUE.
		CALL RDFAIL
		WRITE(IECHO,920)
		WRITE(ISCRN,920)
          GOTO 9999
	ENDIF
      IF (KLINE(ICOL).EQ.ACHAR(34)) THEN
          DOUBLE_QUOTES = .TRUE.
      ELSE
          DOUBLE_QUOTES = .FALSE.
      ENDIF

C	locate the second quote mark

c	ICOL = ICOL + 1
c      IF (DOUBLE_QUOTES) THEN
c	    CALL RDFIND('"')
c      ELSE
c          CALL RDFIND("'")
c      ENDIF
c      IF (FATAL_READ_ERROR) GOTO 9999
c	COLEND = ICOL

      DO ICOL = ICOL+1,80
        IF ( (DOUBLE_QUOTES).AND.(KLINE(ICOL).EQ.'"') ) EXIT
        IF ( (.NOT. DOUBLE_QUOTES).AND.(KLINE(ICOL).EQ."'") ) EXIT
      ENDDO
      COLEND = ICOL

C	IF ( ILINE .NE. STARTLINE ) THEN
      IF (COLEND .EQ. 81) THEN
		CALL RDFAIL
		WRITE(IECHO,920)
		WRITE(ISCRN,920)
		ERROR1 = .TRUE.
          GOTO 9999
	ELSE
C
C     suppress leading blanks
C     
          DO BLANK_COUNTR = COLSTART+1,COLEND-1
              IF ( KTEXT_ORIG(BLANK_COUNTR:BLANK_COUNTR).NE.KBLANK) THEN
                  COLSTART = BLANK_COUNTR
                  EXIT
              ENDIF
          ENDDO

		FNAME = KTEXT_ORIG(COLSTART:COLEND-1)
	ENDIF

9999  IF ( FATAL_READ_ERROR ) ERROR1 = .TRUE.



	RETURN

C========1=========2=========3=========4=========5=========6=========7=C
 920	FORMAT('***ERROR*** FILENAME IN SINGLE OR ',
     &        'DOUBLE QUOTES EXPECTED.')
 910	FORMAT('***ERROR*** FILENAME MAY NOT BE SPREAD OVER MULTIPLE',/
     &	   '            INPUT LINES.')
C========1=========2=========3=========4=========5=========6=========7=C

	END