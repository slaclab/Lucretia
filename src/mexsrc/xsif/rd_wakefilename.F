	SUBROUTINE RD_WAKEFILENAME( WAKETYPE, ERROR1 )
C
C	member of MAD INPUT PARSER
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C	reads the name of a wakefield file from the standard input,
C	allocates it in the appropriate table of wakefield filenames,
C	and replaces the filename in the KLINE/KTEXT line with the
C	integer which is the index of the filename in the filename
C	registry array.
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C     MOD:
C		 22-dec-2003, PT:
C			bugfix, per D. Sagan:  change NUM_LWAKE to NUM_TWAKE
C			in a spot where transverse wake name processing is
C			occurring.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C          24-jan-2000, PT:
C             minor bugfix for new char array pointer arrangement.
C          12-jan-2000, PT:
C             changes to support conversion of wakefield filenames
C             to character array pointers, and support use of $PATH
C             variable.  Correct bug related to improper placement
C             of error message due to too many types of wakefield
C             allocated.
C
C	modules:
C	
	USE XSIF_SIZE_PARS
	USE XSIF_INOUT
	USE XSIF_ELEMENTS
      USE XSIF_INTERFACES
C
	IMPLICIT NONE
	SAVE
C
C	argument declarations
C
	INTEGER*4, INTENT(IN)    :: WAKETYPE
	LOGICAL*4, INTENT(INOUT) :: ERROR1
C
C	local declarations
C
	CHARACTER*80 WFILNAME
      CHARACTER, POINTER :: WFNAM_PTR(:)
	CHARACTER*8  CHAR_FILENUM
	INTEGER*4 COLSTART, COLEND
	INTEGER*4 COUNTER
	INTEGER*4 FILENUM
	LOGICAL*4 FOUND
C	
C	referenced functions
C

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                 C  O  D  E                                           C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C	if all of the wakefields of the desired type are already
C	allocated, fail out

	ERROR1 = .FALSE.
	FOUND  = .FALSE.
	CHAR_FILENUM = '        '
C
C     this is not the right place to make this decision, since we
C     do not know if the structure/rf cavity is using a new wake
C     type or one which is already registered
C
c	IF (  ( WAKETYPE .EQ. LWAKE_FLAG   ) .AND. 
c     &	  ( NUM_LWAKE .EQ. MX_WAKEFILE )       ) THEN
c		WRITE(IECHO,910)
c		WRITE(ISCRN,910)
c		ERROR1 = .TRUE.
c		GOTO 9999
c	ENDIF
c
c	IF (  ( WAKETYPE .EQ. TWAKE_FLAG   ) .AND. 
c     &	  ( NUM_TWAKE .EQ. MX_WAKEFILE )       ) THEN
c		WRITE(IECHO,920)
c		WRITE(ISCRN,920)
c		ERROR1 = .TRUE.
c		GOTO 9999
c	ENDIF

C	read the name of the wakefield file

	CALL RD_FILENAME( WFILNAME, COLSTART, COLEND, ERROR1 )
      IF (FATAL_READ_ERROR) GOTO 9999
	IF ( ERROR1 ) THEN
		GOTO 9999
	ENDIF
C
C     expand the filename and convert to char array pointer
C
      NULLIFY ( WFNAM_PTR )
      CALL XPATH_EXPAND( WFILNAME, WFNAM_PTR )
C
C     if successful in expanding the pointer, continue, otherwise
C     abort
C
      IF ( ERROR ) THEN
          GOTO 9999
      ENDIF

C	compare the name to the ones already in the name
C	registry

	IF ( WAKETYPE .EQ. LWAKE_FLAG ) THEN
		IF ( NUM_LWAKE .NE. 0 ) THEN
			DO COUNTER = 1, NUM_LWAKE
C				IF ( WFILNAME .EQ. LWAKE_FILE(COUNTER) )
                  IF ( ARRCMP ( WFNAM_PTR, 
     &                 LWAKE_FILE(COUNTER)%FNAM_PTR ) )
     &		      THEN
					FOUND = .TRUE.
					FILENUM = COUNTER
				ENDIF
			ENDDO
		ENDIF
	ENDIF

	IF ( WAKETYPE .EQ. TWAKE_FLAG ) THEN
		IF ( NUM_TWAKE .NE. 0 ) THEN
			DO COUNTER = 1, NUM_TWAKE
C				IF ( WFILNAME .EQ. TWAKE_FILE(COUNTER) )
                  IF ( ARRCMP ( WFNAM_PTR, 
     &                 TWAKE_FILE(COUNTER)%FNAM_PTR ) )
     &		      THEN
					FOUND = .TRUE.
					FILENUM = COUNTER
				ENDIF
			ENDDO
		ENDIF
	ENDIF

C	if the filename is not yet in the registry, then
C	put the wakefield file into the filename registry, unless
C     it is full; in this case, send an error message and
C     go to exit

	IF ( .NOT. FOUND ) THEN

	    IF (  ( WAKETYPE .EQ. LWAKE_FLAG   ) .AND. 
     &	      ( NUM_LWAKE .EQ. MX_WAKEFILE )       ) THEN
		    WRITE(IECHO,910)
		    WRITE(ISCRN,910)
		    ERROR1 = .TRUE.
		    GOTO 9999
	    ENDIF

	    IF (  ( WAKETYPE .EQ. TWAKE_FLAG   ) .AND. 
     &	      ( NUM_TWAKE .EQ. MX_WAKEFILE )       ) THEN
		    WRITE(IECHO,920)
	    	WRITE(ISCRN,920)
		    ERROR1 = .TRUE.
		    GOTO 9999
	    ENDIF

		IF ( WAKETYPE .EQ. LWAKE_FLAG ) THEN
			NUM_LWAKE = NUM_LWAKE + 1
			FILENUM = NUM_LWAKE
C			LWAKE_FILE(NUM_LWAKE) = WFILNAME
c      PT, 24jan2000
c              LWAKE_FILE(NUM_LWAKE)%FNAM_PTR = WFNAM_PTR
              LWAKE_FILE(NUM_LWAKE)%FNAM_PTR => WFNAM_PTR
		ELSE
			NUM_TWAKE = NUM_TWAKE + 1
			FILENUM = NUM_TWAKE
C			TWAKE_FILE(NUM_TWAKE) = WFILNAME
c     PT, 24jan2000
c              TWAKE_FILE(NUM_LWAKE)%FNAM_PTR = WFNAM_PTR
              TWAKE_FILE(NUM_TWAKE)%FNAM_PTR => WFNAM_PTR
		ENDIF

	ENDIF

C	blank out the wakefield file name in the input line

	DO COUNTER = COLSTART, COLEND
		KLINE(COUNTER) = ACHAR(32)
	ENDDO

C	reposition the cursor in the read buffer

	ICOL = COLSTART

C	put the character representation of NUM_TWAKE or
C	NUM_LWAKE into the input line at COLSTART...

	CALL INT_TO_CHAR( FILENUM, CHAR_FILENUM, ERROR1 )

	DO COUNTER = 1,LEN_TRIM(CHAR_FILENUM)
		KLINE(ICOL+COUNTER-1) = CHAR_FILENUM(COUNTER:COUNTER)
	ENDDO



9999	CONTINUE
      IF ( FATAL_READ_ERROR) ERROR = .TRUE.
	RETURN

C========1=========2=========3=========4=========5=========6=========7=C
910	FORMAT('***ERROR*** MAX NUMBER LONGITUDINAL WAKEFIELD FILES',/
     &	   'EXCEEDED; PLEASE INCREASE MX_WAKEFILE IN',/
     &	   'DIMAD_SIZE_PARS.')
920	FORMAT('***ERROR*** MAX NUMBER TRANSVERSE WAKEFIELD FILES',/
     &	   'EXCEEDED; PLEASE INCREASE MX_WAKEFILE IN',/
     &	   'DIMAD_SIZE_PARS.')
C========1=========2=========3=========4=========5=========6=========7=C
	END