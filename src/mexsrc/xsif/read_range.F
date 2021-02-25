	SUBROUTINE READ_RANGE( TOKEN, LTOKEN, ELEMNO, INSTANCE, ERROR1 )
C
C	Takes the name of an element which was read by a RANGE= operation
C	in a SELECT statement (read by the SELECT subroutine) and returns
C	the index number of the element in the element data table.  Also
C	returns the instance number, if any, which was passed with the
C	RANGE keyword.  Returns an error if the name does not match any
C	existing elements, the instance number is negative or zero, or in
C	the event of other syntax problems.
C
C     MOD:
C          15-DEC-2003, PT:
C             expand element names to 16 characters.
C
      USE XSIF_SIZE_PARS
	USE XSIF_INOUT
	USE XSIF_ELEMENTS

	IMPLICIT NONE
	SAVE

C	ARGUMENT DECLARATIONS

	CHARACTER(ENAME_LENGTH) TOKEN    ! element name
	INTEGER*4   LTOKEN   ! num chars in element name
	
	INTEGER*4   ELEMNO   ! index of the element
	INTEGER*4   INSTANCE ! which of the identically-named units is it

	LOGICAL*4   ERROR1   ! Signal an error

C	LOCAL DECLARATIONS

C	argument declarations

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C	initialization

	ELEMNO = 0
	INSTANCE = 0
	ERROR1 = .FALSE.

C	look the token up in the element dictionary

	CALL RDLOOK(TOKEN, LTOKEN, KELEM(1), 1, IELEM1, ELEMNO)
	IF (ELEMNO.EQ.0) THEN
	    CALL RDFAIL
		WRITE(ISCRN,910)TOKEN
		WRITE(IECHO,910)TOKEN
		ERROR1 = .TRUE.
		GOTO 9999
	ENDIF

C	If the next non-blank character is a quotation mark,
C	skip it

	IF ( (KLINE(ICOL).EQ.'"') .OR. (KLINE(ICOL).EQ."'") ) THEN
		CALL RDNEXT
		IF (FATAL_READ_ERROR) GOTO 9999
	ENDIF

C	if the next character is a square bracket, read the
C	instance value within the square brackets

	IF ( KLINE(ICOL).EQ.'[' ) THEN

          CALL RDNEXT
	    IF (FATAL_READ_ERROR) GOTO 9999
		CALL RDINT(INSTANCE,ERROR1)
		IF (FATAL_READ_ERROR) GOTO 9999
		IF (ERROR1) GOTO 9999
		IF (INSTANCE.EQ.0) THEN
			CALL RDFAIL
			WRITE(ISCRN,920)
			WRITE(IECHO,920)
			GOTO 9999
		ENDIF

C	now find the other square bracket

		CALL RDTEST(']',ERROR1)
		IF (ERROR1) GOTO 9999
		CALL RDNEXT
		IF (FATAL_READ_ERROR) GOTO 9999

	ENDIF

9999	RETURN

C========1=========2=========3=========4=========5=========6=========7=C

 910	FORMAT(' *** ERROR *** ELEMENT "',A,'" NOT FOUND IN ELEMENT '
     &       'LIST'/' ')
 920  FORMAT(' *** ERROR *** ZERO VALUE FOR ELEMENT INSTANCE NOT '
     &	   'PERMITTED'/' ')

C========1=========2=========3=========4=========5=========6=========7=C

	END