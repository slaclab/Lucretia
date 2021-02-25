	SUBROUTINE INT_TO_CHAR( NUMBER, NSTR, EFLAG )
C
C	member of MAD INPUT PARSER
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C	Takes an integer argument and converts it to the equivalent
C	character string, ie, 12345 -> '12345'.  If the size of the
C	string passed for the value is inadequate, NSTR is returned
C	with all zeroes and EFLAG is TRUE, otherwise EFLAG is false.
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C	modules:
C
	IMPLICIT NONE
	SAVE
C
C	argument declarations
C
	INTEGER*4, INTENT(IN)      :: NUMBER
	CHARACTER*(*), INTENT(OUT) :: NSTR
	LOGICAL*4, INTENT(OUT)     :: EFLAG
C
C	local declarations
C
	INTEGER*4 NDIGIT, STRSIZE
	INTEGER*4 DIGIT, NTEMP1, NTEMP2
	INTEGER*4 NUMSIZE, COUNTER, PLACE
C
C	referenced functions
C

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                     C  O  D  E                                       C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	EFLAG = .FALSE.
	NDIGIT = 0

	STRSIZE = LEN(NSTR)
	IF ( NUMBER .LT. 0 ) THEN
		NSTR(1:1) = '-'
		NTEMP2 = -NUMBER
		STRSIZE = STRSIZE - 1
		NDIGIT = 1
	ELSE
		NTEMP2 = NUMBER
	ENDIF

	NUMSIZE = INT( LOG10( DBLE(NTEMP2) ) ) + 1
	
	IF ( NUMSIZE .GT. STRSIZE ) THEN         ! overflow case
		DO NDIGIT = 1,STRSIZE
			NSTR(NDIGIT:NDIGIT) = '0'
		ENDDO
		EFLAG = .TRUE.
		GOTO 9999
	ENDIF

C	interrogate the number from least significant digit to most,
C	and put into the appropriate place in NSTR

	DO COUNTER = 0, NUMSIZE-1
		PLACE = NDIGIT + NUMSIZE - COUNTER
		NTEMP1 = INT( DBLE( NTEMP2 ) / 10 )
		DIGIT = NTEMP2 - 10*NTEMP1
		NSTR(PLACE:PLACE) = ACHAR( DIGIT+48 )
		NTEMP2 = NTEMP1
	ENDDO



9999	RETURN
	END

