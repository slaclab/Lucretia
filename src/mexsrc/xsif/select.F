	SUBROUTINE SELECT
C
C	emulate a MAD SELECT ERROR statement.  At present, the
C	implementation is limited to selecting a signle element
C	(ie, limited use of the RANGE keyword), or selecting FULL
C	or CLEAR.  Also, this implementation will permit certain
C	syntaxes not allowed by the MAD version.
C
C     MOD:
C          15-dec-2003, PT:
C             allow longer token lengths.
C
      USE XSIF_SIZE_PARS
	USE XSIF_INOUT
	USE XSIF_ELEMENTS
C
C	local declarations
C
	INTEGER*4, PARAMETER :: NUMFLAG = 1
	CHARACTER*8 FLAGVAL(NUMFLAG)
	DATA FLAGVAL /'ERROR   '/
	
	INTEGER*4, PARAMETER :: NUMSELARG = 4
	CHARACTER*8 ARGVAL(NUMSELARG)
	DATA ARGVAL /'FLAG    ','RANGE   ','FULL    ','CLEAR   '/

	LOGICAL*4 FULLFLAG, CLEARFLAG, RANGEFLAG, XORFLAG
	INTEGER*4 ELEM_POINTER

	CHARACTER(MAXCHARLEN) TOKEN
	INTEGER*4 LTOKEN, IARG, IFLG, FLAG_VALUE
	INTEGER*4 ELEMNO, INSTANCE, ELEM_LOOP, INSTANCE_CTR

	LOGICAL*4 ERROR1

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C	initialization

	FULLFLAG = .FALSE.
	CLEARFLAG = .FALSE.
	RANGEFLAG = .FALSE.
	ELEM_POINTER = 0
	FLAG_VALUE = 0
	ELEMNO = 0
	INSTANCE = 0

C	loop over tokens (in emulation of DECPAR).  
C	Note that MAD allows SELECT tokens to be separated by spaces
C	or commas, the only way to tell that the command is ended is
C	by presence of a semicolon

c 100	IF (KLINE(ICOL).NE.';') THEN

	DO

		IF (KLINE(ICOL).EQ.';') EXIT

C	step to next character which is not blank, not a single quote,
C	not a double quote, and not a comma

		DO
			IF ( (KLINE(ICOL).NE.'"') .AND.
     &			 (KLINE(ICOL).NE."'") .AND.
     &			 (KLINE(ICOL).NE.',')       ) 
     &		   EXIT
			CALL RDNEXT
			IF (FATAL_READ_ERROR) GOTO 9999
		ENDDO

C	read the next word from the input buffer

		CALL RDWORD( TOKEN, LTOKEN )
		IF (FATAL_READ_ERROR) GOTO 9999
		IF (LTOKEN .EQ. 0) THEN
			CALL RDFAIL
			WRITE(IECHO,920)
			WRITE(ISCRN,920)
			GOTO 9998
		ENDIF

C	is it a standard MAD keyword associated with SELECT and
C	permitted in this implementation?  Or is it a standard
C	MAD flag associated with SELECT and permitted in this
C	implementation?

		CALL RDLOOK(TOKEN,LTOKEN,ARGVAL, 1,NUMSELARG,IARG)
		CALL RDLOOK(TOKEN,LTOKEN,FLAGVAL,1,NUMFLAG,  IFLG)

C	In the present implementation, either IARG or IFLG or both
C	will be zero.  We can combine them into a single integer which
C	is zero if both IARG and IFLG are zero, and otherwise selects
C	a single outcome:

		IARG = IARG + IFLG * (NUMSELARG+1)

		SELECT CASE (IARG)

			CASE (5)                       ! ERROR

				FLAG_VALUE = 1 

			CASE (4)                       ! CLEAR

				CLEARFLAG = .TRUE.

			CASE (3)                       ! FULL

				FULLFLAG = .TRUE.

			CASE (2)					   ! RANGE

C	skip the equals sign and any quotation marks

				RANGEFLAG = .TRUE.
				CALL RDTEST('=',ERROR)
                  CALL RDNEXT
	            IF (FATAL_READ_ERROR) GOTO 9999
				IF (ERROR) GOTO 9998
				IF ( (KLINE(ICOL).EQ.'"') .OR.
     &			     (KLINE(ICOL).EQ."'")      ) THEN
					CALL RDNEXT
					IF (FATAL_READ_ERROR) GOTO 9999
				ENDIF

C	read the next word.  This is necessary because, in the
C	case of a SEL ERROR elemname syntax we read the elemname
C	in the RDWORD call above, so in the case of the
C	SEL ERROR RANGE=elemname syntax we need to "catch up"

				CALL RDWORD( TOKEN, LTOKEN )
				IF (FATAL_READ_ERROR) GOTO 9999
				IF (LTOKEN .EQ. 0) THEN
					CALL RDFAIL
					WRITE(IECHO,930)
					WRITE(ISCRN,930)
					GOTO 9998
				ENDIF

C	parse the range definition

				CALL READ_RANGE(TOKEN(1:ENAME_LENGTH), 
     &                            MIN(LTOKEN,ENAME_LENGTH),
     &							ELEMNO, INSTANCE, ERROR1)
				IF (FATAL_READ_ERROR) GOTO 9999
				IF (ERROR1) GOTO 9998

			CASE (1)						! FLAG

C	skip the equals sign

				CALL RDTEST('=',ERROR1)
				IF (ERROR1) GOTO 9998
				CALL RDNEXT
				IF (FATAL_READ_ERROR) GOTO 9999

C	read the flag name

				CALL RDWORD( TOKEN, LTOKEN )
				IF (FATAL_READ_ERROR) GOTO 9999
				IF (LTOKEN .EQ. 0) THEN
					CALL RDFAIL
					WRITE(IECHO,940)
					WRITE(ISCRN,940)
					GOTO 9998
				ENDIF

C	decode the flag name

				CALL RDLOOK(TOKEN,LTOKEN,FLAGVAL,1,NUMFLAG,  
     &						FLAG_VALUE)
				IF (FATAL_READ_ERROR) GOTO 9999
				IF (FLAG_VALUE .EQ. 0) THEN
					CALL RDFAIL
					WRITE(IECHO,950)
					WRITE(ISCRN,950)
					GOTO 9998
				ENDIF

			CASE (0)						! Range (inferred)

C	parse the range definition

				RANGEFLAG = .TRUE.
				CALL READ_RANGE(TOKEN(1:ENAME_LENGTH), 
     &                            MIN(LTOKEN,ENAME_LENGTH),
     &							ELEMNO, INSTANCE, ERROR1)
				IF (ERROR1) GOTO 9998
				RANGEFLAG = .TRUE.

		END SELECT

C	at this point, no matter what happened, the input buffer
C	pointer ICOL is pointed at the next unread input char,
C	so cycle back to the top (thus skipping the error-
C	recovery stage below)

		CYCLE

C	recovery from a syntax error -- don't even bother trying
C	to parse the rest of the command

9998		CALL RDFIND(';')
          IF (FATAL_READ_ERROR) GOTO 9999
          ERROR = .TRUE.      
		
	ENDDO ! end of token-reading loop     
	
C	We've now completed parsing the SELECT statement and can move
C	on to taking whatever action the command requires.  The first
C	obvious choice of actions is error-handling.  Error conditions
C	include:

C	a SELECT statement which is not preceded by a USE statement...

	IF (.NOT. LINE_EXPANDED) THEN

		ERROR = .TRUE.
		WRITE(IECHO,910)
		WRITE(ISCRN,910)

	ENDIF

C	...Simultaneously selecting 2 or more of FULL, CLEAR, and RANGE,
C	   or for that matter not selecting any...

	XORFLAG = ( (FULLFLAG .NEQV. CLEARFLAG) .NEQV. RANGEFLAG )

	IF (.NOT. XORFLAG) THEN
	
		ERROR = .TRUE.
		WRITE(IECHO,960)
		WRITE(ISCRN,960)
		
	ENDIF 
	
C	...or failing to select a FLAG value.

	IF (FLAG_VALUE .EQ. 0) THEN

		ERROR = .TRUE.
		WRITE(IECHO,970)
		WRITE(ISCRN,970)
		
	ENDIF 

	IF (ERROR) GOTO 9999
	
C	now we act on the SELECT command.  Although at the time of
C	authorship only the ERROR flag is permitted, we will not
C	assume that this is the only flag that will ever be 
C	permitted...

	SELECT CASE ( FLAG_VALUE )

		CASE( 1 )              ! FLAG = ERROR

			IF (FULLFLAG) ERRFLG = .TRUE.
			IF (CLEARFLAG) ERRFLG = .FALSE.
			IF (RANGEFLAG) THEN

C	loop over entries in the ITEM list

			  INSTANCE_CTR = 0

			  DO ELEM_LOOP = NPOS1,NPOS2

				IF (ITEM(ELEM_LOOP).EQ.ELEMNO) THEN

      				INSTANCE_CTR = INSTANCE_CTR + 1
					IF ( (INSTANCE.EQ.0)   .OR.
     &					 (INSTANCE .EQ. INSTANCE_CTR) ) THEN
      					ERRFLG(ELEM_LOOP) = .TRUE.
					ENDIF

				ENDIF

			  ENDDO

			ENDIF

	END SELECT

C	all done

9999  IF (FATAL_READ_ERROR) ERROR = .TRUE.
	RETURN

C========1=========2=========3=========4=========5=========6=========7=C
 910  FORMAT(' *** ERROR *** "SELECT" REQUIRES BEAMLINE EXPANSION',
     &' VIA "USE" COMMAND'/' ')     

 920  FORMAT(' *** ERROR *** "SELECT" KEYWORD OR ELEMENT RANGE '
     &	   'SPECIFICATION EXPECTED'/' ')
 930  FORMAT(' *** ERROR *** ELEMENT RANGE SPECIFICATION EXPECTED'/' ')
 940  FORMAT(' *** ERROR *** FLAG VALUE EXPECTED'/' ')
 950  FORMAT(' *** ERROR *** UNRECOGNIZED FLAG VALUE SPECIFIED'/' ')
 960	FORMAT(' *** ERROR *** "RANGE", "FULL", AND "CLEAR" ARE MUTUALLY '
     &	   'EXCLUSIVE OPTIONS IN "SELECT" OPERATION'/' ')
 970  FORMAT(' *** ERROR *** NO FLAG VALUE SET IN "SELECT" OPERATION'
     &	   /' ')
C========1=========2=========3=========4=========5=========6=========7=C

	END