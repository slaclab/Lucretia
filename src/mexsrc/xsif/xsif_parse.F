	INTEGER FUNCTION XSIF_PARSE( XSIFFILE, BEAMLINE, XSIFECHO, 
     &							 OUTLUN )

C	wrapper to actual XSIF library calls to parse an XSIF deck.
C	When called with a filename (XSIFFILE), a beamline name
C	(BEAMLINE), an echo flag (XSIFECHO), and an output logical unit
C	number (OUTLUN), will parse an XSIF deck living in a single file
C	into appropriate MAD data structures; if BEAMLINE is not blank,
C	XSIF_PARSE will execute command USE, BEAMLINEname before returning
C	control to calling routine.

C	MOD:
C          24-jul-2000, PT:
C             add DIMAD-style "PATH" statement; renamed USE routine 
C             to XUSE.
C          09-jul-1999, PT:
C             added DIMAD-style "OPEN", "CLOSE", and "CALL" commands
C             to allow user to read multiple decks.  Do not force a
C             USE command unless we are returning to the calling routine
C             (ie, unless the CALL stack is cleared)
C		 23-NOV-1998, PT:
C			eliminated CONTROL_MOD and DEFINITIONS_MOD; added OUTLUN
C			to arglist.  Moved from LIAR to XSIF library.

C	USE DEFINITIONS_MOD
C	USE CONTROL_MOD

	USE XSIF_INOUT

	IMPLICIT NONE
	SAVE

C	argument declarations

	CHARACTER(80), INTENT(IN) ::    XSIFFILE
	CHARACTER(8), INTENT(IN)  ::    BEAMLINE
	LOGICAL, INTENT(IN) ::          XSIFECHO
	INTEGER*4, INTENT(IN) ::        OUTLUN

C	local declarations

	INTEGER	   ERR_LCL
	LOGICAL    USELINE
	INTEGER*4 NCOMM
c	PARAMETER ( NCOMM = 10 )
	PARAMETER ( NCOMM = 11 )
	INTEGER*4  LNAME, LKEYW, ICOMM
	CHARACTER(8) KCOMM(NCOMM),KKEYW,KNAME
	LOGICAL	     PFLAG

	      DATA KCOMM                                                                
     +    /'RETURN  ','LINE    ','PARAMETE',                                                             
     +     'USE     ',                                                  
     +     'NOECHO  ','ECHO    ','CONSTANT',
     +     'OPEN    ','CLOSE   ','CALL    ',
     +     'PATH    '/                                    

C	referenced functions

	LOGICAL     INTRAC

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	USELINE = .FALSE.
	ERR_LCL = 0

C	open the XSIF file for input

	OPEN( 15, FILE=XSIFFILE, STATUS='OLD',IOSTAT=ERR_LCL )

		IF ( ERR_LCL .NE. 0 ) THEN
			WRITE(OUTLUN,*)'ERR> Cannot open XSIF file:  ',
     &				XSIFFILE
			ERR_LCL = XSIF_PARSE_NOOPEN
			GOTO 9999
		ENDIF

C	open the echo file for error messages

	OPEN( 16, FILE='XSIF.ERR',STATUS='REPLACE',IOSTAT=ERR_LCL )

		IF ( ERR_LCL .NE. 0 ) THEN
			WRITE(OUTLUN,*)'ERR> Cannot open XSIF file:  ',
     &				'XSIF.ERR'
			ERR_LCL = XSIF_PARSE_NOOPEN
			GOTO 9999
		ENDIF

C	open the stream file for echoing

	OPEN( 17, FILE='XSIF.STR',STATUS='REPLACE',IOSTAT=ERR_LCL )

		IF ( ERR_LCL .NE. 0 ) THEN
			WRITE(OUTLUN,*)'ERR> Cannot open XSIF file:  ',
     &				'XSIF.STR'
			ERR_LCL = XSIF_PARSE_NOOPEN
			GOTO 9999
		ENDIF

C	call RDINIT, to set up I/O commands

	CALL RDINIT( 15, 17, 16 )

C	setup ECHO or NOECHO status

	CALL CLEAR

	IF ( XSIFECHO ) THEN
		NOECHO = 0
	ELSE
		NOECHO = 1
	ENDIF

	SCAN = .FALSE.
	INTER = INTRAC()

C	BEGIN COMMAND LOOP!

100	CONTINUE
	PFLAG = .TRUE.
C---- SKIP SEPARATORS                                                           
        ERROR = .FALSE.                                                         
        SKIP  = .FALSE.                                                         
        CALL RDSKIP(';')                                                        
C---- END OF FILE READ?                                                         
        IF (ENDFIL) THEN                                                        
          KTEXT = 'RETURN!'                                                   
          ENDFIL = .FALSE.                                                      
        ENDIF                                                                   
        ILCOM = ILINE                                                           
C---- LABEL OR KEYWORD                                                          
        CALL RDWORD(KNAME,LNAME)                                                
        IF (LNAME .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,910)
		WRITE (ISCRN,910)                                                     
          ERROR = .TRUE.                                                        
          GO TO 800                                                             
        ENDIF                                                                   
C---- IF NAME IS LABEL, READ KEYWORD                                            
        IF (KLINE(ICOL) .EQ. ':') THEN                                          
          CALL RDNEXT                                                           
C---- PARAMETER NEW FLAVOR " := "              **** GJR 2 JAN 90 ****
          IF (KLINE(ICOL) .EQ. '=') THEN
            KKEYW = 'PARAMETE'                                                  
            LKEYW = 8                                                           
            PFLAG = .FALSE.                                                     
          ELSE
C----                                          **** GJR 2 JAN 90 ****
            CALL RDWORD(KKEYW,LKEYW)                                            
            IF (LKEYW .EQ. 0) THEN                                              
              CALL RDFAIL                                                       
              WRITE (IECHO,920)
			WRITE (ISCRN,920)                                                 
              ERROR = .TRUE.                                                    
              GO TO 800                                                         
            ENDIF                                                               
          ENDIF
C---- PARAMETER                                                                 
        ELSE IF(KLINE(ICOL).EQ.'=') THEN                                        
          KKEYW = 'PARAMETE'                                                    
          LKEYW = 8                                                             
          PFLAG = .FALSE.                                                       
C---- BEAMLINE WITH FORMAL ARGUMENT LIST                                        
        ELSE IF(KLINE(ICOL).EQ.'(') THEN                                        
          KKEYW = 'LINE'                                                        
          LKEYW = 4                                                             
C---- NAME WAS KEYWORD                                                          
        ELSE                                                                    
          KKEYW = KNAME                                                         
          LKEYW = LNAME                                                         
          KNAME = '        '                                                    
          LNAME = 0                                                             
        ENDIF                                                                   
C---- KNOWN COMMAND KEYWORD?                 
        CALL RDLOOK(KKEYW,LKEYW,KCOMM,1,NCOMM,ICOMM)                            

		SELECT CASE ( ICOMM )

			CASE( 1 )    ! RETURN statement

                  IF ( NUM_CALL .GT. 0 ) THEN
                      CALL XRETRN
                  ELSE

				    IF ( (.NOT. USELINE) 
     &				      .AND. 
     &				    (BEAMLINE .EQ. ' ')  )  THEN
					    ERR_LCL = XSIF_PARSE_NOLINE
					    GOTO 9999
				    ELSEIF ( .NOT. USELINE ) THEN
					    KTEXT(1:1) = ','
					    KTEXT(2:9) = BEAMLINE
					    KTEXT(10:10) = ';'
					    ICOL = 1
					    CALL USE
					    GOTO 9999
				    ELSE
					    GOTO 9999
				    ENDIF

                  ENDIF

			CASE( 2 )   ! LINE statement

				CALL LINE(KNAME,LNAME)

			CASE( 3 )   ! PARAMETER statement

				CALL PARAM(KNAME,LNAME,PFLAG)

			CASE( 4 )   ! USE statement

c				CALL USE
				CALL XUSE
				USELINE = .TRUE.

			CASE( 5 )   ! NOECHO statement

				NOECHO = 1

			CASE( 6 )   ! ECHO statement

				NOECHO = 0

			CASE( 7 )   ! CONSTANT statement
				PFLAG = .FALSE.
				CALL PARAM(KNAME,LNAME,PFLAG)

              CASE( 8 )   ! OPEN statement
                  CALL XOPEN

              CASE( 9 )   ! CLOSE statement
                  CALL XCLOSE

              CASE( 10 )  ! CALL statement
                  CALL XCALL

              CASE( 11 )  ! PATH statement
                  CALL XPATH

			CASE( 0 )   ! element keyword

				CALL ELMDEF( KKEYW,LKEYW,KNAME,LNAME )

		END SELECT

C	respond to errors in input parser

  800   IF (ERROR) THEN                                                         
          IF (INTER) THEN                                                       
            WRITE (IECHO,930)
		  WRITE (ISCRN, 930)                                                   
          ELSE IF (.NOT. SCAN) THEN                                             
            WRITE (IECHO,940)
		  WRITE (ISCRN,940)                                                   
            SCAN = .TRUE.                                                       
          ENDIF                                                                 
          CALL RDFIND(';')                                                      
        ENDIF                                                                   
      GO TO 100                                                                 

C	set return code and exit

9999	IF (SCAN) THEN
		ERR_LCL = XSIF_PARSE_ERROR
	ENDIF

	XSIF_PARSE = ERR_LCL

	CLOSE( 15 )
	CLOSE( 16 )
	CLOSE( 17 )

	RETURN

C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** COMMAND KEYWORD OR LABEL EXPECTED'/' ')            
  920 FORMAT(' *** ERROR *** COMMAND KEYWORD EXPECTED'/' ')                     
  930 FORMAT(' ***** RETYPE COMMAND *****')                                     
  940 FORMAT('0... ENTER SCANNING MODE'/' ')                                    
C-----------------------------------------------------------------------        


	END

