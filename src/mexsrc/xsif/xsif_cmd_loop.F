      INTEGER*4 FUNCTION XSIF_CMD_LOOP( XSIF_EXTRA_CMD )
C
C     Master command loop for evaluating the stuff that comes
C     in on the command line of the XSIF deck.  In order to
C     facilitate additional, user-defined commands, the user
C     may pass a logical*4 function as XSIF_EXTRA_CMD, which
C     is an optional argument.  Keywords go first to XSIF_EXTRA_CMD
C     for handling, and only if it comes back FALSE does 
C     XSIF_CMD_LOOP attempt to decode the keyword.  So users
C     can, deliberately or inadvertantly, override the default
C     XSIF actions on keywords!  Note that to make this work,
C     we have defined an interface in XSIF_INTERFACES; the calling
C     routine needs XSIF_INTERFACES to work properly.
C
C     AUTH: PT, 05-JAN-2001
C
C     MOD:
C          27-feb-2004, PT:
C             initialize SCAN to FALSE.
C          15-DEC-2003, PT:
C             support longer tokens.
C		 05-DEC-2003, PT:
C			support for SELECT and EALIGN statements.
C		 22-MAY-2003, PT:
C			support for dynamic allocation function.  Slightly
C			more compact arrangement of error status control.
C          02-mar-2001, PT:
C             respond to XSIF_STOP condition.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT

      IMPLICIT NONE
      SAVE

C     argument declarations

      LOGICAL*4, EXTERNAL, OPTIONAL :: XSIF_EXTRA_CMD

C     local declarations

      LOGICAL*4   EXTRA_COMMAND

      INTEGER*4   ERR_LCL         ! error statuses, etc.

	INTEGER*4 NCOMM             ! number recognized cmds
	PARAMETER ( NCOMM = 15 )
	INTEGER*4  LNAME, LKEYW, ICOMM
	CHARACTER(8) KCOMM(NCOMM),KKEYW
      CHARACTER(MAXCHARLEN) KNAME
	CHARACTER(ENAME_LENGTH) KNAME_E
	CHARACTER(PNAME_LENGTH) KNAME_P
	LOGICAL	     PFLAG, MEM_OK

	      DATA KCOMM                                                                
     +    /'RETURN  ','LINE    ','PARAMETE',                                                             
     +     'USE     ',                                                  
     +     'NOECHO  ','ECHO    ','CONSTANT',
     +     'OPEN    ','CLOSE   ','CALL    ',
     +     'PATH    ','NLC     ','NONLC    ',
     +     'SELECT  ','EALIGN  '/   
     
C	referenced functions

	LOGICAL*4 XSIF_MEM_MANAGE                                 

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	ERR_LCL = 0
      EXTRA_COMMAND = .FALSE.
	MEM_OK = .TRUE.
      SCAN = .FALSE.                                                         

C	BEGIN COMMAND LOOP!

100	CONTINUE
	MEM_OK = XSIF_MEM_MANAGE()
	IF (.NOT. MEM_OK) GOTO 9999
	PFLAG = .TRUE.
C---- SKIP SEPARATORS                                                           
        ERROR = .FALSE.                                                         
        SKIP  = .FALSE.
        CALL RDSKIP(';')                                                        
        IF (FATAL_READ_ERROR) GOTO 9999
C---- END OF FILE READ?                                                         
        IF (ENDFIL) THEN                                                        
          KTEXT = 'RETURN!'                                                   
          ENDFIL = .FALSE.                                                      
        ENDIF                                                                   
        ILCOM = ILINE                                                           
C---- LABEL OR KEYWORD                                                          
        CALL RDWORD(KNAME,LNAME)                                                
        IF (FATAL_READ_ERROR) GOTO 9999
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
          IF (FATAL_READ_ERROR) GOTO 9999
C---- PARAMETER NEW FLAVOR " := "              **** GJR 2 JAN 90 ****
          IF (KLINE(ICOL) .EQ. '=') THEN
            KKEYW = 'PARAMETE'                                                  
            LKEYW = 8                                                           
            PFLAG = .FALSE.                                                     
          ELSE
C----                                          **** GJR 2 JAN 90 ****
            CALL RDWORD(KKEYW,LKEYW)                                            
            IF (FATAL_READ_ERROR) GOTO 9999
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

C---- Use the XSIF_EXTRA_CMD function to see whether it's some
C---- user-defined command or other.  Respond to errors in 
C---- the user-defined command handler as well.

      IF ( PRESENT( XSIF_EXTRA_CMD ) ) THEN
          EXTRA_COMMAND = XSIF_EXTRA_CMD(KKEYW,LKEYW,
     &                                   KNAME,LNAME,ERROR)
          IF (ERROR) THEN
              CALL RDFAIL
              WRITE(IECHO,950)
              WRITE(ISCRN,950)
          ENDIF
          IF ( (ERROR).OR.(EXTRA_COMMAND) ) GOTO 800
      ENDIF

C     generate "shortened" versions of KNAME

      KNAME_E = KNAME(1:ENAME_LENGTH)
	KNAME_P = KNAME(1:PNAME_LENGTH)
                                                             
C---- KNOWN COMMAND KEYWORD?                 
        CALL RDLOOK(KKEYW,LKEYW,KCOMM,1,NCOMM,ICOMM)                            

		SELECT CASE ( ICOMM )

			CASE( 1 )    ! RETURN statement

                  IF ( NUM_CALL .GT. 0 ) THEN
                      CALL XRETRN
                  ELSE
					GOTO 9999
                  ENDIF

			CASE( 2 )   ! LINE statement

				CALL LINE(KNAME_E,MIN(LNAME,ENAME_LENGTH))
C                  IF (FATAL_READ_ERROR) GOTO 9999

			CASE( 3 )   ! PARAMETER statement

				CALL PARAM(KNAME_P,MIN(LNAME,PNAME_LENGTH),PFLAG)
C                  IF (FATAL_READ_ERROR) GOTO 9999

			CASE( 4 )   ! USE statement

                  XUSE_FROM_FILE = .TRUE. ! USE command is in file
				CALL XUSE
C                  IF (FATAL_READ_ERROR) GOTO 9999

			CASE( 5 )   ! NOECHO statement

				NOECHO = 1

			CASE( 6 )   ! ECHO statement

				NOECHO = 0

			CASE( 7 )   ! CONSTANT statement
				PFLAG = .FALSE.
				CALL PARAM(KNAME_P,MIN(LNAME,PNAME_LENGTH),PFLAG)

              CASE( 8 )   ! OPEN statement
                  CALL XOPEN
C                  IF (FATAL_READ_ERROR) GOTO 9999

              CASE( 9 )   ! CLOSE statement
                  CALL XCLOSE

              CASE( 10 )  ! CALL statement
                  CALL XCALL

              CASE( 11 )  ! PATH statement
                  CALL XPATH
C                  IF (FATAL_READ_ERROR) GOTO 9999

              CASE( 12 )  ! NLC statement
                  NLC_STD = .TRUE.

              CASE( 13 )  ! NONLC statement
                  NLC_STD = .FALSE.

			CASE( 14 )  ! SELECT statement
				CALL SELECT

			CASE( 15 )  ! EALIGN statement
				CALL EALIGN

			CASE( 0 )   ! element keyword

				CALL ELMDEF( KKEYW,LKEYW,KNAME_E,
     &                         MIN(LNAME,ENAME_LENGTH) )
C                  IF (FATAL_READ_ERROR) GOTO 9999

		END SELECT
  800		IF (FATAL_ALLOC_ERROR) GOTO 9999
		IF (FATAL_READ_ERROR) GOTO 9999

C	respond to errors in input parser

        IF (ERROR) THEN                                                         
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
         
C     respond to requests to stop parser operation

        IF (XSIF_STOP) GOTO 9999
                                                               
      GO TO 100                                                                 

C	set return code and exit

9999	IF (SCAN) THEN
		ERR_LCL = XSIF_PARSE_ERROR
	ENDIF

      IF ( FATAL_READ_ERROR ) THEN
          ERR_LCL = XSIF_FATALREAD
      ENDIF

	IF ( (.NOT. MEM_OK)  .OR. (FATAL_ALLOC_ERROR) ) THEN
		ERR_LCL = XSIF_BADALLOC
	ENDIF

      XSIF_CMD_LOOP = ERR_LCL

      RETURN

C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** COMMAND KEYWORD OR LABEL EXPECTED'/' ')            
  920 FORMAT(' *** ERROR *** COMMAND KEYWORD EXPECTED'/' ')                     
  930 FORMAT(' ***** RETYPE COMMAND *****')                                     
  940 FORMAT('0... ENTER SCANNING MODE'/' ')          
  950 FORMAT(' *** ERROR *** ERROR STATUS RETURN FROM USER COMMAND',
     &    ' HANDLER'/' ')                          
C-----------------------------------------------------------------------        

      END