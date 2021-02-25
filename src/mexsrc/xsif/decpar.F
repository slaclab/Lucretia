      SUBROUTINE DECPAR(IEP1,IEP2,CHTYPE,CHLABL,ERROR1)                            
C     
C     part of MAD INPUT PARSER
C
C---- DECODE PARAMETER LIST                                                     
C
C     MOD:
C		  25-MAY-2003, PT:
C			if IEP1==0, don't try to access KPARM
C           15-NOV-2002, PT:
C             support for BEAM statement.
C           31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
c          05-jan-2001, PT:
c             issue non-NLC warnings only if NLC_STD flag is TRUE.
C          05-mar-1999, PT:
C             improved handling of non-NLC-standard parameters for NLC-
C             standard keywords.
C		 23-nov-1998, PT:
C			setup so TYPE information is enclosed in single or double
C			quotes; this is done here by replacing the RDNEXT statement
C			with RDSKIP('"''') (hope that works).
C		 30-OCT-1998, PT:
C			warn if non-NLC-standard features are used.
C		 29-OCT-1998, PT:
C			if the keyword is LRAD, translate to L (MAD compatibility
C			feature).
C	     15-SEP-1998, PT:
C             if parameter takes a FILENAME rather than a NUMBER as
C	        argument, read the filename and appropriately mutilate
C	        the KLINE data.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C		  28-AUG-1998, PT:
C			 replaced KTYPE and KLABL with CHTYPE and CHLABL
C           27-aug-1998, PT:
C              changed KLABL to char*16
C           13-july-1998, PT:
C              replaced ERROR with ERROR1 as error flag for routine.
C              replaced LABEL with ILABEL
C
C----
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
      USE XSIF_ELEM_PARS
C
C-----------------------------------------------------------------------        
c      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)                        
      IMPLICIT NONE
      SAVE
C
      CHARACTER(ETYPE_LENGTH)      CHTYPE                                                   
      CHARACTER(ELABL_LENGTH)      CHLABL                                                   
      LOGICAL*4         ERROR1 
      LOGICAL*4   ::    MULT_WARN = .FALSE.                                                  
C-----------------------------------------------------------------------        
      CHARACTER*8       KNAME                                                   
      LOGICAL           FLAG
C
      INTEGER*4         IEP, ILABEL, IEP2, IEP1, LTYPE, LNAME

      LOGICAL*4   ::    IS_BEAM, LOG_VAL
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.                                                           
      CHTYPE = '        '                                                        
      CHLABL = '                '   
C---- Detect whether the present element is a BEAM element
	IF (IEP1.EQ.0) THEN
	    IS_BEAM = .FALSE.
      ELSEIF ( KPARM(IEP1) .EQ. 'PARTICLE' ) THEN
          IS_BEAM = .TRUE.
C---- if so, set default values of the parameters
          CALL SET_BEAM_DEFAULTS( IEP1, IEP2 )
      ELSE
          IS_BEAM = .FALSE.
      ENDIF
                                               
C---- ANOTHER PARAMETER?                                                        
  100 IF (KLINE(ICOL) .EQ. ',') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
C---- PARAMETER KEYWORD                                                         
        CALL RDWORD(KNAME,LNAME)                                                
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,910)                                                     
          WRITE (ISCRN,910)                                                     
          GO TO 200                                                             
        ENDIF                                                                   
C---- TEST FOR "TYPE" KEYWORD                                                   
        IF (KNAME .EQ. 'TYPE') THEN                                             
          CALL RDTEST('=',FLAG)                                                 
          IF (FLAG) GO TO 200                                                   
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
		call RDSKIP('"''')
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDLABL(CHTYPE,LTYPE)                                              
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (LTYPE .EQ. 0) THEN                                                
            CALL RDFAIL                                                         
            WRITE (IECHO,920)                                                   
            WRITE (ISCRN,920)                                                   
            GOTO 200                                                            
          ENDIF                                                                 
C---- TEST FOR "LABEL" KEYWORD                                                  
        ELSE IF(KNAME .EQ. 'LABEL') THEN                                        
          CALL RDTEST('=',FLAG)                                                 
          IF (FLAG) GO TO 200                                                   
          CALL RDNEXT
          IF (FATAL_READ_ERROR) GOTO 9999
		call RDSKIP('"''')                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDLABL(CHLABL,ILABEL)                                              
          IF (FATAL_READ_ERROR) GOTO 9999                             
          IF (ILABEL .EQ. 0) THEN                                                
            CALL RDFAIL                                                         
            WRITE (IECHO,925)                                                   
            WRITE (ISCRN,925)                                                   
            GOTO 200                                                            
          ENDIF                                                                 
C---- KEYWORD MUST DESIGNATE NUMERIC PARAMETER                                  
        ELSE                                                                    
		IF ( KNAME(1:4) .EQ. 'LRAD' ) THEN
			KNAME = 'L'
			LNAME = 1
		ENDIF
c
c		non-NLC feature:  solenoid K1 value
c
C		IF ( (KNAME(1:2) .EQ. 'K1') .AND.
C     &		 (KPARM(IEP1+1)(1:2) .EQ. 'KS')  ) THEN
C			CALL RDWARN
C			WRITE(IECHO,960)KNAME
C			WRITE(ISCRN,960)KNAME
C		ENDIF
C
C		non-NLC feature:  monitor error values
C
C		IF ( (KNAME(1:1) .NE. 'L') .AND.
C     &		 (KPARM(IEP1+1)(1:5) .EQ. 'XSERR')  ) THEN
C			CALL RDWARN
C			WRITE(IECHO,960)KNAME
C			WRITE(ISCRN,960)KNAME
C		ENDIF 

          CALL RDLOOK(KNAME,LNAME,KPARM(1),IEP1,IEP2,IEP)                          
          IF (IEP .EQ. 0) THEN                                                  
            CALL RDWARN                                                         
            WRITE (IECHO,930) KNAME(1:LNAME)                                    
            WRITE (ISCRN,930) KNAME(1:LNAME)                                    
            CALL RDFIND(',;')                                                   
            IF (FATAL_READ_ERROR) GOTO 9999
          ELSE IF (IPTYP(IEP) .GE. 0) THEN                                      
            CALL RDWARN                                                         
            WRITE (IECHO,940) KNAME(1:LNAME)                                    
            WRITE (ISCRN,940) KNAME(1:LNAME)                                    
            CALL RDFIND(',;')                                                   
            IF (FATAL_READ_ERROR) GOTO 9999
          ELSE                                                                  
C---- EQUALS SIGN?                                                              
            CALL RDTEST('=,;',FLAG)                                             
            IF (FLAG) GO TO 200                                                 
            IF (KLINE(ICOL) .EQ. '=') THEN                                      
              CALL RDNEXT
              IF (FATAL_READ_ERROR) GOTO 9999

C----	if the parameter takes a FILENAME as an argument, then
C---- decode that here; intercept the filename in KLINE and
C---- replace with a pointer to the appropriate filename
C---- registry

			IF ( KNAME .EQ. 'LFILE   ' ) THEN
				CALL RD_WAKEFILENAME( LWAKE_FLAG, ERROR1 )
                  IF (FATAL_READ_ERROR) GOTO 9999
			ENDIF                                                   
			IF ( KNAME .EQ. 'TFILE   ' ) THEN
				CALL RD_WAKEFILENAME( TWAKE_FLAG, ERROR1 )
                  IF (FATAL_READ_ERROR) GOTO 9999
			ENDIF
			IF ( ERROR1 ) THEN
				GOTO 200
			ENDIF

C----	End of hacking in filename-reading for arguments

C---- if we are parsing a BEAM, then it's possible that 
C---- the parameter is a PARTICLE, BUNCHED, or RADIATE
C---- statement (none of which take a numeric value).
C---- Manage that here.

              IF (IS_BEAM) THEN

                IF (KNAME .EQ. 'PARTICLE') THEN

		        call RDSKIP('"''')                                                           
                  IF (FATAL_READ_ERROR) GOTO 9999
                  CALL RDLABL(CHLABL,ILABEL)                                              
                  IF (FATAL_READ_ERROR) GOTO 9999  
                  IF (ILABEL .EQ. 0) THEN                                                
                   CALL RDFAIL                                                         
                   WRITE (IECHO,926)                                                   
                   WRITE (ISCRN,926)                                                   
                   GOTO 200                                                            
                  ENDIF                                                                 
                  CALL DECODE_PART_NAME(CHLABL,ILABEL,IEP) 
                  GOTO 100

                ELSE IF ( (KNAME .EQ. 'BUNCHED') .OR.
     &                    (KNAME .EQ. 'RADIATE')      ) THEN
                  CALL RD_LOGICAL(LOG_VAL,FLAG)
                  IF (FATAL_READ_ERROR) GOTO 9999
                  IF (FLAG) GOTO 200
                  IF (LOG_VAL) THEN
                    PDATA(IEP) = MAD_RTRUE
                  ELSE
                    PDATA(IEP) = MAD_RFALSE
                  ENDIF
                  IPTYP(IEP) = 0
                  GOTO 100

                ENDIF

              ENDIF

C     end of hacking in PARTICLE/BUNCHED/RADIATE conditionals.
                      
             CALL DECEXP(IEP,FLAG)                                             
              IF (FATAL_READ_ERROR) GOTO 9999
              IF (FLAG) GO TO 200                                               
            ELSE IF (IPTYP(IEP) .EQ. -2) THEN                                   
              IPTYP(IEP) = 0                                                    
            ELSE                                                                
              CALL RDWARN                                                       
              WRITE (IECHO,950)                                                 
              WRITE (ISCRN,950)                                                 
            ENDIF                                                               
          ENDIF
          
C         detect non-NLC features

c          IF ( IEP .NE. 0 ) THEN
          IF ( ( IEP .NE. 0 ) .AND. (NLC_STD)  ) THEN
	        IF ( (IKEYW_GLOBAL .EQ. MAD_MULTI) .AND.
     &                   (.NOT. MULT_WARN) .AND.
     &                   (IEP-IEP1 .GT. 44) .AND.
     &                   (IEP-IEP1 .LT. 49)           ) THEN
                  MULT_WARN = .TRUE.
                  CALL RDWARN
                  WRITE(IECHO,970)
                  WRITE(ISCRN,970)
              ELSE IF ( (NLC_STANDARD) .AND.
     &                (IKEYW_GLOBAL .NE. MAD_MULTI) .AND.
     &                (IKEYW_GLOBAL .NE. MAD_BET0) .AND.
     &                (IKEYW_GLOBAL .NE. MAD_BEAM) .AND.
     &          (.NOT. NLC_PARAM(IKEYW_GLOBAL,IEP-IEP1+1) ) ) THEN
                  CALL RDWARN
                  WRITE(IECHO,960)KNAME
                  WRITE(ISCRN,960)KNAME
              ENDIF
          ENDIF 
                                                                           
        ENDIF                                                                   
        GO TO 100                                                               
C---- ERROR RECOVERY                                                            
  200   CALL RDFIND(',;')                                                       
        IF (FATAL_READ_ERROR) GOTO 9999
        ERROR1 = .TRUE.                                                          
        GO TO 100                                                               
      ENDIF                                                                     
C---- FINAL CHECK                                                               
      CALL RDTEST(',;',FLAG)                                                    
      IF (FLAG .OR. ERROR1) THEN                                                 
        ERROR1 = .TRUE.                                                          
        IEP1 = 0                                                                
        IEP2 = 0                                                                
      ELSE IF (IEP1 .NE. 0) THEN                                                
        DO 290 IEP = IEP1, IEP2                                                 
          IF (IPTYP(IEP) .LT. -1) THEN                                          
            IPTYP(IEP) = -1                                                     
            PDATA(IEP) = 0.0                                                    
          ENDIF                                                                 
  290   CONTINUE                                                                
      ENDIF         

9999  IF (FATAL_READ_ERROR) ERROR1=.TRUE.

                                                            
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** PARAMETER KEYWORD EXPECTED'/' ')                   
  920 FORMAT(' *** ERROR *** TYPE IDENTIFIER EXPECTED'/' ')                     
  925 FORMAT(' *** ERROR *** LABEL IDENTIFIER EXPECTED'/' ')   
  926 FORMAT(' *** ERROR *** PARTICLE IDENTIFIER EXPECTED'/' ')                 
  930 FORMAT(' ** WARNING ** UNKNOWN PARAMETER KEYWORD "',A,                    
     +       '" --- PARAMETER IGNORED'/' ')                                     
  940 FORMAT(' ** WARNING ** MULTIPLE DEFINITION OF PARAMETER "',A,             
     +       '" --- PREVIOUS VALUE USED'/' ')                                   
  950 FORMAT(' ** WARNING ** "=" EXPECTED --- DEFAULT VALUE USED'/' ')
  960 FORMAT(' ** WARNING ** NON-"NLC STANDARD" PARAMETER "',A8,
     +	   '" DETECTED.'/' ')  
  970 FORMAT(' ** WARNING ** ONE OR MORE MULTIPOLE WITH SOLENOIDAL ',
     +        'PARAMETERS DETECTED -- NON-NLC-STANDARD!')                
C-----------------------------------------------------------------------        
      END                                                                       
