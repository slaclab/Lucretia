      SUBROUTINE XCALL
C
C     member of MAD INPUT PARSER
C
C---- PERFORM "CALL" COMMAND                                             
C----------------------------------------------------------------------- 
C
C	MOD:
C          16-may-2003, PT:
C             support for new CALL, FILENAME = filename syntax.
C             In old syntax, check to see whether selected unit has
C             been OPENed yet (if not, it's an ERROR, not a WARNING).
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C	     28-AUG-1998, PT:
C			adjusted code to allow nested CALLs
C
C     modules:
C
	USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_INTERFACES
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
      CHARACTER*8 KFILE
      INTEGER*4   LFILE, COLSTART, COLEND
      CHARACTER*80 FILNAM
      LOGICAL*4 XSM_STAT, XOPEN_STACK_MANAGE
      TYPE(XSIF_FILETYPE), POINTER :: IFILE_PTR
C
C----------------------------------------------------------------------- 
C     next thing should be a comma

      CALL RDTEST(',',ERROR)
      IF (ERROR) RETURN
      CALL RDNEXT
      IF (FATAL_READ_ERROR) RETURN

C     next thing should be EITHER an integer OR the keyword FILENAME

      CALL RDTEST('1234567890F',ERROR)
      IF (ERROR) THEN
        CALL RDFAIL
        WRITE(IECHO,940)
        WRITE(ISCRN,940)
        RETURN
      ENDIF

C     if it was an "F", read the keyword

      IF ( KLINE(ICOL).EQ.'F' ) THEN

        CALL RDWORD(KFILE,LFILE)
        IF (KFILE.NE.'FILENAME') THEN
          ERROR = .TRUE.
          CALL RDFAIL
          WRITE(IECHO,940)
          WRITE(ISCRN,940)
          RETURN
        ENDIF

c        CALL RDNEXT
        IF (FATAL_READ_ERROR) RETURN
        CALL RDTEST('=',ERROR)
        IF (ERROR) THEN
          CALL RDFAIL
          WRITE(IECHO,945)
          WRITE(ISCRN,945)
          RETURN
        ENDIF

C     read the filename

        CALL RDNEXT
        IF (FATAL_READ_ERROR) RETURN
        CALL RD_FILENAME(FILNAM,COLSTART,COLEND,ERROR)
        IF (ERROR) RETURN

C     open the filename to the XCALL_UNITNO unit

        XSM_STAT = XOPEN_STACK_MANAGE( FILNAM, XCALL_UNITNO, 'OLD' )
        IF (XSM_STAT) THEN
          IFILE = XCALL_UNITNO
          XCALL_UNITNO = XCALL_UNITNO + 1
        ELSE
          WRITE(IECHO,960)FILNAM
          WRITE(ISCRN,960)FILNAM
          ERROR = .TRUE.
          RETURN
        ENDIF

C     execute a RDNEXT to make this case symmetric with the unit #
C     case below...

        CALL RDNEXT
        IF (FATAL_READ_ERROR) RETURN

      ELSE ! unit number, in this case read the unit number

        CALL RDINT(IFILE,ERROR)
        IF (FATAL_READ_ERROR.OR.ERROR) RETURN
        IF(IFILE.LT.1.OR.IFILE.GT.99) THEN 
          CALL RDFAIL
          WRITE(IECHO,970)
          WRITE(ISCRN,970)
          ERROR = .TRUE.
          RETURN
        ENDIF

C     make sure that unit IFILE is open to something

        IFILE_PTR => XSIF_STACK_SEARCH( IFILE )
        IF (.NOT.ASSOCIATED(IFILE_PTR)) THEN
          WRITE(IECHO,980)IFILE
          WRITE(ISCRN,980)IFILE
          ERROR = .TRUE.
          RETURN
        ENDIF

      ENDIF ! unit # or FILENAME keyword condition

C     since the last thing we did in RDINT was a RDNEXT, we SHOULD
C     be positioned at the end-of-line semicolon now:

      CALL RDTEST(';',ERROR)
      IF (ERROR) RETURN

c      CALL RDFILE(IFILE,ERROR)                                           
c      IF (ERROR) RETURN                                                  
C      IF (IDATA .NE. 5) THEN                                             
C        CALL RDWARN                                                      
C        WRITE (IECHO,910)                                                
C      ELSE                                                               
	IF ( NUM_CALL .EQ. MXCALL ) THEN
		CALL RDWARN
		WRITE (IECHO,9100)MXCALL
		WRITE (ISCRN,9100)MXCALL
	ELSE
		IF (ICOL .NE. 81) THEN                                           
			CALL RDWARN                                                    
			WRITE (IECHO,920)                                              
			WRITE (ISCRN,920)                                              
			ICOL = 81                                                      
		ENDIF                                                            
		NUM_CALL = NUM_CALL + 1
		IO_UNIT(NUM_CALL) = IDATA
		IDATA = IFILE                                                    
		ENDFIL = .FALSE.                                                 
		REWIND (IDATA) 
          IF ( IFILE .LE. 99 ) THEN                                                  
		  WRITE (IECHO,930) IFILE                                          
		  WRITE (ISCRN,930) IFILE       
          ELSE
            WRITE(IECHO,950)ARR_TO_STR(XSIF_OPEN_STACK_TAIL%FILE_NAME)                                   
            WRITE(ISCRN,950)ARR_TO_STR(XSIF_OPEN_STACK_TAIL%FILE_NAME) 
          ENDIF                                  
      ENDIF                                                              
      RETURN                                                             
C----------------------------------------------------------------------- 
C  910 FORMAT(' ** WARNING ** "CALL"S CANNOT BE NESTED --- ',             
C     +       '"CALL" IGNORED'/' ')                                       
  920 FORMAT(' ** WARNING ** TEXT AFTER "CALL" COMMAND SKIPPED'/' ')     
  930 FORMAT('0... READING LOGICAL UNIT NUMBER ',I2,/' ')           
  940 FORMAT(' *** ERROR *** INTEGER OR "FILENAME" KEYWORD EXPECTED'/
     &       ' ')  
  945 FORMAT(' *** ERROR *** "=" SIGN EXPECTED'/' ')
  950 FORMAT(' ... READING FILE ',A/' ')
  960 FORMAT(' *** ERROR *** FAILED TO OPEN FILE ',A,' ')
  970 FORMAT(' *** ERROR *** LOGICAL UNIT NUMBER MUST BE IN ',           
     +       'THE RANGE 1...99'/' ')
  980 FORMAT(' *** ERROR *** UNIT NUMBER ',I2,' HAS NOT BEEN OPENED'
     &        /' ')
 9100	FORMAT( ' ** WARNING ** EXCEEDED MAX_CALL=',I2,' NESTED',
     &	    ' "CALL"S --- "CALL" IGNORED'/' ')
C----------------------------------------------------------------------- 
      END                                                                
