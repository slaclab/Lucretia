      SUBROUTINE XRETRN
C
C     member of MAD INPUT PARSER
C
C---- PERFORM "RETURN" COMMAND                                           
C----------------------------------------------------------------------- 
C	
C	MOD:
C          16-may-2003, PT:
C             if the file was opened with the CALL, FILENAME = ...
C             syntax, close it now.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C		 28-aug-1998, PT:
C			added code to handle nested CALLs
C
C     modules:
C
      USE XSIF_INOUT
      USE XSIF_INTERFACES
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE

      TYPE(XSIF_FILETYPE), POINTER :: SEARCH_PTR
C
C----------------------------------------------------------------------- 
      CALL RDTEST(';',ERROR)                                             
      IF (ERROR) RETURN                                                  
      IF ( (IDATA .EQ. 5) .OR. (NUM_CALL .EQ. 0) ) THEN                                             
        CALL RDWARN                                                      
        WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
      ELSE                                                               
		IF (ICOL .NE. 81) THEN                                           
			CALL RDWARN                                                    
			WRITE (IECHO,920)                                              
			WRITE (ISCRN,920)                                              
			ICOL = 81                                                      
		ENDIF 
          IF (IDATA.GT.99) THEN
              CALL XCLOSE_STACK_MANAGE(IDATA)
          ENDIF
		IDATA = IO_UNIT(NUM_CALL)
		NUM_CALL = NUM_CALL - 1                                                         
C        IDATA = 5                                                        
		ENDFIL = .FALSE.                                                 
		IF ( IDATA .EQ. 5 ) THEN
			WRITE (IECHO,930)
			WRITE (ISCRN,930)
		ELSEIF (IDATA .LE.99) THEN
			WRITE(IECHO,9300)IDATA
			WRITE(ISCRN,9300)IDATA
          ELSE
              SEARCH_PTR => XSIF_STACK_SEARCH( IDATA )
              IF (.NOT. ASSOCIATED(SEARCH_PTR)) THEN
                WRITE(IECHO,940)IDATA
                WRITE(ISCRN,940)IDATA
                ERROR = .TRUE.
                RETURN
              ELSE
                WRITE(IECHO,950)ARR_TO_STR(SEARCH_PTR%FILE_NAME)
                WRITE(ISCRN,950)ARR_TO_STR(SEARCH_PTR%FILE_NAME)
              ENDIF
		ENDIF                                                
      ENDIF                                                              
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' ** WARNING ** "RETURN" NOT PERMITTED ON STANDARD ',       
     +       'INPUT FILE --- "RETURN" IGNORED')                          
  920 FORMAT(' ** WARNING ** TEXT AFTER "RETURN" SKIPPED'/' ')           
  930 FORMAT('0... READING STANDARD INPUT FILE'/' ')       
  940 FORMAT(' *** ERROR *** UNABLE TO SWITCH TO UNIT NUMBER ',I6/' ')
  950 FORMAT(' ... READING FILE ',A/' ')              
 9300	FORMAT('0... READING LOGICAL UNIT NUMBER ',I2,' '/' ')
C----------------------------------------------------------------------- 
      END                                                                
