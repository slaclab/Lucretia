      SUBROUTINE XOPEN                                                   
C---- OPEN A FILE                                                        
C----------------------------------------------------------------------- 
C
C	MOD:
C          14-may-2003, PT:
C             modify to support use of the xsif open stack (linked
C             list of opened files).
C          30-mar-2001, PT:
C             suppress leading blanks in file specification to
C             appease UNIX.
C          11-jan-2001, PT:
C             eliminate use of the DIMAT header for XSIF library.
C          11-JAN-2000, PT:
C             changes to support new path variable.
C		 19-oct-1998, PT:
C			use KTEXT_ORIG so that filenames have their original
C			upper/lower case prserved (important for UNIX systems).
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C		 28-aug-1998, PT:
C		    if the file to be opened is opened to the ECHO unit,
C			or the PRINT repeat the DIMAD header from MADIN.
C
C     modules:
C
      USE XSIF_INOUT
      USE XSIF_INTERFACES
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
      CHARACTER*80      FILNAM  
      CHARACTER, POINTER :: FILNAM_PTR(:)   
      INTEGER*4 FIRST_CHAR_INDX, LOOP_COUNTER
      CHARACTER*1 :: KBLANK = ' '     
      LOGICAL*4 XOPEN_STATUS, XOPEN_STACK_MANAGE                                 
C----------------------------------------------------------------------- 
      CALL RDFILE(IOPEN,ERROR)                                           
      IF (ERROR) RETURN                                                  
      IF (ICOL .NE. 81) THEN                                             
        CALL RDWARN                                                      
        WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
        ICOL = 81                                                        
      ENDIF                                                              
      CALL RDLINE                                                        
      IF (ENDFIL) THEN                                                   
        CALL RDFAIL                                                      
        WRITE (IECHO,920)                                                
        WRITE (ISCRN,920)                                                
        ERROR = .TRUE.                                                   
        RETURN                                                           
      ENDIF
C
C     find first non-blank character
C
      DO LOOP_COUNTER = 1,80
          IF (KTEXT_ORIG(LOOP_COUNTER:LOOP_COUNTER).NE.KBLANK) THEN
              FIRST_CHAR_INDX = LOOP_COUNTER
              EXIT
          ENDIF
      ENDDO

      FILNAM = KTEXT_ORIG(FIRST_CHAR_INDX:80)
C
C     expand any $PATH values in FILNAM and return the result
C     in FILNAM_PTR
C
c      NULLIFY( FILNAM_PTR )
c      CALL XPATH_EXPAND( FILNAM, FILNAM_PTR )
c      IF ( .NOT. ERROR ) THEN                                                     
c          ICOL = 81                                                          
c          OPEN (UNIT=IOPEN,FILE=ARR_TO_STR(FILNAM_PTR),
c     &        STATUS='UNKNOWN',ERR=800)
c          WRITE (IECHO,930) ARR_TO_STR(FILNAM_PTR)                                          
c          WRITE (ISCRN,930) ARR_TO_STR(FILNAM_PTR)      
c          DEALLOCATE( FILNAM_PTR )                                   
c          RETURN                  
c      ENDIF                                           
      XOPEN_STATUS = XOPEN_STACK_MANAGE( FILNAM, IOPEN, 'UNKNOWN' )
      IF (.NOT.XOPEN_STATUS) THEN
  800   CALL RDWARN                                                        
        WRITE (IECHO,940) FILNAM                                           
        WRITE (ISCRN,940) FILNAM  
      ELSE
        WRITE(IECHO,930) ARR_TO_STR(XSIF_OPEN_STACK_TAIL%FILE_NAME)
        WRITE(ISCRN,930) ARR_TO_STR(XSIF_OPEN_STACK_TAIL%FILE_NAME)
      ENDIF
      ICOL=81                     
      RETURN                                                             
C----------------------------------------------------------------------- 
 9100 FORMAT('1DIMAT VERSION ',A/                                        
     +       '0DATE AND TIME OF THIS RUN:     ',A,5X,A)                  
 9200 FORMAT('0INPUT STREAM AND MESSAGE LOG:'/' ')                       
  910 FORMAT(' ** WARNING ** TEXT AFTER "OPEN" SKIPPED'/' ')             
  920 FORMAT(' *** ERROR *** END OF FILE SEEN WHEN READING FILE NAME'/   
     +       ' ')                                                        
  930 FORMAT('0... SUCCESSFUL TO OPEN FILE: ',A)                          
  940 FORMAT(' ** WARNING ** FAIL TO OPEN FILE: ',A/' ')                 
C----------------------------------------------------------------------- 
      END
