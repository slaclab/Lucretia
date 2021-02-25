      SUBROUTINE XPATH
C
C     This subroutine reads in a pathname from the next input line,
C     and passes it to XPATH_EXPAND to see whether it contains a
C     $PATH statement which needs expanding.  The full resulting
C     pathname is then associated with pointer PATH_PTR.
C 
C========1=========2=========3=========4=========5=========6=========7=C

      USE XSIF_INOUT
      USE XSIF_INTERFACES

      IMPLICIT NONE
      SAVE

      CHARACTER*80 PATHNAM
      CHARACTER, POINTER :: RET_PTR(:)

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     ignore anything further on the line containing the PATH command,
C     read the next line
C
      CALL RDLINE
      IF (ENDFIL) THEN                                                   
        CALL RDFAIL                                                      
        WRITE (IECHO,920)                                                
        WRITE (ISCRN,920)                                                
        ERROR = .TRUE.                                                   
        RETURN                                                           
      ENDIF     
C
C     get the pathname from KTEXT_ORIG to preserve original
C     capitalization; force a linefeed after that
C
      PATHNAM = KTEXT_ORIG
      ICOL = 81
C
C     pass PATHNAM to XPATH_EXPAND to be searched for any $PATH
C     variables, and to return the resulting path as RET_PTR
C
      CALL XPATH_EXPAND( PATHNAM, RET_PTR )
C
C     if XPATH_EXPAND was successful, associate PATH_PTR with
C     RET_PTR
C
      IF ( .NOT. ERROR ) THEN
          IF ( .NOT. ASSOCIATED (PATH_PTR , PATH_LCL) ) THEN
              DEALLOCATE(PATH_PTR)
          ENDIF
          PATH_PTR => RET_PTR
          NULLIFY( RET_PTR )
      ENDIF
C
C     write the new path to the echo and screen files

C      WRITE(IECHO,910)ARR_TO_STR(PATH_PTR,SIZE(PATH_PTR))
C      WRITE(ISCRN,910)ARR_TO_STR(PATH_PTR,SIZE(PATH_PTR))
      WRITE(IECHO,910)ARR_TO_STR(PATH_PTR)
      WRITE(ISCRN,910)ARR_TO_STR(PATH_PTR)



      RETURN
                                                         
C----------------------------------------------------------------------- 
  910 FORMAT(/,'0... PATH VARIABLE SET TO:',/,A,/)
  920 FORMAT(' *** ERROR *** END OF FILE SEEN WHEN READING PATH NAME'/   
     +       ' ')                                                        
C----------------------------------------------------------------------- 

      END