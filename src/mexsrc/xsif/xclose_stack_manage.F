      SUBROUTINE XCLOSE_STACK_MANAGE( UNITNO )
C
C     close a logical unit which has previously been opened and
C     adjust the opened-file stack accordingly
C
      USE XSIF_INOUT

      IMPLICIT NONE
      SAVE

C     argument declarations

      INTEGER*4 UNITNO

C     local declarations

      TYPE (XSIF_FILETYPE), POINTER :: PREV_FILE, THIS_FILE, NEXT_FILE

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      NULLIFY( PREV_FILE )
      NULLIFY( THIS_FILE )
      NULLIFY( NEXT_FILE )
C
C     first handle the case of no files opened at all
C
      IF (.NOT. ASSOCIATED(XSIF_OPEN_STACK_HEAD) ) RETURN
C
C     now the case of opened files
C
      THIS_FILE => XSIF_OPEN_STACK_HEAD

      DO

        IF (ASSOCIATED(THIS_FILE%NEXT_FILE))
     &    NEXT_FILE => THIS_FILE%NEXT_FILE

        IF (THIS_FILE%UNIT_NUMBER.EQ.UNITNO) EXIT
        IF (.NOT.ASSOCIATED(NEXT_FILE)) EXIT
        PREV_FILE => THIS_FILE
        THIS_FILE => NEXT_FILE
        NULLIFY(NEXT_FILE)

      ENDDO
C
C     at this point either we found the record for the file to be
C     closed, or we reached the end of the stack.  In the former
C     case, close the file, deallocate its record, and "close the
C     hole" in the list:
C
      IF (THIS_FILE%UNIT_NUMBER.EQ.UNITNO) THEN
C
          CLOSE(UNIT=UNITNO)
C
C     at the risk of being rather pedantic, there are 4 cases to
C     consider:
C
C         Several files are opened, and we close the first
C                                                    last
C                                      or one in the middle
C         or only one file is opened, and we close it.
C
C     First possibility:  we are closing the first file
C
          IF (.NOT.ASSOCIATED(PREV_FILE)) THEN
            NULLIFY(XSIF_OPEN_STACK_HEAD)
C
C     Is it the only file?
C    
            IF (.NOT. ASSOCIATED(NEXT_FILE)) THEN
              NULLIFY(XSIF_OPEN_STACK_TAIL)  
            ELSE
              XSIF_OPEN_STACK_HEAD => NEXT_FILE
            ENDIF
C
C     now for the case of clearing the last file (where there
C     are more than 1 open)
C
          ELSEIF (.NOT.ASSOCIATED(NEXT_FILE)) THEN
            XSIF_OPEN_STACK_TAIL => PREV_FILE
            NULLIFY(XSIF_OPEN_STACK_TAIL%NEXT_FILE)   
C
C     now for the case of clearing one in the middle
C
          ELSE
            PREV_FILE%NEXT_FILE => NEXT_FILE
          ENDIF
C
C     in any case, remove the closed unit
C
          DEALLOCATE(THIS_FILE)
C
      ENDIF
      RETURN
      END