      FUNCTION XSIF_STACK_SEARCH( UNITNO )
C
C     searches for the open file with unit UNITNO, returns a pointer to
C     it if found or a null pointer otherwise
C
      USE XSIF_INOUT

      IMPLICIT NONE
      SAVE

      TYPE (XSIF_FILETYPE), POINTER :: XSIF_STACK_SEARCH
      INTEGER*4 UNITNO

      TYPE(XSIF_FILETYPE), POINTER :: SEARCH_PTR

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      NULLIFY(SEARCH_PTR)
      NULLIFY(XSIF_STACK_SEARCH)

      IF (ASSOCIATED(XSIF_OPEN_STACK_HEAD)) THEN

          SEARCH_PTR => XSIF_OPEN_STACK_HEAD

          DO

            IF (SEARCH_PTR%UNIT_NUMBER .EQ. UNITNO) THEN
              EXIT
            ENDIF
            IF (.NOT.ASSOCIATED(SEARCH_PTR%NEXT_FILE) ) THEN
              EXIT
            ELSE
              SEARCH_PTR => SEARCH_PTR%NEXT_FILE
            ENDIF

          ENDDO

          IF (SEARCH_PTR%UNIT_NUMBER .EQ. UNITNO) THEN
              XSIF_STACK_SEARCH => SEARCH_PTR
          ENDIF

      ENDIF

      RETURN
      END
