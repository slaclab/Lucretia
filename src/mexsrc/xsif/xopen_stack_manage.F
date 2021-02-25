      LOGICAL*4 FUNCTION XOPEN_STACK_MANAGE( FILNAM, UNITNO,
     &                                       FILESTAT )
C
C     function to manage opening a new file and sticking its info onto 
C     the XSIF_OPEN_STACK.  Return status = .TRUE. if everything is ok
C     and .FALSE. if an error occurred.
C
      USE XSIF_INOUT
      USE XSIF_INTERFACES

      IMPLICIT NONE
      SAVE

C     argument declarations

      CHARACTER*(*) FILNAM
      INTEGER*4 UNITNO
      CHARACTER*(*) FILESTAT

C     local declarations

      TYPE (XSIF_FILETYPE), POINTER :: SEARCH_PTR
      CHARACTER, POINTER :: FILNAM_PTR(:)
      INTEGER*4 :: IO_STATUS
      CHARACTER*7 :: FILESTAT_INTERNAL

C     referenced functions

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C     expand any $PATH values in FILNAM and return the result in
C     FILNAM_PTR

      NULLIFY( FILNAM_PTR )
      NULLIFY( SEARCH_PTR )
      CALL XPATH_EXPAND( FILNAM, FILNAM_PTR )
C
C     did XPATH_EXPAND succeed? If so, continue...
C
      IF (ERROR) GOTO 9999

      OPEN( UNIT = UNITNO, FILE = ARR_TO_STR(FILNAM_PTR), 
     &      STATUS = FILESTAT, IOSTAT = IO_STATUS )
C
C     did that succeed?  if so, add it to the stack
C
      IF ( IO_STATUS.NE.0 ) GOTO 9999
C
C     if there are open files, search to see whether this
C     logical unit number is already opened
C
C      IF (ASSOCIATED(XSIF_OPEN_STACK_HEAD)) THEN
C
C          SEARCH_PTR => XSIF_OPEN_STACK_HEAD
C
C          DO
C
C            IF (SEARCH_PTR%UNIT_NUMBER .EQ. UNITNO) THEN
C              EXIT
C            ENDIF
C            IF (.NOT.ASSOCIATED(SEARCH_PTR%NEXT_FILE) ) THEN
C              EXIT
C            ELSE
C              SEARCH_PTR => SEARCH_PTR%NEXT_FILE
C            ENDIF
C
C          ENDDO
C
C          IF (SEARCH_PTR%UNIT_NUMBER .EQ. UNITNO) THEN
          SEARCH_PTR => XSIF_STACK_SEARCH(UNITNO) 
          IF (ASSOCIATED(SEARCH_PTR)) THEN
            WRITE(ISCRN,910)ARR_TO_STR(SEARCH_PTR%FILE_NAME),UNITNO
            WRITE(IECHO,910)ARR_TO_STR(SEARCH_PTR%FILE_NAME),UNITNO
            NULLIFY(SEARCH_PTR)
            CALL XCLOSE_STACK_MANAGE( UNITNO )
          ENDIF
C
C      ENDIF
C
C     now allocate a file structure, add it to the stack, and fill it
C
      IF (.NOT. ASSOCIATED(XSIF_OPEN_STACK_HEAD)) THEN
C
          ALLOCATE(XSIF_OPEN_STACK_HEAD,STAT=IO_STATUS)
          IF (IO_STATUS .NE. 0) GOTO 9999

          XSIF_OPEN_STACK_TAIL => XSIF_OPEN_STACK_HEAD
          NULLIFY( XSIF_OPEN_STACK_TAIL%NEXT_FILE )

      ELSE

          ALLOCATE(XSIF_OPEN_STACK_TAIL%NEXT_FILE,STAT=IO_STATUS)
          IF (IO_STATUS .NE. 0) GOTO 9999

          XSIF_OPEN_STACK_TAIL => XSIF_OPEN_STACK_TAIL%NEXT_FILE
          NULLIFY( XSIF_OPEN_STACK_TAIL%NEXT_FILE )

      ENDIF

      XSIF_OPEN_STACK_TAIL%UNIT_NUMBER = UNITNO
      XSIF_OPEN_STACK_TAIL%FILE_NAME => FILNAM_PTR
      NULLIFY(FILNAM_PTR)

9999  IF ( (ERROR) .OR. (IO_STATUS.NE.0) ) THEN
          XOPEN_STACK_MANAGE = .FALSE.
      ELSE
          XOPEN_STACK_MANAGE = .TRUE.
      ENDIF

      RETURN
C----------------------------------------------------------------------- 
 910  FORMAT(' ** WARNING ** FILE ',A,' PREVIOUSLY OPENED TO UNIT '
     &      ,I6,/'  IT WILL BE CLOSED AND ITS UNIT NUMBER REASSIGNED')
C----------------------------------------------------------------------- 
      END
