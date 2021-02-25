      SUBROUTINE XSIF_IO_CLOSE
C
C     closes files opened by XSIF_IO_SETUP.
C
C     AUTH: PT, 05-JAN-2001
C
C     MOD:
C          14-may-2003, PT:
C             use the xsif open stack when closing files.
C          31-JAN-2001, PT:
C             if there are nested CALLs which are not yet
C             resolved, close all remaining files.  Such a situation
C             (remaining open files via CALL on exit from xsif) can
C             only happen if there is a serious failure during parsing.
C
      USE XSIF_INOUT

      IMPLICIT NONE

      TYPE (XSIF_FILETYPE), POINTER :: THIS_FILE, LAST_FILE   

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

c      CLOSE( IECHO )
c      CLOSE( IDATA )
c      CLOSE( IPRNT )
c
c      IF ( NUM_CALL .GT. 0 ) THEN
c          DO CALL_COUNT = 1,NUM_CALL-1
c              CLOSE( IO_UNIT(CALL_COUNT) )
c          ENDDO
c      ENDIF

      IF (ASSOCIATED(XSIF_OPEN_STACK_HEAD)) THEN

        THIS_FILE => XSIF_OPEN_STACK_HEAD

        DO
          CLOSE(THIS_FILE%UNIT_NUMBER)
          LAST_FILE => THIS_FILE
          IF (ASSOCIATED(LAST_FILE%NEXT_FILE)) THEN
            THIS_FILE => LAST_FILE%NEXT_FILE
            DEALLOCATE(LAST_FILE)
            NULLIFY(LAST_FILE)
          ELSE
            EXIT
          ENDIF
        ENDDO
        DEALLOCATE(THIS_FILE)
        NULLIFY(THIS_FILE)
        NULLIFY(XSIF_OPEN_STACK_HEAD)
        NULLIFY(XSIF_OPEN_STACK_TAIL)
	  DEALLOCATE(IO_UNIT)
	  NULLIFY(IO_UNIT)

      ENDIF


      RETURN
      END