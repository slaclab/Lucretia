      INTEGER*4 FUNCTION PARCHK( ERROR_MODE )
C
C     This function examines all the XSIF parameters and ensures that they
C     are all defined.  If undefined parameters are detected, PARCHK returns
C     an error status, otherwise a 0 status is returned.
C
C     AUTH: PT, 08-jan-2001
C
C     MOD:
C
      USE XSIF_INOUT
      USE XSIF_ELEMENTS

      IMPLICIT NONE
      SAVE

C     argument declarations

      LOGICAL*4 ERROR_MODE        ! should we treat undefined pars as
                                  ! WARNINGs or ERRORs?

C     local declarations

      INTEGER*4 COUNT

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     Loop from 1 to IPARM1, looking to see whether IPTYP is still
C     set to -1 (which signifies undefined parameter).  If so, set
C     error status and send a message.
C

      PARCHK = 0

      DO COUNT = 1,IPARM1
          IF (IPTYP(COUNT) .EQ. -1) THEN
              IF (ERROR_MODE) THEN
                  WRITE(IECHO,910)KPARM(COUNT)
                  WRITE(ISCRN,910)KPARM(COUNT)
                  PARCHK = -1 * ABS(XSIF_PAR_NODEFINE)
              ELSE
                  WRITE(IECHO,920)KPARM(COUNT)
                  WRITE(ISCRN,920)KPARM(COUNT)
                  PARCHK = ABS(XSIF_PAR_NODEFINE)
              ENDIF
          ENDIF
      ENDDO

      RETURN

C----------------------------------------------------------------------- 

  910 FORMAT(' *** ERROR *** PARAMETER "',
     &    A8,'" NOT DEFINED IN INPUT.'/' ' )
  920 FORMAT(' ** WARNING ** PARAMETER "',
     &    A8,'" NOT DEFINED IN INPUT.'/' ' )

C----------------------------------------------------------------------- 

      END