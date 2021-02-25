      SUBROUTINE XPATH_EXPAND( PATHSTRING , RET_PTR )
C
C     searches PATHSTRING for the $XPATH variable; if found, it will
C     expand PATHSTRING using the existing path and return a pointer
C     to the expanded string as RET_PTR
C
C========1=========2=========3=========4=========5=========6=========7=C

      USE XSIF_INOUT

      IMPLICIT NONE
      SAVE

C     argument declarations

      CHARACTER(LEN=*), INTENT(IN) :: PATHSTRING
      CHARACTER, POINTER :: RET_PTR(:)

C     local declarations

      INTEGER*4 COUNT, INDX_$, LENPATH, SIZE_RETSTR, N_$PATH, ALL_STAT
      INTEGER*4 POS_PATHSTR, POS_RETPTR, COUNT2

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     search the PATHSTRING for all occurrences of $PATH, to determine
C     how long a string needs to be returned

      NULLIFY(RET_PTR)

      INDX_$  = 0
      LENPATH = LEN_TRIM(PATHSTRING)
      N_$PATH = 0
C
C     no point looking at PATHSTRING for $PATH unless PATHSTRING is
C     at least 5 chars long...
C
      IF ( LENPATH .GE. 5 ) THEN

          DO COUNT = 1,LENPATH-4
              IF ( PATHSTRING(COUNT:COUNT+4) .EQ. '$PATH' ) THEN
                  N_$PATH = N_$PATH + 1
              ENDIF
          ENDDO
      

          SIZE_RETSTR = LENPATH + N_$PATH * (SIZE(PATH_PTR) - 5)
C
C     allocate the return pointer to be size required above; if failure
C     occurs, notify the user and set error flag to notify CNTROL
C
          ALLOCATE( RET_PTR(SIZE_RETSTR), STAT=ALL_STAT )
          IF ( ALL_STAT .NE. 0 ) THEN
              WRITE(IECHO,910)SIZE_RETSTR
              WRITE(ISCRN,910)SIZE_RETSTR
              ERROR = .TRUE.
              GOTO 9999
          ENDIF

          POS_PATHSTR    = 0
          POS_RETPTR     = 0
C
C     copy and expand PATHSTRING into RET_PTR
C
          DO COUNT = 1,LENPATH

              POS_PATHSTR = POS_PATHSTR + 1
              POS_RETPTR = POS_RETPTR + 1

              IF ( (PATHSTRING(POS_PATHSTR:POS_PATHSTR+4) 
     &                                        .EQ. '$PATH') ) THEN
                 
                  POS_PATHSTR = POS_PATHSTR + 4
                  
                  DO COUNT2 = 1,SIZE(PATH_PTR)
                      RET_PTR(POS_RETPTR+COUNT2-1) = PATH_PTR(COUNT2)
                  ENDDO
                  POS_RETPTR = POS_RETPTR + SIZE(PATH_PTR) - 1

              ELSE

                  RET_PTR(POS_RETPTR) = 
     &                PATHSTRING(POS_PATHSTR:POS_PATHSTR)
              ENDIF

              IF ( POS_PATHSTR .GE. LENPATH-4 ) EXIT

          ENDDO
C
C     if we did not reach the end of the PATHSTRING, copy any leftover
C     stuff from the end of PATHSTRING into RET_PTR

          IF ( POS_PATHSTR .LT. LENPATH ) THEN
              DO COUNT2 = POS_PATHSTR+1,LENPATH
                  RET_PTR(SIZE_RETSTR-LENPATH+COUNT2) =
     &                PATHSTRING(COUNT2:COUNT2)
              ENDDO
          ENDIF

      ELSE  ! PATHSTRING is under 5 chars in length, just copy it

          SIZE_RETSTR = LENPATH

          ALLOCATE( RET_PTR(SIZE_RETSTR), STAT=ALL_STAT )
          IF ( ALL_STAT .NE. 0 ) THEN
              WRITE(IECHO,910)SIZE_RETSTR
              WRITE(ISCRN,910)SIZE_RETSTR
              ERROR = .TRUE.
              GOTO 9999
          ENDIF

          DO COUNT2 = 1,SIZE_RETSTR
              RET_PTR(COUNT2) = PATHSTRING(COUNT2:COUNT2)
          ENDDO

      ENDIF

9999  RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE LEN=',I5,' STRING',/,
     &       '               IN "XPATH_EXPAND" SUBROUTINE',/)
C----------------------------------------------------------------------- 

      END

