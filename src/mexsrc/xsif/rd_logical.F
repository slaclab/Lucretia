      SUBROUTINE RD_LOGICAL( LOG_VAL, ERR_FLG )
C
C     subroutine to decode a MAD logical into a FORTRAN logical.
C
      USE XSIF_INOUT
      IMPLICIT NONE

C     argument declarations

      LOGICAL*4 LOG_VAL, ERR_FLG

C     local declarations

      CHARACTER*8 MAD_LOGICALS(8)
      DATA MAD_LOGICALS /
     &'.YES.   ','.TRUE.  ','.T.     ','.ON.    ',
     &'.NO.    ','.FALSE. ','.F.     ','.OFF.   ' /
      CHARACTER*8 NEW_WORD, MAD_LOG
      INTEGER*4 L_WORD, WORD_INDX, WORD_COUNT

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      ERR_FLG = .FALSE.

C     the first character is supposed to be a period; skip it
      
      IF (KLINE(ICOL).NE.'.') GOTO 200
      CALL RDSKIP('.')
      IF (FATAL_READ_ERROR) GOTO 9999

C     Get the rest of the word

      CALL RDWORD(NEW_WORD,L_WORD)
      IF (FATAL_READ_ERROR) GOTO 9999

C     build an 8 char or less word with '.' as first char

      MAD_LOG(1:1) = '.'
      L_WORD = MIN(L_WORD+1,8)
      MAD_LOG(2:L_WORD) = NEW_WORD(1:L_WORD-1)
      IF (L_WORD .LT. 8) THEN
        DO WORD_COUNT = L_WORD+1,8
          MAD_LOG(WORD_COUNT:WORD_COUNT) = ' '
        ENDDO
      ENDIF

C     perform lookup

      CALL RDLOOK(MAD_LOG,L_WORD,MAD_LOGICALS,1,8,WORD_INDX)

C     Translate the lookup

      SELECT CASE( WORD_INDX )

          CASE( 1, 2, 3, 4 )
              LOG_VAL = .TRUE.
              GOTO 9999

          CASE( 5, 6, 7, 8 )
              LOG_VAL = .FALSE.
              GOTO 9999

      END SELECT

C     in the event of a failure, jump to here

200   CALL RDFAIL
      WRITE(IECHO,910)
      WRITE(ISCRN,910)
      ERR_FLG = .TRUE.

9999  CONTINUE
      RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** LOGICAL VALUE REQUIRED'/' ')
      END