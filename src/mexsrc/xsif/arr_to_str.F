      FUNCTION ARR_TO_STR( CHAR_ARRAY )
C      FUNCTION ARR_TO_STR( CHAR_ARRAY, CHAR_ARRAY_SIZE )
C
C     function which takes a character array and returns a variable-
C     length string
C
C========1=========2=========3=========4=========5=========6=========7=C

      IMPLICIT NONE
      SAVE

      CHARACTER*1                     CHAR_ARRAY(:)
      CHARACTER(LEN=SIZE(CHAR_ARRAY)) ARR_TO_STR

      INTEGER*4                       COUNT, CHAR_ARRAY_SIZE

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      CHAR_ARRAY_SIZE = SIZE(CHAR_ARRAY)
      DO COUNT = 1,CHAR_ARRAY_SIZE

          ARR_TO_STR(COUNT:COUNT) = CHAR_ARRAY(COUNT)

      END DO

      RETURN
      END FUNCTION ARR_TO_STR
