      LOGICAL*4 FUNCTION ARRCMP ( ARRAY1 , ARRAY2 )
C
C     compares ARRAY1 and ARRAY2 element-by-element; if they are
C     itentical returns TRUE otherwise returns FALSE.  ARRAY1 and
C     ARRAY2 are constrained to be CHAR*1 arrays.
C
C========1=========2=========3=========4=========5=========6=========7=C

      IMPLICIT NONE
      SAVE

C     argument declarations

      CHARACTER*1 ARRAY1(:)
      CHARACTER*1 ARRAY2(:)

C     local declarations

      INTEGER*4 COUNT

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     if the two arrays are not the same size, then right away they
C     are not equal
C
      IF ( SIZE( ARRAY1 ) .NE. SIZE( ARRAY2 ) ) THEN
          ARRCMP = .FALSE.
          GOTO 9999
      ENDIF

      DO COUNT = 1,SIZE( ARRAY1 )
          IF ( ARRAY1(COUNT) .NE. ARRAY2(COUNT) ) THEN
              ARRCMP = .FALSE.
              GOTO 9999
          ENDIF
      END DO
C
C     if we have gotten here then they must be the same
C
      ARRCMP = .TRUE.

 9999 RETURN
      END