      MODULE XSIF_INTERFACES
C
C     module containing explicit interfaces for any XSIF routine
C     requiring same.  Descended from DIMAD_INTERFACES.
C
C     AUTH: PT, 05-JAN-2001
C
C     MOD:
C          26-jun-2003, PT:
C             add interface for XSIF_STACK_SEARCH
C          05-jan-2001, PT:
C             added interface for XSIF_CMD_LOOP
C
      IMPLICIT NONE
      SAVE

C========1=========2=========3=========4=========5=========6=========7=C

      INTERFACE

          FUNCTION ARR_TO_STR ( CHAR_ARRAY )
                CHARACTER*1 CHAR_ARRAY(:)
                CHARACTER(LEN=SIZE(CHAR_ARRAY)) ARR_TO_STR
          END FUNCTION ARR_TO_STR

          SUBROUTINE XPATH_EXPAND ( PATHSTRING , RET_PTR )
              CHARACTER(LEN=*) PATHSTRING
              CHARACTER, POINTER :: RET_PTR(:)
          END SUBROUTINE XPATH_EXPAND

          FUNCTION ARRCMP ( ARR1, ARR2 )
              LOGICAL*4 ARRCMP
              CHARACTER*1 ARR1(:)
              CHARACTER*1 ARR2(:)
          END FUNCTION ARRCMP

          FUNCTION XSIF_CMD_LOOP ( XSIF_EXTRA_CMD )
              INTEGER*4 XSIF_CMD_LOOP
              LOGICAL*4, EXTERNAL, OPTIONAL :: XSIF_EXTRA_CMD
          END FUNCTION XSIF_CMD_LOOP

          FUNCTION XSIF_STACK_SEARCH( UNITNO )
              use xsif_inout
              TYPE(XSIF_FILETYPE), POINTER :: XSIF_STACK_SEARCH
              INTEGER*4 :: UNITNO
          END FUNCTION XSIF_STACK_SEARCH

      END INTERFACE

      END MODULE XSIF_INTERFACES
