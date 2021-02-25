      SUBROUTINE VMS_TIMEDATE( KDATE_8, KTIME_10, IVAL_2, 
     &                         KDATE_11, KTIME_8 )
C
C     member of MAD INPUT PARSER
C
C==================================================================
C
C     modules:
C
      IMPLICIT NONE
      SAVE
C
C     argument declarations
C
      CHARACTER(8),  INTENT(IN)  :: KDATE_8
      CHARACTER(10), INTENT(IN)  :: KTIME_10
      INTEGER(4),    INTENT(IN)  :: IVAL_2
      CHARACTER(11), INTENT(OUT) :: KDATE_11
      CHARACTER(8),  INTENT(OUT) :: KTIME_8
C
C     local declarations
C
      CHARACTER(3), DIMENSION(12) :: MONTH =      ! 3-char month names
     &     (/'JAN','FEB','MAR','APR','MAY','JUN',
     &     'JUL','AUG','SEP','OCT','NOV','DEC'/)
C
C     This subroutine takes a date and a time in the format 
C     returned by the DATE_AND_TIME subroutine, and the integer
C     representing the month number (also from DATE_AND_TIME),
C     and returns a VMS-like date of DD-MMM-YYYY HH:MM:SS.
C
C=====================================================================

      KDATE_11 = KDATE_8(7:8) // '-' // MONTH(IVAL_2) // '-' //
     &           KDATE_8(1:4)

      KTIME_8 = KTIME_10(1:2) // ':' // KTIME_10(3:4) // ':' //
     &          KTIME_10(5:6)

      RETURN
      END
