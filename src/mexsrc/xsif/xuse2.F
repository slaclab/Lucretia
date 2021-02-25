      INTEGER*4 FUNCTION XUSE2( BEAMLINENAME )
C
C     Alternate method of specifying a beamline for expansion.  The
C     standard method is a USE, beamlinename statement in the XSIF
C     file.  This subroutine allows the calling routine to specify a
C     beamline to be expanded regardless of whether a USE statement is
C     present in the decks.  It works by copying the string BEAMLINENAME
C     into KTEXT (XSIF's line buffer) and calling the standard XUSE
C     subroutine.
C
C     AUTH: PT, 05-jan-2001
C
C     MOD:
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C
      USE XSIF_INOUT

      IMPLICIT NONE
      SAVE

C     argument declarations

      CHARACTER(*)    BEAMLINENAME        ! name of the beamline

C     local declarations

      INTEGER*4 NAMELENGTH, COUNT, I, IND

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      ERROR = .FALSE.

C     copy the first 80 characters of BEAMLINENAME into KTEXT

      NAMELENGTH = MIN(80,LEN_TRIM(BEAMLINENAME))

      DO COUNT = 1,80
          IF ( COUNT .LE. NAMELENGTH ) THEN
              KTEXT(COUNT:COUNT) = BEAMLINENAME(COUNT:COUNT)
          ELSE
              KTEXT(COUNT:COUNT) = ' '
          ENDIF
      END DO
      KTEXT_ORIG = KTEXT
C
C     convert to upper case
C
        DO 1 I=1,80                                                      
          IND = INDEX(LOTOUP, KLINE(I))                                  
          IF(IND.NE.0) KLINE(I) = UPTOLO(IND)                            
 1      CONTINUE             

      ICOL = 1
C
C     call XUSE with a flag set so that it does not read a line
C
      XUSE_FROM_FILE = .FALSE.
      CALL XUSE
      IF (FATAL_READ_ERROR) GOTO 9999

      IF ( ERROR ) THEN
          XUSE2 = XSIF_PARSE_ERROR
      ELSE
          XUSE2 = 0                                            
      ENDIF

9999  IF (FATAL_READ_ERROR) ERROR = .TRUE.


      RETURN
      END