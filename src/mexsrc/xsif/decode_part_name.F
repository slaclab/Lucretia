      SUBROUTINE DECODE_PART_NAME( PNAME, LNAME, IEP1 )
C
C     decodes a particle name string and converts it to an
C     equivalent real constant for storage in the XSIF data
C     tables.
C
C     Auth:  PT, 15-nov-2002
C
C     Mod:
C
C========1=========2=========3=========4=========5=========6=========7=C

      USE XSIF_ELEM_PARS
      USE XSIF_ELEMENTS
      USE XSIF_INOUT

      IMPLICIT NONE

C     argument declarations

      CHARACTER*(*) PNAME
      INTEGER*4 LNAME
      INTEGER*4 IEP1

C     local declarations

      LOGICAL*4 PNAME_FOUND
      CHARACTER*8 PNAME8
      INTEGER*4 LNAME8, COUNT
      INTEGER*4 PARTNO

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                 C  O  D  E                                           C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      PARTNO = 0

C     generate shortened version of PNAME

      IF (LNAME .LT. 8) THEN
          PNAME8(1:LNAME) = PNAME
          DO COUNT = LNAME+1,8
            PNAME8(COUNT:COUNT) = ' '
          ENDDO
      ELSE
          PNAME8 = PNAME(1:8)
      ENDIF

C     look it up in the dictionary of particles

      CALL RDLOOK(PNAME8,8,DMADPART,1,NMADPART,PARTNO)    
      IF (PARTNO .EQ. 0) THEN
          CALL RDWARN
          WRITE(IECHO,910)PNAME
          WRITE(ISCRN,910)PNAME
          PARTNO = 1
      ENDIF
      PDATA(IEP1) = DBLE(PARTNO)
      IPTYP(IEP1) = 0
      

      RETURN

C-----------------------------------------------------------------------        

  910 FORMAT(' ** WARNING ** UNKNOWN PARTICLE "',A,
     &        '" -- "POSITRON" ASSUMED'/' ')

C-----------------------------------------------------------------------        
      END