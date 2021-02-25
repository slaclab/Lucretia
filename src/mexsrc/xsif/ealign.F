	SUBROUTINE EALIGN
C
C	Parses and implements the XSIF emulation of a MAD EALIGN
C     statement.
C
C     MOD:
C          25-feb-2004, PT:
C             moved EALIGN_DICT to XSIF_ELEM_PARS.
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
	USE XSIF_ELEMENTS
	USE XSIF_ELEM_PARS
C
      IMPLICIT NONE
      SAVE
      
C     local declarations

      CHARACTER(ENAME_LENGTH) EALIGN_NAME
      INTEGER*4 :: EALIGN_COUNT = 0

	INTEGER*4 I_ELEM, I_EP1, I_EP2
	INTEGER*4 ELEM_COUNT
      
C     referenced functions

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C     increment the EALIGN counter

      EALIGN_COUNT = EALIGN_COUNT + 1
      
C     generate a unique name for the element

      WRITE(EALIGN_NAME,910)EALIGN_COUNT

C     allocate an element cell for the EALIGN

      CALL FNDELM(ILCOM,EALIGN_NAME,I_ELEM)
      
C     allocate parameter space for the EALIGN

      CALL DEFPAR( ED_SIZE, EALIGN_DICT, I_EP1, I_EP2, ILCOM )

C     decode parameter list

      CALL DECPAR(I_EP1, I_EP2, KTYPE, KLABL, ERROR)
	IF (FATAL_READ_ERROR) GOTO 9999
	IF (ERROR) GOTO 9999

C     if no error, perform final assignment of element data

      IEDAT(I_ELEM,1) = I_EP1
	IEDAT(I_ELEM,2) = I_EP2

C     If any elements in ITEM have their ERRFLG set right now,
C     point them at this particular EALIGN

      DO ELEM_COUNT = NPOS1,NPOS2
	  IF (ERRFLG(ELEM_COUNT)) ERRPTR(ELEM_COUNT) = I_ELEM
	ENDDO


9999	RETURN

C========1=========2=========3=========4=========5=========6=========7=C

 910  FORMAT('*E',I5.5,'*        ')
 
C========1=========2=========3=========4=========5=========6=========7=C

      END  	