	LOGICAL*4 FUNCTION MORE_LINKS( DELTA )
C
C	Increases the size of the ILIST link table.
C
C	AUTH: PT, 19-MAY-2003
C
C	MOD:

	USE XSIF_SIZE_PARS
	USE XSIF_INOUT
	USE XSIF_ELEMENTS

	IMPLICIT NONE
	SAVE

C	argument declarations

	INTEGER*4 DELTA

C	local declarations

	INTEGER*4 ALLSTAT, LOOP_1, LOOP_2
	INTEGER*4, POINTER :: ILDAT_A(:,:)

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	NULLIFY( ILDAT_A )
	ALLOCATE(ILDAT_A(MAXLST+DELTA,6),STAT=ALLSTAT)
	IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,910)MAXLST,MAXLST+DELTA
		WRITE(ISCRN,910)MAXLST,MAXLST+DELTA
		MORE_LINKS = .FALSE.
		FATAL_ALLOC_ERROR = .TRUE.
		GOTO 9999
	ENDIF

	ILDAT_A = 0

	DO LOOP_1 = 1,MAXLST
		DO LOOP_2 = 1,6
			ILDAT_A(LOOP_1,LOOP_2) = ILDAT(LOOP_1,LOOP_2)
		ENDDO
	ENDDO

	MAXLST = MAXLST + DELTA

	DEALLOCATE(ILDAT)
	NULLIFY(ILDAT)
	ILDAT => ILDAT_A
	NULLIFY(ILDAT_A)

	MORE_LINKS = .TRUE.

9999	RETURN

C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO INCREASE MAX LINK TABLE ',
     &	   'FROM ',I10,' TO ',I10,' '/' ')
	END
