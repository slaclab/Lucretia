	LOGICAL*4 FUNCTION MORE_WAKES( DELTA )
C
C	expands the tables of wakefield file names
C
C	AUTH:  PT, 19-MAY-2003
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

	INTEGER*4 ALLSTAT, LOOP_COUNT
	TYPE (WAKEFILE), POINTER :: LWAKE_FILE_A(:)
	TYPE (WAKEFILE), POINTER :: TWAKE_FILE_A(:)

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	NULLIFY(LWAKE_FILE_A, TWAKE_FILE_A)
	ALLOCATE(LWAKE_FILE_A(MX_WAKEFILE+DELTA),
     &	     TWAKE_FILE_A(MX_WAKEFILE+DELTA),
     &		 STAT=ALLSTAT)
	IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,910)MX_WAKEFILE,MX_WAKEFILE+DELTA
		WRITE(ISCRN,910)MX_WAKEFILE,MX_WAKEFILE+DELTA
		MORE_WAKES = .FALSE.
		FATAL_ALLOC_ERROR = .TRUE.
		GOTO 9999
	ENDIF

	DO LOOP_COUNT = 1,MX_WAKEFILE+DELTA
	  NULLIFY(LWAKE_FILE_A(LOOP_COUNT)%FNAM_PTR,
     &     	  TWAKE_FILE_A(LOOP_COUNT)%FNAM_PTR )
	ENDDO

	DO LOOP_COUNT = 1,MX_WAKEFILE
		LWAKE_FILE_A(LOOP_COUNT)%FNAM_PTR =>
     &	 LWAKE_FILE(LOOP_COUNT)%FNAM_PTR
		TWAKE_FILE_A(LOOP_COUNT)%FNAM_PTR =>
     &	 TWAKE_FILE(LOOP_COUNT)%FNAM_PTR
	ENDDO

	MX_WAKEFILE = MX_WAKEFILE + DELTA

	DEALLOCATE(LWAKE_FILE,TWAKE_FILE)
	LWAKE_FILE => LWAKE_FILE_A
	TWAKE_FILE => TWAKE_FILE_A
	NULLIFY(LWAKE_FILE_A,TWAKE_FILE_A)

	MORE_WAKES = .TRUE.

9999	RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO INCREASE MAX WAKE FILE COUNT ',
     &	   'FROM ',I10,' TO ',I10,' '/' ')
	END
