	LOGICAL*4 FUNCTION MORE_CALLS( DELTA )
C
C	increases the number of entries in the IO_UNIT array.
C
C	AUTH: PT, 19-MAY-2003
C
	USE XSIF_SIZE_PARS
	USE XSIF_INOUT

	IMPLICIT NONE
	SAVE

C	argument declarations

	INTEGER*4 DELTA

C	local declarations

	INTEGER*4 LOOP_COUNT
	INTEGER*4, POINTER :: IO_UNIT_A(:)
	INTEGER*4 ALLSTAT

C	referenced functions

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	NULLIFY(IO_UNIT_A)
	ALLOCATE(IO_UNIT_A(DELTA+MXCALL),STAT = ALLSTAT)
	IF (ALLSTAT.NE.0) THEN
	  WRITE(IECHO,910)MXCALL,DELTA+MXCALL
	  WRITE(ISCRN,910)MXCALL,DELTA+MXCALL
	  FATAL_ALLOC_ERROR = .TRUE.
	  MORE_CALLS = .FALSE.
	  GOTO 9999
	ENDIF

C	move call data from old array into new one

	DO LOOP_COUNT = 1,NUM_CALL
		IO_UNIT_A(LOOP_COUNT) = IO_UNIT(LOOP_COUNT)
	ENDDO

	MXCALL = MXCALL + DELTA

C	blow away the old data

	DEALLOCATE(IO_UNIT)
	NULLIFY(IO_UNIT)

C	reference the IO_UNIT pointer

	IO_UNIT => IO_UNIT_A
	NULLIFY(IO_UNIT_A)

	MORE_CALLS = .TRUE.

 9999 RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO INCREASE MAX CALL STACK SIZE ',
     &	   'FROM ',I10,' TO ',I10,' '/' ')
	END
