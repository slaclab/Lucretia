	LOGICAL*4 FUNCTION MORE_PARS( DELTA )
C
C	increases the size of the parameter tables
C
C	AUTH: PT, 19-MAY-2003
C
C	MOD:
C          15-DEC-2003, PT:
C             increase parameter names to 16 characters.

	USE XSIF_SIZE_PARS
	USE XSIF_ELEMENTS
	USE XSIF_INOUT

	IMPLICIT NONE
	SAVE

C	argument declarations

	INTEGER*4 DELTA

C	local declarations

	INTEGER*4 LOOP_1, LOOP_2
	INTEGER*4 ALLSTAT

	INTEGER*4, POINTER :: IPTYP_A(:),
     &					  IPLIN_A(:),
     &					  IPNEXT_A(:),
     &					  IPDAT_A(:,:)

	REAL*8, POINTER :: PDATA_A(:)
	CHARACTER(PNAME_LENGTH), POINTER :: KPARM_A(:)

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	NULLIFY( IPTYP_A, IPLIN_A, IPNEXT_A, IPDAT_A, PDATA_A, KPARM_A )

	ALLOCATE( IPTYP_A(MAXPAR+DELTA),
     &          IPLIN_A(MAXPAR+DELTA),
     &          IPNEXT_A(MAXPAR+DELTA),
     &          IPDAT_A(MAXPAR+DELTA,2),
     &          PDATA_A(MAXPAR+DELTA),
     &          KPARM_A(MAXPAR+DELTA), STAT = ALLSTAT)
	IF (ALLSTAT.NE.0) THEN
	  WRITE(IECHO,910)MAXPAR,MAXPAR+DELTA
	  WRITE(ISCRN,910)MAXPAR,MAXPAR+DELTA
	  MORE_PARS = .FALSE.
	  FATAL_ALLOC_ERROR = .TRUE.
	  GOTO 9999
	ENDIF

	IPTYP_A = 0
	IPLIN_A = 0
	IPNEXT_A = 0
	IPDAT_A = 0
	PDATA_A = 0.
	KPARM_A = ' '

	DO LOOP_1 = 1,IPARM1
		IPTYP_A(LOOP_1) = IPTYP(LOOP_1)
		IPLIN_A(LOOP_1) = IPLIN(LOOP_1)
		IPNEXT_A(LOOP_1) = IPNEXT(LOOP_1)
		IPDAT_A(LOOP_1,1) = IPDAT(LOOP_1,1)
		IPDAT_A(LOOP_1,2) = IPDAT(LOOP_1,2)
		PDATA_A(LOOP_1) = PDATA(LOOP_1)
		KPARM_A(LOOP_1) = KPARM(LOOP_1)
	ENDDO

	DO LOOP_1 = IPARM2,MAXPAR
		IPTYP_A(LOOP_1+DELTA) = IPTYP(LOOP_1)
		IPLIN_A(LOOP_1+DELTA) = IPLIN(LOOP_1)
		IPNEXT_A(LOOP_1+DELTA) = IPNEXT(LOOP_1)
		IPDAT_A(LOOP_1+DELTA,1) = IPDAT(LOOP_1,1)
		IPDAT_A(LOOP_1+DELTA,2) = IPDAT(LOOP_1,2)
		PDATA_A(LOOP_1+DELTA) = PDATA(LOOP_1)
		KPARM_A(LOOP_1+DELTA) = KPARM(LOOP_1)
	ENDDO

C	one refinement needed here is that we need to go back
C	to the element table and change the pointers from the 
C	elements into the parameter table (since the element
C	parameters are in the end of the tables and their index
C	numbers are now changed)

	DO LOOP_1 = 1,IELEM1
		IF (IETYP(LOOP_1) .GT. 0) THEN
			IEDAT(LOOP_1,1) = IEDAT(LOOP_1,1)+DELTA
			IEDAT(LOOP_1,2) = IEDAT(LOOP_1,2)+DELTA
		ENDIF
	ENDDO

C	similarly, any parameters which pointed at other parameters
C	which have been moved, we need to update the links now

	DO LOOP_1 = 1,MAXPAR+DELTA
	  IF (IPDAT_A(LOOP_1,1) .GT.IPARM1)
     &	  IPDAT_A(LOOP_1,1) = IPDAT_A(LOOP_1,1)+DELTA
	  IF (IPDAT_A(LOOP_1,2) .GT.IPARM1)
     &	  IPDAT_A(LOOP_1,2) = IPDAT_A(LOOP_1,2)+DELTA      
	ENDDO

	MAXPAR = MAXPAR + DELTA
	IPARM2 = IPARM2 + DELTA

	DEALLOCATE( IPTYP, IPDAT, IPLIN, IPNEXT, KPARM, PDATA )

	IPTYP => IPTYP_A
	IPDAT => IPDAT_A
	IPLIN => IPLIN_A
	IPNEXT => IPNEXT_A
	KPARM => KPARM_A
	PDATA => PDATA_A

	NULLIFY( IPTYP_A, IPDAT_A, IPNEXT_A, IPLIN_A, KPARM_A, PDATA_A )

	MORE_PARS = .TRUE.

 9999 RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO INCREASE PARAMETER TABLE SIZE ',
     &	   'FROM ',I10,' TO ',I10,' '/' ')
	END
