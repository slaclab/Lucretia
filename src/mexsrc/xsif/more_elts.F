	LOGICAL*4 FUNCTION MORE_ELTS( DELTA )
C
C	increases the size of the element table by increment DELTA
C
C     MOD:
C          15-DEC-2003, PT:
C             expand element names to 16 characters.
C
	USE XSIF_SIZE_PARS
	USE XSIF_ELEMENTS
	USE XSIF_INOUT

	IMPLICIT NONE
	SAVE

C	argument declarations

	INTEGER*4 DELTA

C	local declarations

	INTEGER*4, POINTER :: IELIN_A(:),
     &					  IETYP_A(:),
     &					  IEDAT_A(:,:),
     &					  IECNT_A(:)
	CHARACTER(ENAME_LENGTH), POINTER :: KELEM_A(:)
	LOGICAL*4, POINTER :: ELEM_LOCKED_A(:)

	CHARACTER(ETYPE_LENGTH), POINTER ::  KETYP_A(:)        ! type
      CHARACTER(ELABL_LENGTH), POINTER ::  KELABL_A(:)       ! label

	INTEGER*4 LOOP_1, ALLSTAT

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	NULLIFY( IELIN_A, IETYP_A, IEDAT_A, IECNT_A, ELEM_LOCKED_A,
     &		 KELEM_A, KETYP_A, KELABL_A )

	ALLOCATE( IELIN_A(MAXELM+DELTA), IETYP_A(MAXELM+DELTA),
     &		  IEDAT_A(MAXELM+DELTA,3), IECNT_A(MAXELM+DELTA),
     &		  ELEM_LOCKED_A(MAXELM+DELTA), KELEM_A(MAXELM+DELTA),
     &		  KETYP_A(MAXELM+DELTA), KELABL_A(MAXELM+DELTA),
     &		  STAT = ALLSTAT )
	IF (ALLSTAT.NE.0) THEN
	  WRITE(IECHO,910)MAXELM,MAXELM+DELTA
	  WRITE(ISCRN,910)MAXELM,MAXELM+DELTA
	  MORE_ELTS = .FALSE.
	  FATAL_ALLOC_ERROR = .TRUE.
	  GOTO 9999
	ENDIF

	IELIN_A = 0
	IETYP_A = 0
	IEDAT_A = 0
	IECNT_A = 0
	ELEM_LOCKED_A = .FALSE.
	KELEM_A = ' '
	KETYP_A = ' '
	KELABL_A = ' '

	DO LOOP_1 = 1,IELEM1
	  IELIN_A(LOOP_1) = IELIN(LOOP_1)
	  IETYP_A(LOOP_1) = IETYP(LOOP_1)
	  IEDAT_A(LOOP_1,1) = IEDAT(LOOP_1,1)
	  IEDAT_A(LOOP_1,2) = IEDAT(LOOP_1,2)
	  IEDAT_A(LOOP_1,3) = IEDAT(LOOP_1,3)
	  IECNT_A(LOOP_1) = IECNT(LOOP_1)
	  ELEM_LOCKED_A(LOOP_1) = ELEM_LOCKED(LOOP_1)
	  KELEM_A(LOOP_1) = KELEM(LOOP_1)
	  KETYP_A(LOOP_1) = KETYP(LOOP_1)
	  KELABL_A(LOOP_1) = KELABL(LOOP_1)
	ENDDO

	DO LOOP_1 = IELEM2,MAXELM
	  IELIN_A(LOOP_1+DELTA) = IELIN(LOOP_1)
	  IETYP_A(LOOP_1+DELTA) = IETYP(LOOP_1)
	  IEDAT_A(LOOP_1+DELTA,1) = IEDAT(LOOP_1,1)
	  IEDAT_A(LOOP_1+DELTA,2) = IEDAT(LOOP_1,2)
	  IEDAT_A(LOOP_1+DELTA,3) = IEDAT(LOOP_1,3)
	  IECNT_A(LOOP_1+DELTA) = IECNT(LOOP_1)
	  ELEM_LOCKED_A(LOOP_1+DELTA) = ELEM_LOCKED(LOOP_1)
	  KELEM_A(LOOP_1+DELTA) = KELEM(LOOP_1)
	  KETYP_A(LOOP_1+DELTA) = KETYP(LOOP_1)
	  KELABL_A(LOOP_1+DELTA) = KELABL(LOOP_1)
	ENDDO

C	The expansion process has changed the indices of formal 
C	arguments to beamlines.  Update the links now.

	DO LOOP_1 = 1,IELEM1
		IF ( (IETYP_A(LOOP_1) .EQ. 0) .AND.
     &		 (IEDAT_A(LOOP_1,1).GT.IELEM1)  ) THEN
			IEDAT_A(LOOP_1,1) = IEDAT_A(LOOP_1,1) + DELTA
			IEDAT_A(LOOP_1,2) = IEDAT_A(LOOP_1,2) + DELTA
		ENDIF
	ENDDO

C	now do the same thing in the link table

	DO LOOP_1 = 1,IUSED
		IF ( ILDAT(LOOP_1,5) .GT. IELEM1 ) THEN
			ILDAT(LOOP_1,5) = ILDAT(LOOP_1,5) + DELTA
		ENDIF
	ENDDO

C	Finally, if a beamline expansion has been performed, 
C	the pointers in the ITEM table which indicate start or
C	end of a beamline have changed.  Update these now.

	IF (ALLOCATED(ITEM)) THEN
	  DO LOOP_1 = 1,MAXPOS
		IF (ITEM(LOOP_1).GT.MAXELM) THEN
			ITEM(LOOP_1) = ITEM(LOOP_1) + DELTA
		ENDIF
	  ENDDO
	ENDIF

C	discard old tables and re-associate

	DEALLOCATE( KELEM, ELEM_LOCKED, IECNT,
     &			IEDAT, IETYP, IELIN, KETYP, KELABL )
	KELEM => KELEM_A
	KETYP => KETYP_A
	KELABL => KELABL_A
	ELEM_LOCKED => ELEM_LOCKED_A
	IECNT => IECNT_A
	IEDAT => IEDAT_A
	IETYP => IETYP_A
	IELIN => IELIN_A
	NULLIFY( KELEM_A, KETYP_A, KELABL_A, ELEM_LOCKED_A,
     &		 IECNT_A, IEDAT_A, IETYP_A, IELIN_A )
	MAXELM = MAXELM + DELTA
	IELEM2 = IELEM2 + DELTA

	MORE_ELTS = .TRUE.

 9999 RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO INCREASE ELEMENT TABLE SIZE ',
     &	   'FROM ',I10,' TO ',I10,' '/' ')
	END
