	LOGICAL*4 FUNCTION XSIF_ALLOCATE_INITIAL( )
C
C	perform initial allocation of element, parameter, list, link
C	tables, wakefiles, I/O call array.  Returns TRUE if
C	successful, FALSE if not.
C
C	Auth: PT, 19-may-2003
C
C	MOD:
C          15-JAN-2004, PT:
C             when ITEM is deallocated, wipe out ERRFLG and
C             ERRPTR as well.
C
	USE XSIF_SIZE_PARS
	USE XSIF_INOUT
	USE XSIF_ELEMENTS
C
	IMPLICIT NONE
	SAVE

C	local declarations

	INTEGER*4 ALLSTAT, COUNT

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C	start with the element table

	IF (ASSOCIATED(KELEM))
     &  DEALLOCATE( KELEM, KETYP, KELABL, IECNT, IETYP, IEDAT, IELIN,
     &  			  ELEM_LOCKED )

	MAXELM = MAXELM_INIT

	ALLOCATE( KELEM(MAXELM), KETYP(MAXELM), KELABL(MAXELM),
     &		  IECNT(MAXELM), IETYP(MAXELM), IEDAT(MAXELM,3),
     &		  IELIN(MAXELM), ELEM_LOCKED(MAXELM), STAT=ALLSTAT)
	IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,910)MAXELM
		WRITE(ISCRN,910)MAXELM
		ERROR = .TRUE.
		GOTO 9999
	ENDIF

	IELEM1 = 0
	IELEM2 = MAXELM + 1

	KELEM = ' '
	KETYP = ' '
	KELABL = ' '

	IECNT = 0
	IETYP = 0
	IEDAT = 0
	IELIN = 0
	ELEM_LOCKED = .FALSE.

C	now the parameter table

	IF(ASSOCIATED(KPARM))
     &  DEALLOCATE( KPARM, IPTYP, IPDAT, IPLIN, IPNEXT, PDATA ) 

	MAXPAR = MAXPAR_INIT

	ALLOCATE( KPARM(MAXPAR), IPTYP(MAXPAR), IPDAT(MAXPAR,3),
     &		  IPLIN(MAXPAR), IPNEXT(MAXPAR), PDATA(MAXPAR),
     &		  STAT = ALLSTAT )
	IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,920)MAXPAR
		WRITE(ISCRN,920)MAXPAR
		ERROR=.TRUE.
		GOTO 9999
	ENDIF

	IPARM1 = 0
	IPARM2 = MAXPAR+1
	IPLIST = 0

	KPARM = ' '
	IPTYP = 0
	IPDAT = 0
	IPLIN = 0
	IPNEXT = 0
	PDATA = 0.

C	now the link table
	IF (ASSOCIATED(ILDAT))
     &  DEALLOCATE( ILDAT )

	MAXLST = MAXLST_INIT
	ALLOCATE( ILDAT(MAXLST,6), STAT = ALLSTAT )
	IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,930)MAXLST
		WRITE(ISCRN,930)MAXLST
		ERROR = .TRUE.
	ENDIF

	IUSED = 0
	ILDAT = 0

C	now the position table -- this is actually allocated by
C	EXPAND, so just wipe it out here.

	IF (ALLOCATED(ITEM))
     &  DEALLOCATE( ITEM, PRTFLG , ERRFLG, ERRPTR )
	NPOS1 = 0
	NPOS2 = 0

C	now for the wakefield file name tables

	IF (ASSOCIATED(LWAKE_FILE))
     &  DEALLOCATE( LWAKE_FILE, TWAKE_FILE )

	MX_WAKEFILE = MX_WAKE_INIT
	ALLOCATE( LWAKE_FILE(MX_WAKEFILE), TWAKE_FILE(MX_WAKEFILE),
     &		  STAT = ALLSTAT )
	IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,940)MX_WAKEFILE
		WRITE(ISCRN,940)MX_WAKEFILE
		ERROR = .TRUE.
		GOTO 9999
	ENDIF
	DO COUNT = 1,MX_WAKEFILE
	  NULLIFY(LWAKE_FILE(COUNT)%FNAM_PTR)
	  NULLIFY(TWAKE_FILE(COUNT)%FNAM_PTR)
	ENDDO
	NUM_LWAKE = 0
	NUM_TWAKE = 0

C	Now for the I/O stack:  this one is a bit different, 
C	in that we don't want to blow away the call stack if it's
C	got stuff in it...

	IF ( (.NOT.ASSOCIATED(IO_UNIT)) .OR. (NUM_CALL.EQ.0) ) THEN
	  IF (ASSOCIATED(IO_UNIT))
     &    DEALLOCATE(IO_UNIT)
	  MXCALL = MXCALL_INIT
	  ALLOCATE(IO_UNIT(MXCALL),STAT=ALLSTAT)
	  IF (ALLSTAT.NE.0) THEN
		WRITE(IECHO,950)MXCALL
		WRITE(ISCRN,950)MXCALL
		ERROR=.TRUE.
		GOTO 9999
	  ENDIF
	  IO_UNIT = 0
	  NUM_CALL = 0
	ENDIF

C	all done; set the return status and return

 9999 IF (ERROR) THEN
		XSIF_ALLOCATE_INITIAL = .FALSE.
	ELSE
		XSIF_ALLOCATE_INITIAL = .TRUE.
	ENDIF

	RETURN
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE ',I10,' ENTRIES IN ',
     &	   'ELEMENT TABLE'/' ')
  920 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE ',I10,' ENTRIES IN ',
     &	   'PARAMETER TABLE'/' ')
  930 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE ',I10,' ENTRIES IN ',
     &	   'LINK TABLE'/' ')
  940 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE ',I10,' ENTRIES IN ',
     &	   'WAKE FILENAME TABLES'/' ')
  950 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE ',I10,' ENTRIES IN ',
     &	   'CALL STACK TABLE'/' ')
	END
