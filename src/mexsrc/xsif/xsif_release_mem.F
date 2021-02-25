	SUBROUTINE XSIF_RELEASE_MEM
C
C	deallocates element, parameter, link, position,
C	wakefield tables.
C
C	AUTH: PT, 25-MAY-2003
C
C	MOD:
C          27-feb-2004, PT:
C             bugfix:  when deallocating arrays related to the
C             expanded beamline, get ERRPTR and ERRFLG as well.

	USE XSIF_SIZE_PARS
	USE XSIF_ELEMENTS

	IMPLICIT NONE
	SAVE

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                 C  O  D  E                                           C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	IF (ASSOCIATED(KELEM)) THEN

	  DEALLOCATE(KELEM, IELIN, IETYP, IEDAT, KETYP, KELABL, IECNT,
     &			 ELEM_LOCKED)
	  NULLIFY(KELEM, IELIN, IETYP, IEDAT, KETYP, KELABL, IECNT,
     &			 ELEM_LOCKED)
	  IELEM1 = 0
	  IELEM2 = 0
	  MAXELM = 0

	ENDIF

	IF (ASSOCIATED(KPARM)) THEN

	  DEALLOCATE( KPARM, PDATA, IPLIN, IPDAT, IPTYP, IPNEXT )
	  NULLIFY( KPARM, PDATA, IPLIN, IPDAT, IPTYP, IPNEXT )
	  IPARM1 = 0
	  IPARM2 = 0
	  MAXPAR = 0

	ENDIF

	IF (ASSOCIATED(ILDAT)) THEN

	  DEALLOCATE( ILDAT )
	  NULLIFY( ILDAT )
	  IUSED = 0
	  MAXLST = 0

	ENDIF

	IF (ASSOCIATED(LWAKE_FILE)) THEN

	  DEALLOCATE(LWAKE_FILE,TWAKE_FILE)
	  NULLIFY(LWAKE_FILE,TWAKE_FILE)
	  MX_WAKEFILE = 0
	  NUM_LWAKE = 0
	  NUM_TWAKE = 0

	ENDIF

	IF (ALLOCATED(ITEM)) THEN

	  DEALLOCATE(ITEM,PRTFLG,ERRPTR,ERRFLG)
	  NPOS1=0
	  NPOS2=0
	  MAXPOS = 0

	ENDIF

	RETURN
	END