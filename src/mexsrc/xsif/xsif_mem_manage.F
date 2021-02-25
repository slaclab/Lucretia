	LOGICAL*4 FUNCTION XSIF_MEM_MANAGE( )
C
C	decides whether any of XSIF data tables need to be expanded,
C	and if so calls the appropriate expander function
C
C	AUTH:  PT, 19-MAY-2003
C
C	MOD:
C
	USE XSIF_SIZE_PARS
	USE XSIF_ELEMENTS
	USE XSIF_INOUT

	IMPLICIT NONE
	SAVE

C	local declarations

	LOGICAL*4 MEMSTAT

C	referenced functions

	LOGICAL*4 MORE_ELTS, MORE_PARS, MORE_LINKS
	LOGICAL*4 MORE_WAKES, MORE_CALLS

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

	MEMSTAT = .TRUE.

C	first the elements

	IF (IELEM2 - IELEM1 .LT. ELM_TOL) THEN
		MEMSTAT = MORE_ELTS( MAXELM_STEP )
	ENDIF

C	then the parameters

	IF (IPARM2 - IPARM1 .LT. PAR_TOL) THEN
		MEMSTAT = (MEMSTAT.AND.MORE_PARS(MAXPAR_STEP))
	ENDIF

C	now the link tables

	IF (MAXLST - IUSED .LT. LST_TOL) THEN
		MEMSTAT = (MEMSTAT.AND.MORE_LINKS(MAXLST_STEP))
	ENDIF

C	now the wakefield filenames

	IF ( (NUM_LWAKE.EQ.MX_WAKEFILE).OR.(NUM_TWAKE.EQ.MX_WAKEFILE) ) 
     &THEN
		MEMSTAT = (MEMSTAT.AND.MORE_WAKES(MAX_WAKE_STEP))
	ENDIF

C	finally the call stack

	IF (NUM_CALL .EQ. MXCALL) THEN
		MEMSTAT = (MEMSTAT.AND.MORE_CALLS(MXCALL_STEP))
	ENDIF

	XSIF_MEM_MANAGE = MEMSTAT

	RETURN
	END
