#include "fintrf.h"

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
C
C     Matlab wrapper to the xsif parser.  Returns the following:
C
C     plhs(1) = xsif status value, integer
C
C     plhs(2) = element data structure.  IELEM1 x 1 data structure
C               with the fields
C                 NAME  (char string, translated from KELEM)
C                 CLASS (char string, translated from KETYP)
C                 LABEL (char string, translated from KELABL)
C                 TYPE  (integer, translated from IETYP)
C                 DATA  (2x1 integer array, translated from IEDAT)
C
C     plhs(3) = parameter data structure.  (MAXPAR-IPARM2) x 1 data structure
C               with the fields
C                 NAME  (char string, translated from KPARM)
C                 VALUE (real8, translated from PDATA)
C
C     plhs(4) = line data structure, MAXPOS x 1 data structure with the
C               fields
C                 ELEMENT (integer, translated from ITEM)
C                 EALIGN  (integer, translated from ERRPTR)
C
C     plhs(5) = beam/beta0 pointers.  1x1 data structure with fields:
C                 BEAMPOINTER : index into plhs(2) of beam element.
C                 BETAPOINTER : index into plhs(2) of beta0 element.
C
C     plhs(6) = wakefield data structure.  1x1 data structure with the
C               fields:
C                 LONGIT : cell array of longitudinal wake file names
C                 TRANSV : cell array of transverse wake file names
C
C               note that plhs(5) and (6) are optional, the rest are mandatory.
C
C     AUTH:  PT, 25-FEB-2004
C
C     MOD:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
	USE XSIF_ELEMENTS
	USE XSIF_INTERFACES

      IMPLICIT NONE

C     argument declarations

      INTEGER*4 NLHS, NRHS       ! number of args returned and called with
	mwPointer PLHS(*), PRHS(*) ! pointers to returned/called args

C     local declarations

      INTEGER*4, PARAMETER :: ALTSCREENUNIT = 88
      INTEGER*4, PARAMETER :: MAXFILENAMELEN = 256
      CHARACTER(MAXFILENAMELEN) XSIFFILENAME
      CHARACTER(ENAME_LENGTH) LINENAME
	INTEGER*4 M,N,MXN, ISTAT
        mwPointer IDMY
	LOGICAL*4 LINEINARGLIST, ECHO, NLCWARN
	REAL*8 REALLOGICAL
	LOGICAL*4 IOSETUPFAIL, CMDLOOPFAIL, USEFAIL, PARAMERROR
	mwPointer ELEMENTS, PARAMS, USEDLINE, WAKES, BEAMBETA0
	INTEGER*4 LOOP_COUNT, EPARS(2), I1, I2

	CHARACTER*5 ELEMFIELDS(5)
	DATA ELEMFIELDS / 'name ','class','label','type ','data ' / 

	CHARACTER*5 PARFIELDS(2)
	DATA PARFIELDS / 'name ','value' /

	CHARACTER*7 LINEFIELDS(2)
	DATA LINEFIELDS / 'element','ealign ' /

      CHARACTER*6 WAKEFIELDS(2)
	DATA WAKEFIELDS / 'longit','transv' /

	CHARACTER*11 BEAMBETA0FIELDS(2)
	DATA BEAMBETA0FIELDS / 'beampointer','betapointer' /

        mwPointer pMATLAB   ! general purpose matlab pointer
	mwPointer pWAKEFILE ! matlab pointer to wakefile name

C     referenced functions

      INTEGER*4 XSIF_IO_SETUP  ! opens files, etc.
	INTEGER*4 PARCHK         ! finds undefined parameters
	INTEGER*4 XUSE2          ! perform USE, beamline operation

	mwPointer mxCreateStructMatrix ! create structure matrix
	mwPointer mxCreateDoubleMatrix ! create matlab matrix of real8's
	mwPointer mxCreateString       ! create a string
	mwPointer mxCreateCellMatrix   ! create a cell matrix
      INTEGER*4 mxIsChar             ! is it a character string?
	INTEGER*4 mxGetM, mxGetN       ! matlab array dimensions
	INTEGER*4 mxGetString          ! string copy routine
	INTEGER*4 mxIsLogical          ! check to see if mxArray is logical
	INTEGER*4 mexEvalString        ! execute command in Matlab dataspace
	mwPointer mxGetPr              ! get a pointer...


C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                         C  O  D  E                                   C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     begin by checking the number of calling and returning arguments
C     against the requirements of XSIFParse:
C
      IF ( (NLHS .LT. 4) .OR. (NLHS .GT. 6) ) THEN
          CALL mexErrMsgTxt(
     &' XSIFParse requires between 4 and 6 output arguments')
      ENDIF
      
      IF ( (NRHS .LT. 1) .OR. (NRHS .GT. 4) ) THEN
        CALL mexErrMsgTxt(
     &' XSIFParse requires between 1 and 4 input arguments')
      ENDIF
C
C     further inspect the input arguments and make sure they are
C     okay
C
      IF (mxIsChar(PRHS(1)) .NE. 1) THEN
        CALL mexErrMsgTxt(
     &' First XSIFParse input argument must be a character array')
      ENDIF
	M = mxGetM(PRHS(1))
	N = mxGetN(PRHS(1))
	MXN = M * N
	IF (MXN .GT. MAXFILENAMELEN) THEN
	   call mexErrMsgTxt(
     &' Filename argument exceeds max length')
	ENDIF
	IDMY = mxGetString(PRHS(1),XSIFFILENAME,MXN)

      IF ( NRHS .GE. 2 ) THEN

	    IF (mxIsChar(PRHS(2)) .NE. 1) THEN

	      CALL mexErrMsgTxt(
     &' Second XSIFParse input argument must be a character array')

	    ENDIF
	    M = mxGetM(PRHS(2))
	    N = mxGetN(PRHS(2))
	    MXN = M * N
	    IF (MXN .GT. ENAME_LENGTH) THEN
	        CALL mexErrMsgTxt(
     &' XSIFParse beamline name cannot exceed XSIF name length limit')
	    ENDIF
	    IDMY = mxGetString(PRHS(2),LINENAME,MXN)
          LINEINARGLIST = .TRUE.

	ELSE

	    LINEINARGLIST = .FALSE.

	ENDIF

	IF ( NRHS .GE. 3 ) THEN

	    IF (mxIsLogical(PRHS(3)) .NE. 1) THEN
	      CALL mexErrMsgTxt(
     &' Third XSIFParse input argument must be boolean')
	    ENDIF
          CALL mxCopyPtrToReal8(PRHS(3),REALLOGICAL,1)
          IF (REALLOGICAL .NE. 0.) THEN
            ECHO = .TRUE.
	    ELSE
	      ECHO = .FALSE.
	    ENDIF

	ELSE

	    ECHO = .FALSE.

	ENDIF

		IF ( NRHS .GE. 4 ) THEN

	    IF (mxIsLogical(PRHS(4)) .NE. 1) THEN
	      CALL mexErrMsgTxt(
     &' Fourth XSIFParse input argument must be boolean')
	    ENDIF
          CALL mxCopyPtrToReal8(PRHS(4),REALLOGICAL,1)
          IF (REALLOGICAL .NE. 0.) THEN
            NLCWARN = .TRUE.
	    ELSE
	      NLCWARN = .FALSE.
	    ENDIF

	ELSE

	    NLCWARN = .FALSE.

	ENDIF

C========1=========2=========3=========4=========5=========6=========7=C
C
C     So 83 lines later we're done quality-checking the input arguments.
C     Now initialize the xsif i/o (open files, etc).
C     xsif requires 3 files:  the deckfile, stream file, and error
C     file.  It also sends text to the terminal window.  For these
C     applications we will set the 3 files to logical units 15, 16, and
C     17, and reroute screen messages to the ALTSCREENUNIT.
C
C========1=========2=========3=========4=========5=========6=========7=C

	IOSETUPFAIL = .FALSE. 
      CMDLOOPFAIL = .FALSE.
      USEFAIL = .FALSE.
	PARAMERROR = .FALSE.

      ISCRN = ALTSCREENUNIT
	OPEN (UNIT=ISCRN, FILE = "xsifscreen.txt", STATUS = "REPLACE", 
     &      ACTION = "WRITE", IOSTAT = ISTAT)
	IF (ISTAT .NE. 0) THEN
	
	    CALL mexErrMsgTxt(
     &' Unable to open file for screen text in XSIFParse')
	ENDIF

C========1=========2=========3=========4=========5=========6=========7=C
C
C     IMPORTANT NOTE:  the instance of mexErrMsgTxt above is the last
C                      one in XSIFParse,f!  Henceforth, any failures
C                      internal to XSIF do not lead to a Matlab abort,
C                      instead they result in a jump to the egress.
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C     create dummy structures for "return" in case things blow up
C     along the way here...
C
      ELEMENTS = mxCreateDoubleMatrix(1,1,0) 
	PARAMS = mxCreateDoubleMatrix(1,1,0) 
      USEDLINE = mxCreateDoubleMatrix(1,1,0) 
	BEAMBETA0 = mxCreateDoubleMatrix(1,1,0) 
	WAKES = mxCreateDoubleMatrix(1,1,0) 

      ISTAT = XSIF_IO_SETUP( XSIFFILENAME, 
     &                       'xsiferror.txt',
     &                       'xsifstream.txt', 15, 16, 17,
     &                       ISCRN, ECHO, NLCWARN           )
C
C     if the IO setup failed, assign the return code to plhs(1) 
C     and go to the egress 
C
      IF (ISTAT .NE. 0) THEN
          IOSETUPFAIL = .TRUE.
	    GOTO 9999
	ENDIF
C
C     if on the other hand the I/O setup failed, then we can
C     execute the xsif master command loop
C
      ISTAT = XSIF_CMD_LOOP( )
C
C     once again, if the command loop failed, do appropriate
C     failure-related things
C
      IF (ISTAT.NE.0) THEN
          CMDLOOPFAIL = .TRUE.
	    GOTO 9999
	ENDIF
C
C     If the user specified a line to expand, do that now
C
      IF (LINEINARGLIST) THEN
          ISTAT = XUSE2( LINENAME )
          IF (ISTAT.NE.0) THEN
              USEFAIL = .TRUE.
	        GOTO 9999
	    ENDIF
	ENDIF
C
C     alternately, if there was no beamline passed as an
C     argument, but also no USE command in the input files,
C     throw an error
C
      IF (.NOT. LINE_EXPANDED) THEN
          ISTAT = XSIF_PARSE_NOLINE
	    USEFAIL = .TRUE.
	    GOTO 9999
	ENDIF
C
C     finally, perform parameter checkout and evaluation.  Note
C     that if PARCHK returns nonzero status (undefined parameters
C     detected), we are treating it as a WARNING and not an error.
C
      CALL PARORD( PARAMERROR )
	IF (PARAMERROR) THEN
	    ISTAT = XSIF_PARSE_ERROR
	    GOTO 9999
	ENDIF
	CALL PAREVL
	ISTAT = PARCHK( .FALSE. )
C
C     create data structures for return to Matlab at the end of execution
C
      CALL mxDestroyArray( ELEMENTS )
      CALL mxDestroyArray( PARAMS )
      CALL mxDestroyArray( USEDLINE )
      CALL mxDestroyArray( BEAMBETA0 )
      CALL mxDestroyArray( WAKES )

      I1=IELEM1;
      I1=MAXPAR;
      I2=IPARM2;
      WRITE(*,*),MAXPAR,IPARM2, MAXPOS
      ELEMENTS = mxCreateStructMatrix( IELEM1, 1, 5, ELEMFIELDS )
      PARAMS=mxCreateStructMatrix( MAXPAR-IPARM2+1, 1, 2, PARFIELDS )
      USEDLINE = mxCreateStructMatrix( MAXPOS, 1, 2, LINEFIELDS )
	BEAMBETA0 = mxCreateStructMatrix( 1, 1, 2, 
     &                BEAMBETA0FIELDS )
      WAKES = mxCreateStructMatrix( 1, 1, 2, WAKEFIELDS )
      
C========1=========2=========3=========4=========5=========6=========7=C
C
C     If we've gotten this far, then the xsif parser thinks that it
C     has read some deckfiles and expanded a line successfully.  We
C     can now allocate some data structures for return to matlab, 
C     and fill them.
C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     start by creating and filling the ELEMENTS structure.  Since
C     we will ignore the parameters which are prior to IPARM2+1, 
C     we need to change the value of IEDAT to point at the correct
C     locations.
C
      DO LOOP_COUNT = 1,IELEM1

	    pMATLAB = mxCreateString( KELEM(LOOP_COUNT) )
          CALL mxSetField( ELEMENTS, LOOP_COUNT, ELEMFIELDS(1),
     &                     pMATLAB )
	    pMATLAB = mxCreateString( KETYP(LOOP_COUNT) )
          CALL mxSetField( ELEMENTS, LOOP_COUNT, ELEMFIELDS(2),
     &                     pMATLAB )
	    pMATLAB = mxCreateString( KELABL(LOOP_COUNT) )
          CALL mxSetField( ELEMENTS, LOOP_COUNT, ELEMFIELDS(3),
     &                     pMATLAB )
	    pMATLAB = mxCreateDoubleMatrix(1, 1, 0)
          IDMY = mxGetPr( pMATLAB )
	    CALL mxCopyReal8ToPtr( dble(IETYP(LOOP_COUNT)), IDMY, 1 )
	    CALL mxSetField( ELEMENTS, LOOP_COUNT, ELEMFIELDS(4),
     &                     pMATLAB )
          pMATLAB = mxCreateDoubleMatrix( 2, 1, 0 )
	    IDMY = mxGetPr( pMATLAB )
	    EPARS(1) = IEDAT(LOOP_COUNT,1) - IPARM2 + 1
	    EPARS(2) = IEDAT(LOOP_COUNT,2) - IPARM2 + 1
	    CALL mxCopyReal8ToPtr( DBLE(EPARS), IDMY, 2 )
	    CALL mxSetField( ELEMENTS, LOOP_COUNT, ELEMFIELDS(5),
     &                     pMATLAB )

      ENDDO
C
C     now capture the parameters at the end of the list (ie, the ones
C     from IPARM2+1 to MAXPAR).
C
      DO LOOP_COUNT = IPARM2,MAXPAR-1
	    pMATLAB = mxCreateString( KPARM(LOOP_COUNT) )
	    CALL mxSetField( PARAMS, LOOP_COUNT-IPARM2+1, PARFIELDS(1),
     &                     pMATLAB )
          pMATLAB = mxCreateDoubleMatrix( 1, 1, 0 )
	    IDMY = mxGetPr( pMATLAB )
          CALL mxCopyReal8ToPtr( PDATA(LOOP_COUNT), IDMY, 1 )
	    CALL mxSetField( PARAMS, LOOP_COUNT-IPARM2+1, PARFIELDS(2),
     &                     pMATLAB )

	ENDDO
C
C     now capture the expanded beamline
C
	DO LOOP_COUNT = 1,MAXPOS

	    pMATLAB = mxCreateDoubleMatrix(1,1,0)
	    IDMY = mxGetPr( pMATLAB )
	    IF (ITEM(LOOP_COUNT).LT.0) ITEM(LOOP_COUNT) = 0
	    IF (ITEM(LOOP_COUNT).GT.IELEM1) ITEM(LOOP_COUNT) = 0
	    CALL mxCopyReal8ToPtr( DBLE(ITEM(LOOP_COUNT)), IDMY, 1 )
	    CALL mxSetField( USEDLINE, LOOP_COUNT, LINEFIELDS(1),
     &                     pMATLAB )
	    pMATLAB = mxCreateDoubleMatrix(1,1,0)
	    IDMY = mxGetPr( pMATLAB )
	    CALL mxCopyReal8ToPtr( DBLE(ERRPTR(LOOP_COUNT)), IDMY, 1 )
	    CALL mxSetField( USEDLINE, LOOP_COUNT, LINEFIELDS(2),
     &                     pMATLAB )

      ENDDO
C
C     if the beam/beta0 pointers are desired, captue them now
C
      IF ( NLHS .GE. 5 ) THEN

	    pMATLAB = mxCreateDoubleMatrix( 1, 1, 0 )
	    IDMY = mxGetPr( pMATLAB )
	    CALL mxCopyReal8ToPtr( DBLE(IBETA0_PTR), IDMY, 1 )
	    CALL mxSetField( BEAMBETA0, 1, BEAMBETA0FIELDS(2),
     &                     pMATLAB )
	    pMATLAB = mxCreateDoubleMatrix( 1, 1, 0 )
	    IDMY = mxGetPr( pMATLAB )
	    CALL mxCopyReal8ToPtr( DBLE(IBEAM_PTR), IDMY, 1 )
	    CALL mxSetField( BEAMBETA0, 1, BEAMBETA0FIELDS(1),
     &                     pMATLAB )

	ENDIF
C
C     if the wakefield file names are desired, capture them now
C
      IF ( NLHS .GE. 6 ) THEN
C
C     allocate a cell array for longitudinal wakefields
C
          pMATLAB = mxCreateCellMatrix(NUM_LWAKE,1)
          DO LOOP_COUNT = 1,NUM_LWAKE

      pWAKEFILE=
     & mxCreateString(ARR_TO_STR(LWAKE_FILE(LOOP_COUNT)%FNAM_PTR))
	        CALL mxSetCell( pMATLAB, LOOP_COUNT, pWAKEFILE )
	    
	    ENDDO

          CALL mxSetField( WAKES, 1, WAKEFIELDS(1), pMATLAB )

          pMATLAB = mxCreateCellMatrix(NUM_TWAKE,1)
          DO LOOP_COUNT = 1,NUM_TWAKE

	        pWAKEFILE = mxCreateString(
     &           ARR_TO_STR(TWAKE_FILE(LOOP_COUNT)%FNAM_PTR) )
	        CALL mxSetCell( pMATLAB, LOOP_COUNT, pWAKEFILE )
	    
	    ENDDO

          CALL mxSetField( WAKES, 1, WAKEFIELDS(2), pMATLAB )

      ENDIF

C========1=========2=========3=========4=========5=========6=========7=C
C
C     Next stop is the egress!
C
C========1=========2=========3=========4=========5=========6=========7=C

 9999 CONTINUE
C
C     set the status return to the correct value
C
      pMATLAB = mxCreateDoubleMatrix(1,1,0)
	IDMY = mxGetPr(pMATLAB) 
      CALL mxCopyReal8ToPtr( DBLE(ISTAT), IDMY, 1 )
	PLHS(1) = pMATLAB
	PLHS(2) = ELEMENTS
	PLHS(3) = PARAMS
	PLHS(4) = USEDLINE
	IF (NLHS.GE.5) PLHS(5) = BEAMBETA0
      IF (NLHS.GE.6) PLHS(6) = WAKES

C
C     close the screen output file
C
      CLOSE( ISCRN )
C
C     execute the Matlab script that reads the screen output file to the
C     matlab window (giving credit where it's due, this technology was
C     developed by L. Hendrickson for mat-liar)
C
      IDMY = mexEvalString('type xsifscreen.txt')
C
C     Depending on whether there was some failure during the parsing
C     or not, we may want to display a message to the user.  Handle
C     that now.
C
      IF (IOSETUPFAIL) THEN
	    CALL mexWarnMsgTxt(' XSIFParse failure in IO Setup stage')
	ENDIF
	IF (CMDLOOPFAIL) THEN
	    CALL mexWarnMsgTxt(' XSIFParse failure in main loop stage')
	ENDIF
	IF (USEFAIL) THEN
	    IF (LINEINARGLIST) THEN
	        CALL mexWarnMsgTxt(
     &        ' XSIFParse unable to expand line named in input args')
	    ELSE
	        CALL mexWarnMsgTxt(
     &        ' No "USE" statement found in input files')
	    ENDIF
      ENDIF
C
C     close any files opened by XSIF
C
      CALL XSIF_IO_CLOSE
C
C     deallocate XSIF's memory
C
      CALL XSIF_RELEASE_MEM
C
C     and return
C
      RETURN
	END
