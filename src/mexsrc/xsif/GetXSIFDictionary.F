#include "fintrf.h"

      SUBROUTINE mexFunction(nlhs, plhs, nrhs, prhs)
C
C     This is a Matlab interface routine which allows Matlab to
C     extract the present XSIF dictionary tables from XSIF_ELEM_PARS
C     to a Matlab data structure.  
C
C     AUTH: PT, 25-feb-2004
C
C     MOD:
C
      USE XSIF_ELEM_PARS

      IMPLICIT NONE

C     argument declarations

      INTEGER*4 NRHS, NLHS       ! nummber of passed and returned arguments, resp.
      mwpointer PRHS(*), PLHS(*)          ! C pointers to the passed and returned args

C     local declarations

      INTEGER*4 LOOP_COUNT             ! self-explanatory

	mwpointer pELEMDICT, pEALIGNDICT ! Pointers to the element and ealign
	                                 ! dictionaries
	mwpointer pSTRING, pCELL         ! misc pointers

	CHARACTER*9 ELEMFIELD(2)        ! names of the element structure fields
      DATA ELEMFIELD / 'keyword  ','parameter' /
      CHARACTER*9 EALIGNFIELD         ! name of the EALIGN structure field
	DATA EALIGNFIELD / 'parameter' /

C     referenced functions

      INTEGER*4 mxCreateStructMatrix ! Create a Matlab structure
      INTEGER*4 mxCreateString       ! create a string  
      
C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C     start by checking that there are 2 returned arguments, otherwise
C     fail out

      IF (NLHS .NE. 2) THEN
      
          CALL mexErrMsgTxt( 
     &'ERR> GetXSIFDictionary requires 2 output arguments.')

	ENDIF

C
C     build the data structure for the element dictionary
C
      pELEMDICT = mxCreateStructMatrix(NKEYW,1,2,ELEMFIELD)
	IF (pELEMDICT .EQ. 0) THEN

	    CALL mexErrMsgTxt(
     &'ERROR> Unable to generate element dictionary structure')

	ENDIF
C
C     build the data structure for the ealign dictionary
C
      pEALIGNDICT = mxCreateStructMatrix(1,1,1,EALIGNFIELD)
	IF (pEALIGNDICT .EQ. 0) THEN

	    CALL mexErrMsgTxt(
     &'ERROR> Unable to generate EALIGN dictionary structure')

	ENDIF
C
C     loop over element keywords and add them to the keyword fields
C
      DO LOOP_COUNT = 1,NKEYW

C
C     allocate a string and fill it
C
          pSTRING = mxCreateString( DKEYW(LOOP_COUNT) )

	    IF (pSTRING .EQ. 0) THEN

	        CALL mexErrMsgTxt(
     &'ERROR> Unable to create keyword string')

	    ENDIF

          CALL mxSetField( pELEMDICT, LOOP_COUNT, ELEMFIELD(1), 
     &                     pSTRING                              )

	ENDDO

C========1=========2=========3=========4=========5=========6=========7=C
C
C     Generating the cell arrays of element parameter names:
C     Somewhat cumbersome because each element type has its own
C     dictionary string array.  To ease maintenance, a utility
C     routine is used to fill the appropriate slots in the data
C     structure.
C
C========1=========2=========3=========4=========5=========6=========7=C

C     DRIFTs

      CALL PutElemParsToMatlab( MAD_DRIFT, NDRFT, DDRFT, pELEMDICT,
     &                          ELEMFIELD(2) )

C     RBENDs

      CALL PutElemParsToMatlab( MAD_RBEND, NBEND, DBEND, pELEMDICT,
     &                          ELEMFIELD(2) )

C     SBENDs

      CALL PutElemParsToMatlab( MAD_SBEND, NBEND, DBEND, pELEMDICT,
     &                          ELEMFIELD(2) )

c     WIGGs:  no parameters in the present implementation!

C     QUADs

      CALL PutElemParsToMatlab( MAD_QUAD, NQUAD, DQUAD, pELEMDICT,
     &                          ELEMFIELD(2) )

C     SEXTs

      CALL PutElemParsToMatlab( MAD_SEXT, NSEXT, DSEXT, pELEMDICT,
     &                          ELEMFIELD(2) )

C     OCTUs

      CALL PutElemParsToMatlab( MAD_OCTU, NOCT, DOCT, pELEMDICT,
     &                          ELEMFIELD(2) )

C     MULTs

      CALL PutElemParsToMatlab( MAD_MULTI, NMULT, DMULT, pELEMDICT,
     &                          ELEMFIELD(2) )

C     SOLOs

      CALL PutElemParsToMatlab( MAD_SOLN, NSOLO, DSOLO, pELEMDICT,
     &                          ELEMFIELD(2) )

C     RFCAVs

      CALL PutElemParsToMatlab( MAD_RFCAV, NCVTY, DCVTY, pELEMDICT,
     &                          ELEMFIELD(2) )

C     SEPAs

      CALL PutElemParsToMatlab( MAD_SEPA, NSEPA, DSEPA, pELEMDICT,
     &                          ELEMFIELD(2) )

C     ROLLs

      CALL PutElemParsToMatlab( MAD_ROLL, NROTA, DROTA, pELEMDICT,
     &                          ELEMFIELD(2) )

C     ZROTs

      CALL PutElemParsToMatlab( MAD_ZROT, NROTA, DROTA, pELEMDICT,
     &                          ELEMFIELD(2) )

C     HKICs

      CALL PutElemParsToMatlab( MAD_HKICK, NKICK, DKICK, pELEMDICT,
     &                          ELEMFIELD(2) )

C     VKICs

      CALL PutElemParsToMatlab( MAD_VKICK, NKICK, DKICK, pELEMDICT,
     &                          ELEMFIELD(2) )

C     HMONs

      CALL PutElemParsToMatlab( MAD_HMON, NMON, DMON, pELEMDICT,
     &                          ELEMFIELD(2) )

C     VMONs

      CALL PutElemParsToMatlab( MAD_VMON, NMON, DMON, pELEMDICT,
     &                          ELEMFIELD(2) )

C     MONIs

      CALL PutElemParsToMatlab( MAD_MONI, NMON, DMON, pELEMDICT,
     &                          ELEMFIELD(2) )

C     MARKs have no parameters in the present implementation

C     ECOLLs

      CALL PutElemParsToMatlab( MAD_ECOLL, NCOLL, DCOLL, pELEMDICT,
     &                          ELEMFIELD(2) )

C     RCOLLs

      CALL PutElemParsToMatlab( MAD_RCOLL, NCOLL, DCOLL, pELEMDICT,
     &                          ELEMFIELD(2) )

C     QUADSEXTs

      CALL PutElemParsToMatlab( MAD_QUSE, NQUSE, DQUSE, pELEMDICT,
     &                          ELEMFIELD(2) )
      
C     GKICKs

      CALL PutElemParsToMatlab( MAD_GKICK, NGKIK, DGKIK, pELEMDICT,
     &                          ELEMFIELD(2) )

C     ARBITELMs

      CALL PutElemParsToMatlab( MAD_ARBIT, NARBI, DARBI, pELEMDICT,
     &                          ELEMFIELD(2) )

C     MTWISSs

      CALL PutElemParsToMatlab( MAD_MTWIS, NTWIS, DTWIS, pELEMDICT,
     &                          ELEMFIELD(2) )

C     MATRIXs

      CALL PutElemParsToMatlab( MAD_MATR, NMATR, DMATR, pELEMDICT,
     &                          ELEMFIELD(2) )

C     LCAVs

      CALL PutElemParsToMatlab( MAD_LCAV, NLCAV, DLCAV, pELEMDICT,
     &                          ELEMFIELD(2) )

C     INSTs

      CALL PutElemParsToMatlab( MAD_INST, NINST, DINST, pELEMDICT ,
     &                          ELEMFIELD(2))

C     BLMOs

      CALL PutElemParsToMatlab( MAD_BLMO, NINST, DINST, pELEMDICT,
     &                          ELEMFIELD(2) )

C     PROFs

      CALL PutElemParsToMatlab( MAD_PROF, NINST, DINST, pELEMDICT,
     &                          ELEMFIELD(2) )

C     WIREs

      CALL PutElemParsToMatlab( MAD_WIRE, NINST, DINST, pELEMDICT,
     &                          ELEMFIELD(2) )

C     SLMOs

      CALL PutElemParsToMatlab( MAD_SLMO, NINST, DINST, pELEMDICT,
     &                          ELEMFIELD(2) )

C     IMONs

      CALL PutElemParsToMatlab( MAD_IMON, NINST, DINST, pELEMDICT,
     &                          ELEMFIELD(2) )

C     DIMUs

      CALL PutElemParsToMatlab( MAD_DIMU, NMULT, DDIMU, pELEMDICT,
     &                          ELEMFIELD(2) )

C     YROTs

      CALL PutElemParsToMatlab( MAD_YROT, NROTA, DROTA, pELEMDICT,
     &                          ELEMFIELD(2) )

C     SROTs

      CALL PutElemParsToMatlab( MAD_SROT, NROTA, DROTA, pELEMDICT,
     &                          ELEMFIELD(2) )

C     BETA0s

      CALL PutElemParsToMatlab( MAD_BET0, NBET0, DBET0, pELEMDICT,
     &                          ELEMFIELD(2) )

C     BEAMs

      CALL PutElemParsToMatlab( MAD_BEAM, NBEAM, DBEAM, pELEMDICT,
     &                          ELEMFIELD(2) )

C     KICKERs

      CALL PutElemParsToMatlab( MAD_KICKMAD, NKICKMAD, DKICKMAD, 
     &                                                 pELEMDICT,
     &                          ELEMFIELD(2) )

C     If we've gotten this far, then the element data table is filled 
C     and can be assigned to the return structure.

      PLHS(1) = pELEMDICT

C========1=========2=========3=========4=========5=========6=========7=C
C
C     Filling the EALIGN dictionary structure is simple by comparison
C
C========1=========2=========3=========4=========5=========6=========7=C

      CALL PutElemParsToMatlab( 1, ED_SIZE, EALIGN_DICT, pEALIGNDICT,
     &                          EALIGNFIELD )

9999  PLHS(2) = pEALIGNDICT
      PLHS(1) = pELEMDICT

	RETURN 
	END
