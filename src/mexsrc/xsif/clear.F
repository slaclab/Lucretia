      SUBROUTINE CLEAR                                                          
C
C     part of MAD INPUT PARSER
C
C---- CLEAR DATA STRUCTURE                                                      
C
C	MOD:
C          27-feb-2004, PT:
C             initialize LINE_EXPANDED to FALSE.
C		 19-MAY-2003, PT:
C			move initialization of stuff related to dynamically-
C			allocated stuff to XSIF_ALLOCATE_INITIAL.
C          02-MAR-2001, PT:
C             initialize XSIF_STOP to FALSE.  Initialize FATAL_READ_
C             ERROR to FALSE.
C          20-JAN-2000, PT:
C             initialization of some DIMAT inst parameters moved
C             here from DIMAD_ELEM_PARS for Solaris compatibility.
C          11-JAN-2000, PT:
C             associate PATH_PTR with PATH_LCL (".")
C          05-MAR-1999, PT:
C             set appropriate values of NLC_PARAM to TRUE (indicates
C             an NLC-std. parameter in an NLC-std. element)
C		 28-sep-1998, PT:
C			pre-allocate first 8 cells of PDATA to hold
C			MAD's predefined constants (see table 2.2 in
C			the MAD version 8.1x manuals); set appropriate
C			values in MAD_LOCATION integer array.
C	     15-SEP-1998, PT:
C	        set NUM_LWAKE and NUM_TWAKE to zero.
C		 28-AUG-1998, PT:
C			added NUM_CALL = 0 statement
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
      USE XSIF_CONSTANTS
	USE XSIF_ELEM_PARS
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)                        
      SAVE
C-----------------------------------------------------------------------        
      INTEGER*4 ILOOP, JLOOP
	LOGICAL*4 XSIF_ALLOCATE_INITIAL, ALLSTAT

      INVAL  = .FALSE.                                                          
      NEWCON = .FALSE.                                                          
      NEWPAR = .FALSE.                                                          
      PERI   = .FALSE.                                                          
      STABX  = .FALSE.                                                          
      STABZ  = .FALSE.                                                          
      SYMM   = .FALSE.       
C
      XSIF_STOP = .FALSE.   
      FATAL_READ_ERROR = .FALSE.    
      LINE_EXPANDED = .FALSE.                                            
C                                                                               
C      IELEM1 = 0                                                                
C      IELEM2 = MAXELM + 1                                                       
C                                                                               
C      IUSED  = 0                                                                
C                                                                               
C      IPARM1 = 0                                                                
C      IPARM2 = MAXPAR + 1                                                       
C      IPLIST = 0
C	
C	NUM_CALL = 0
C	NUM_LWAKE = 0
C	NUM_TWAKE = 0                                                                
C
	ALLSTAT = XSIF_ALLOCATE_INITIAL()
	IF (ERROR) RETURN
C                                                                               C
C	set up the appropriate parameters
C
	IPARM1 = MX_SHARECON

	KPARM(1) = 'PI'
	PDATA(1) = PI_INIT
	IPTYP(1) = 0
	IPDAT(1,1) = 0
	IPDAT(1,2) = 0
	IPLIN(1)   = 0
	MAD_LOCATION(PI_INDEX) = 1
	PI = PI_INIT
C	                                                          
	KPARM(2) = 'TWOPI'
	PDATA(2) = 2.D0 * PI_INIT
	IPTYP(2) = 0
	IPDAT(2,1) = 0
	IPDAT(2,2) = 0
	IPLIN(2)   = 0
	TWOPI = 2.D0*PI_INIT
	MAD_LOCATION(TWOPI_INDEX) = 2
C	                                                          
	KPARM(3) = 'DEGRAD'
	PDATA(3) = DEGRAD_INIT
	IPTYP(3) = 0
	IPDAT(3,1) = 0
	IPDAT(3,2) = 0
	IPLIN(3)   = 0
	MAD_LOCATION(DEGRAD_INDEX) = 3
C                                                                               
	KPARM(4) = 'RADDEG'
	PDATA(4) = 1.D0 / DEGRAD_INIT
	IPTYP(4) = 0
	IPDAT(4,1) = 0
	IPDAT(4,2) = 0
	IPLIN(4)   = 0
	CRDEG = 1.D0 / DEGRAD_INIT
	MAD_LOCATION(RADDEG_INDEX) = 4
C
	KPARM(5) = 'E'
	PDATA(5) = E_INIT
	IPTYP(5) = 0
	IPDAT(5,1) = 0
	IPDAT(5,2) = 0
	IPLIN(5)   = 0
	MAD_LOCATION(E_INDEX) = 5
C
	KPARM(6) = 'EMASS'
	PDATA(6) = EMASS_INIT
	IPTYP(6) = 0
	IPDAT(6,1) = 0
	IPDAT(6,2) = 0
	IPLIN(6)   = 0
	EMASS = EMASS_INIT
	MAD_LOCATION(EMASS_INDEX) = 6
C
	KPARM(7) = 'PMASS'
	PDATA(7) = PMASS_INIT
	IPTYP(7) = 0
	IPDAT(7,1) = 0
	IPDAT(7,2) = 0
	IPLIN(7)   = 0
	MAD_LOCATION(PMASS_INDEX) = 7
C
	KPARM(8) = 'CLIGHT'
	PDATA(8) = CLIGHT_INIT
	IPTYP(8) = 0
	IPDAT(8,1) = 0
	IPDAT(8,2) = 0
	IPLIN(8)   = 0
	CLIGHT = CLIGHT_INIT
	MAD_LOCATION(CLIGHT_INDEX) = 8
C
C     initialize NLC_PARAM
C
      NLC_PARAM(MAD_DRIFT,1) = .TRUE.
	DO ILOOP = 1,NBEND
	    NLC_PARAM(MAD_SBEND,ILOOP) = .TRUE.
	ENDDO
      NLC_PARAM(MAD_SBEND,12) = .FALSE.
	DO ILOOP = 1,NQUAD
	    NLC_PARAM(MAD_QUAD,ILOOP) = .TRUE.
	ENDDO
	DO ILOOP = 1,NSEXT
	    NLC_PARAM(MAD_SEXT,ILOOP) = .TRUE.
	ENDDO
	DO ILOOP = 1,NOCT
	    NLC_PARAM(MAD_OCTU,ILOOP) = .TRUE.
	ENDDO
      DO ILOOP = 1,NSOLO
	    NLC_PARAM(MAD_SOLN,ILOOP) = .TRUE.
	ENDDO
	NLC_PARAM(MAD_SOLN,3) = .FALSE.
      DO ILOOP = 1,NCVTY
	    NLC_PARAM(MAD_RFCAV,ILOOP) = .TRUE.
	ENDDO
	NLC_PARAM(MAD_RFCAV,5) = .FALSE.
	DO ILOOP = 1,NLCAV
	    NLC_PARAM(MAD_LCAV,ILOOP) = .TRUE.
	ENDDO
	NLC_PARAM(MAD_LCAV,6) = .FALSE.
	NLC_PARAM(MAD_LCAV,7) = .FALSE.
	DO ILOOP = 1,NKICK
	    NLC_PARAM(MAD_HKICK,ILOOP) = .TRUE.
	    NLC_PARAM(MAD_VKICK,ILOOP) = .TRUE.
	ENDDO
	NLC_PARAM(MAD_HMON,1) = .TRUE.
	NLC_PARAM(MAD_VMON,1) = .TRUE.
	NLC_PARAM(MAD_MONI,1) = .TRUE.
	DO ILOOP = 1,NCOLL
	    NLC_PARAM(MAD_ECOLL,ILOOP) = .TRUE.
	    NLC_PARAM(MAD_RCOLL,ILOOP) = .TRUE.
	ENDDO
      NLC_PARAM(MAD_INST,1) = .TRUE.
	NLC_PARAM(MAD_BLMO,1) = .TRUE.
	NLC_PARAM(MAD_PROF,1) = .TRUE.
	NLC_PARAM(MAD_WIRE,1) = .TRUE.
	NLC_PARAM(MAD_SLMO,1) = .TRUE.
	NLC_PARAM(MAD_IMON,1) = .TRUE.
	NLC_PARAM(MAD_YROT,1) = .TRUE.
	NLC_PARAM(MAD_SROT,1) = .TRUE.

      PATH_PTR => PATH_LCL

      RETURN                                                                    
C-----------------------------------------------------------------------        
      END                                                                       
