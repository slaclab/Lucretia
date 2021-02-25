      SUBROUTINE ELMDEF(KKEYW,LKEYW,KNAME,LNAME)                         
C
C     member of MAD INPUT PARSER
C
C---- DECODE ELEMENT DEFINITION                                          
C
C     MOD:
C		 23-DEC-2003, PT:
C			set IBEAM_PTR and IBETA0_PTR to most recent BEAM/
C			BETA0 statement unless user has selected them via
C			USE statement.
C          15-dec-2003, PT:
C             expand element names to 16 characters.  Add
C             KICKER element.
C          15-nov-2002, PT:
C             add support for BEAM and BETA0 elements.
C          23-JAN-2002, PT:
C             bugfix:  make HGAPX/FINTX, if unassigned, into
C             pointers to th HGAP/FINT variables.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C          05-JAN-2001, PT:
c             only issue non-NLC standard warning if NLC_STD
C             flag is TRUE.
C          16-jun-1999, PT:
C             change default roll angles to conform to MAD 8 
C             convention
C          14-jun-1999, PT:
C             add emulation of MAD element class feature.
C          01-jun-1999, PT:
C             bugfix for DIMULTs which sets default tilt angles,
C             SCALEFAC and APERTURE values.
C          05-MAR-1999, PT:
C             improved NLC-std checking code
C		 23-NOV-1998, PT:
C			eliminated TORO, added SLMO per MDW.
C		 30-oct-1998, PT:
C			add code to warn if non-NLC-standard elements are
C			used.
C		 29-oct-1998, PT:
C			add code to handle MDW instrument sub-types and new
C			DIMULTI multipole type.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C          26-AUG-1998, PT:
C             replaced long list of MAD element keywords etc. with
C             USE XSIF_ELEM_PARS call.  Replaced computed GOTO with
C             CASE statement.  Added code for INSTRUMENT (INST) type.
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
      USE XSIF_ELEM_PARS
      USE XSIF_CONSTANTS
C
C----------------------------------------------------------------------- 
C
C      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      IMPLICIT NONE
      SAVE
C
      CHARACTER*8       KKEYW
	CHARACTER(ENAME_LENGTH) KNAME
C
      INTEGER*4         IEP2, IP, IEP1, LNAME, LKEYW, IELEM, IKEYW
      INTEGER*4         I_TEMPLATE ! address of template element
      INTEGER*4         IEPT1, IEPT2, IPT
C
C----------------------------------------------------------------------- 
C
C---- IF OLD FORMAT, READ ELEMENT NAME                                   
      IF (LNAME .EQ. 0) THEN                                             
        CALL RDTEST(',',ERROR)                                           
        IF (ERROR) RETURN                                                
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        CALL RDWORD(KNAME,LNAME)                                         
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
          WRITE (IECHO,910)                                              
          WRITE (ISCRN,910)                                              
          ERROR = .TRUE.                                                 
          RETURN                                                         
        ENDIF                                                            
      ENDIF                   
c
c     initialize I_TEMPLATE to zero
c
      I_TEMPLATE = 0
                                           
C---- LOOK UP ELEMENT KEYWORD                                            
      CALL RDLOOK(KKEYW,LKEYW,DKEYW,1,NKEYW,IKEYW)
c
c     if the element keyword was not found in the dictionary, it
c     may be that this element is an instance of another element;
c     look for KKEYW in the list of element names
c
      IF ( IKEYW .EQ. 0 ) THEN
          CALL RDLOOK(KKEYW,LKEYW,KELEM(1),1,IELEM1,I_TEMPLATE)
          IF ( I_TEMPLATE .NE. 0 ) THEN
              IKEYW = IETYP(I_TEMPLATE)
          ENDIF
      ENDIF
C
C     if we still haven't found it, then it really can't be found
C                       
      IF (IKEYW .EQ. 0) THEN                                             
        CALL RDWARN                                                      
        WRITE (IECHO,920) KKEYW(1:LKEYW)                                 
        WRITE (ISCRN,920) KKEYW(1:LKEYW)                                 
        IKEYW = 1                                                        
      ENDIF      
      IKEYW_GLOBAL = IKEYW                                                        
C---- ALLOCATE ELEMENT CELL AND TEST FOR REDEFINITION                    
      CALL FNDELM(ILCOM,KNAME,IELEM)                                     
      IF (IETYP(IELEM) .GE. 0) THEN   
c
c     if it's a protected element (ie, if ELEM_LOCKED is TRUE),
c     don't allow a redefinition, set an error flag, and
c     return to CNTROL (which handles error-recovery)
c       
        IF ( ELEM_LOCKED(IELEM) ) THEN
          CALL RDFAIL
          WRITE(IECHO,950)KELEM(IELEM),IELIN(IELEM)
          WRITE(ISCRN,950)KELEM(IELEM),IELIN(IELEM)
          ERROR = .TRUE.
          RETURN
        ELSE                                   
          CALL RDWARN                                                      
          WRITE (IECHO,930) IELIN(IELEM)                                   
          WRITE (ISCRN,930) IELIN(IELEM)                                   
          IETYP(IELEM) = -1
        ENDIF
                                               
      ENDIF                                                              
C---- DEFINE ELEMENT PARAMETER LIST                                      
C
C      GO TO ( 10, 20, 30, 40, 50, 60, 70, 80, 90,100,                    
C     +       110,120,130,140,150,160,170,180,190,200,                    
C     +       210,220,230,240,250,260,270) IKEYW                          
C
C	IF (  (IKEYW .EQ. MAD_RBEND)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_WIGG)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_SEPA)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_ROLL)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_ZROT)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_QUSE)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_GKICK)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_ARBIT)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_MTWIS)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_MATR)
C     &			   .OR.
C     &	  (IKEYW .EQ. MAD_DIMU)   ) THEN
      IF ( .NOT. NLC_KEYW(IKEYW)  ) THEN
        if (NLC_STD) THEN
		CALL RDWARN
		WRITE(IECHO,940)KKEYW,KNAME
		WRITE(ISCRN,940)KKEYW,KNAME
	    NLC_STANDARD = .FALSE.
        endif
	ELSE
	    NLC_STANDARD = .TRUE.
	ENDIF

      SELECT CASE ( IKEYW )
C
C---- "DRIFT" --- DRIFT SPACE                                            
C
      CASE ( MAD_DRIFT ) 
C
   10 CONTINUE                                                           
        CALL DEFPAR(NDRFT,DDRFT,IEP1,IEP2,ILCOM)                         
      GO TO 500
C                                                          
C---- "RBEND" --- RECTANGULAR BENDING MAGNET                             
   20 CONTINUE                                                           
C---- "SBEND" --- SECTOR BENDING MAGNET                                  
C
      CASE ( MAD_RBEND, MAD_SBEND, MAD_WIGG )
C 
   30 CONTINUE                                                           
        CALL DEFPAR(NBEND,DBEND,IEP1,IEP2,ILCOM)                         
        IPTYP(IEP1+5) = -2                                               
c        PDATA(IEP1+5) = -PI / 2.0                                         
c        IF(KFLAGU.EQ.2) PDATA(IEP1+5) = -90.                              
        PDATA(IEP1+5) = PI / 2.0                                         
        IF(KFLAGU.EQ.2) PDATA(IEP1+5) = 90.                              
        PDATA(IEP1+10) = 0.5                                             
      GO TO 500                             
C                             
C---- "WIGGLER" --- WIGGLER MAGNET                                       
   40 CONTINUE                                                           
      GO TO 30                                                           
C---- "QUADRUPO" --- QUADRUPOLE                                          
C
      CASE ( MAD_QUAD )
C
   50 CONTINUE                                                           
        CALL DEFPAR(NQUAD,DQUAD,IEP1,IEP2,ILCOM)                         
        IPTYP(IEP1+2) = -2                                               
c        PDATA(IEP1+2) = -PI / 4.0                                         
c        IF(KFLAGU.EQ.2) PDATA(IEP1+2) = -45.                              
        PDATA(IEP1+2) = PI / 4.0                                         
        IF(KFLAGU.EQ.2) PDATA(IEP1+2) = 45.                              
        PDATA(IEP1+3) = 1.0                                              
      GO TO 500                                                          
C
C---- "SEXTUPOL" --- SEXTUPOLE                                           
C
      CASE ( MAD_SEXT )
C
   60 CONTINUE                                                           
        CALL DEFPAR(NSEXT,DSEXT,IEP1,IEP2,ILCOM)                         
        IPTYP(IEP1+2) = -2                                               
c        PDATA(IEP1+2) = -PI / 6.0                                         
c        IF(KFLAGU.EQ.2) PDATA(IEP1+2) = -30.                              
        PDATA(IEP1+2) = PI / 6.0                                         
        IF(KFLAGU.EQ.2) PDATA(IEP1+2) = 30.                              
        PDATA(IEP1+3) = 1.0                                              
      GO TO 500                                                          
C
C---- "OCTUPOLE" --- OCTUPOLE                                            
C
      CASE ( MAD_OCTU )
C
   70 CONTINUE                                                           
        CALL DEFPAR(NOCT,DOCT,IEP1,IEP2,ILCOM)                           
        IPTYP(IEP1+2) = -2                                               
c        PDATA(IEP1+2) = -PI / 8.0                                         
c        IF(KFLAGU.EQ.2) PDATA(IEP1+2) = -22.5                             
        PDATA(IEP1+2) = PI / 8.0                                         
        IF(KFLAGU.EQ.2) PDATA(IEP1+2) = 22.5                             
        PDATA(IEP1+3) = 1.0                                              
      GO TO 500                                                          
C
C---- "MULTIPOL" --- GENERAL MULTIPOLE                                   
C
      CASE ( MAD_MULTI )
C
   80 CONTINUE                                                           
        CALL DEFPAR(NMULT,DMULT,IEP1,IEP2,ILCOM)                         
        DO 85 IP = 2, 42, 2                                              
          IPTYP(IEP1+IP) = -2                                            
c          PDATA(IEP1+IP) = -PI / FLOAT(IP)                                
c          IF(KFLAGU.EQ.2) PDATA(IEP1+IP) = -180./FLOAT(IP)                
          PDATA(IEP1+IP) = PI / FLOAT(IP)                                
          IF(KFLAGU.EQ.2) PDATA(IEP1+IP) = 180./FLOAT(IP)                
   85   CONTINUE                                                         
        PDATA(IEP1+43) = 1.0                                             
        PDATA(IEP1+44) = 1.0                                             
      GO TO 500                                                          
C
C---- "SOLENOID" --- SOLENOID                                            
C
      CASE ( MAD_SOLN )
C
   90 CONTINUE                                                           
        CALL DEFPAR(NSOLO,DSOLO,IEP1,IEP2,ILCOM)                         
        IPTYP(IEP1+3) = -2                                               
c        PDATA(IEP1+3) = -PI / 4.0                                         
c        IF(KFLAGU.EQ.2) PDATA(IEP1+3) = -45.                              
        PDATA(IEP1+3) = PI / 4.0                                         
        IF(KFLAGU.EQ.2) PDATA(IEP1+3) = 45.                              
        PDATA(IEP1+4) = 1.0                                              
      GO TO 500                                                          
C
C---- "RF" --- RF CAVITY                                                 
C
      CASE ( MAD_RFCAV )
C
  100 CONTINUE                                                           
        CALL DEFPAR(NCVTY,DCVTY,IEP1,IEP2,ILCOM)                         
      GO TO 500                                                          
C
C---- "RF" --- RF LINAC CAVITY                                           
C
      CASE ( MAD_LCAV )
C
  270 CONTINUE                                                           
        CALL DEFPAR(NLCAV,DLCAV,IEP1,IEP2,ILCOM)                         
      GO TO 500                                                          
C
C---- "SEPARATOR" --- ELECTROSTATIC SEPARATOR                            
C
      CASE ( MAD_SEPA )
C
  110 CONTINUE                                                           
        CALL DEFPAR(NSEPA,DSEPA,IEP1,IEP2,ILCOM)                         
      GO TO 500                                                          
C
C---- "ROLL" --- ROTATE AROUND LONGITUDINAL AXIS                         
C---- "ZROT" --- ROTATE AROUND VERTICAL AXIS                             
C    
      CASE ( MAD_ROLL, MAD_ZROT, MAD_YROT, MAD_SROT )
C
 120  CONTINUE                                                           
  130 CONTINUE
                                       
	  IF ( IKEYW .EQ. MAD_YROT ) THEN
		IKEYW = MAD_ZROT
	  ENDIF
	  IF ( IKEYW .EQ. MAD_SROT ) THEN
		IKEYW = MAD_ROLL
	  ENDIF
                        
        CALL DEFPAR(NROTA,DROTA,IEP1,IEP2,ILCOM)                         
      GO TO 500                                                          
C
C---- "HKICK" --- HORIZONTAL ORBIT CORRECTOR                             
C---- "VKICK" --- VERTICAL ORBIT CORRECTOR                               
C
      CASE ( MAD_HKICK, MAD_VKICK )
C
  140 CONTINUE                                                           
  150 CONTINUE                                                           
        CALL DEFPAR(NKICK,DKICK,IEP1,IEP2,ILCOM)                         
      GO TO 500       
C
C---- "KICKER" -- combined horizontal/vertical corrector
C
      CASE ( MAD_KICKMAD )
        CALL DEFPAR(NKICKMAD,DKICKMAD,IEP1,IEP2,ILCOM)
        GOTO 500                                                   
C
C--- "HMON, VMON, MON" -- MONITORS                                       
C
      CASE ( MAD_HMON, MAD_VMON, MAD_MONI )
C
  160 CONTINUE                                                           
  170 CONTINUE                                                           
  180 CONTINUE                                                           
        CALL DEFPAR(NMON,DMON,IEP1,IEP2,ILCOM)                           
      GO TO 500                                                          
C
C---- "ECOLLIMA" --- ELLIPTIC COLLIMATOR                                 
C---- "RCOLLIMA" --- RECTANGULAR COLLIMATOR                              
C
      CASE ( MAD_ECOLL, MAD_RCOLL )
C
  200 CONTINUE                                                           
  210 CONTINUE                                                           
        CALL DEFPAR(NCOLL,DCOLL,IEP1,IEP2,ILCOM)                         
        PDATA(IEP1+1) = 1.0                                              
        PDATA(IEP1+2) = 1.0                                              
      GO TO 500                                                          
C
C---- "MARKER" --- MARKER ELEMENT                                        
C
      CASE ( MAD_MARK )
C
  190 CONTINUE                                                           
        IEP1 = 0                                                         
        IEP2 = 0                                                         
      GO TO 500                                                          
C
C---- "QUADSEXT"--- QUADRUPOLE-SEXTUPOLE COMBINATION                     
C
      CASE ( MAD_QUSE )
C
  220 CONTINUE                                                           
        CALL DEFPAR(NQUSE,DQUSE,IEP1,IEP2,ILCOM)                         
        IPTYP(IEP1+3) = -2                                               
c        PDATA(IEP1+3) = -PI / 4.0                                         
c        IF(KFLAGU.EQ.2) PDATA(IEP1+3) = -45.                              
        PDATA(IEP1+3) = PI / 4.0                                         
        IF(KFLAGU.EQ.2) PDATA(IEP1+3) = 45.                              
        PDATA(IEP1+4) = 1.0                                              
      GO TO 500                                                          
C
C---- "GKICK" --- GENERAL KICK A LA DIMAT                                
C
      CASE ( MAD_GKICK )
C
  230 CONTINUE                                                           
        CALL DEFPAR(NGKIK,DGKIK,IEP1,IEP2,ILCOM)                         
        PDATA(IEP1+9) = 1.0                                              
      GO TO 500                                                          
C
C--- ARBITRARY ELEMENT                                                   
C
      CASE ( MAD_ARBIT )
C
  240 CONTINUE                                                           
        CALL DEFPAR(NARBI,DARBI,IEP1,IEP2,ILCOM)                         
      GO TO 500                                                          
C
C--- TWISS MATRIX
C                                                        
      CASE ( MAD_MTWIS )
C
 250  CONTINUE                                                           
        CALL DEFPAR(NTWIS,DTWIS,IEP1,IEP2,ILCOM)                         
        PDATA(IEP1+2) = 1.0                                              
        PDATA(IEP1+5) = 1.0                                              
      GO TO 500                                                          
C
C--- GENERAL MATRIX                                                      
C
      CASE ( MAD_MATR )
C
  260 CONTINUE                                                           
        CALL DEFPAR(NMATR,DMATR,IEP1,IEP2,ILCOM)                         
C        PDATA(IEP1+1) = 1.0                                             
C        PDATA(IEP1+29) = 1.0                                            
C        PDATA(IEP1+57) = 1.0                                            
C        PDATA(IEP1+85) = 1.0                                            
C        PDATA(IEP1+113) = 1.0                                           
C        PDATA(IEP1+141) = 1.0                                           
C
C--- INSTRUMENTs of all kinds
C
        CASE( MAD_INST, MAD_SLMO, MAD_PROF, MAD_WIRE,
     &		MAD_BLMO, MAD_IMON )
C
           CALL DEFPAR(NINST,DINST,IEP1,IEP2,ILCOM)
C
	  CASE ( MAD_DIMU )

		 CALL DEFPAR(NMULT,DDIMU,IEP1,IEP2,ILCOM)
           DO 895 IP = 2, 42, 2                                              
              IPTYP(IEP1+IP) = -2                                            
c              PDATA(IEP1+IP) = -PI / FLOAT(IP)                                
c              IF(KFLAGU.EQ.2) PDATA(IEP1+IP) = -180./FLOAT(IP)                
              PDATA(IEP1+IP) = PI / FLOAT(IP)                                
              IF(KFLAGU.EQ.2) PDATA(IEP1+IP) = 180./FLOAT(IP)                
  895      CONTINUE                                                         
           PDATA(IEP1+43) = 1.0                                             
           PDATA(IEP1+44) = 1.0                                             
c           GO TO 500    

        CASE ( MAD_BET0 )

          CALL DEFPAR(NBET0,DBET0,IEP1,IEP2,ILCOM)
		IF (.NOT. BETA0_FROM_USE) IBETA0_PTR = IELEM                                                      

        CASE ( MAD_BEAM )

          CALL DEFPAR(NBEAM,DBEAM,IEP1,IEP2,ILCOM)
		IF (.NOT. BEAM_FROM_USE) IBEAM_PTR = IELEM                                                      

      END SELECT
C
  500 CONTINUE                                                           
C
C
C---- DECODE PARAMETER LIST, IF ANY    
                                 
      CALL DECPAR(IEP1,IEP2,KTYPE,KLABL,ERROR)                           
      IF (FATAL_READ_ERROR) GOTO 9999
C---- CHECK ON HGAPX AND FINTX FOR BEND                                  
      IF (IKEYW.EQ.2 .OR. IKEYW.EQ.3 .OR. IKEYW.EQ.4) THEN               
C        IF(IPTYP(IEP1+11).EQ.-1) PDATA(IEP1+11) = PDATA(IEP1+9)          
        IF (IPTYP(IEP1+11).EQ.-1) THEN
          IPTYP(IEP1+11) = 11
          IPDAT(IEP1+11,1) = 0
          IPDAT(IEP1+11,2) = IEP1+9
        ENDIF
c        IF(IPTYP(IEP1+12).EQ.-1) PDATA(IEP1+12) = PDATA(IEP1+10)         
        IF (IPTYP(IEP1+12).EQ.-1) THEN
          IPTYP(IEP1+12) = 11
          IPDAT(IEP1+12,1) = 0
          IPDAT(IEP1+12,2) = IEP1+10
        ENDIF
      ENDIF                                                              
      IF (ERROR) RETURN                                                  
      KETYP(IELEM) = KTYPE                                               
      KELABL(IELEM) = KLABL                                              
      IETYP(IELEM) = IKEYW                                               
      IEDAT(IELEM,1) = IEP1                                              
      IEDAT(IELEM,2) = IEP2     
c
c     If this element is an instance of another element, which is
c     a class template, then copy all undefined stuff over.  Also,
c     make the template element protected                                        
c
      IF ( I_TEMPLATE .NE. 0 ) THEN
          IEPT1 = IEDAT(I_TEMPLATE,1)
          IEPT2 = IEDAT(I_TEMPLATE,2)
          IPT = IEPT1
          DO IP = IEP1, IEP2    
              IF ( IPTYP(IP) .LT. 0 ) THEN
                  IPTYP(IP) = IPTYP(IPT)
                  PDATA(IP) = PDATA(IPT)
                  IPDAT(IP,1) = IPDAT(IPT,1)
                  IPDAT(IP,2) = IPDAT(IPT,2)
              ENDIF
              IPT = IPT + 1
          ENDDO
          IF ( LEN_TRIM(KELABL(IELEM)) .EQ. 0 ) THEN
              KELABL(IELEM) = KELABL(I_TEMPLATE)
          ENDIF
          IF ( LEN_TRIM(KETYP(IELEM)) .EQ. 0 ) THEN
              KETYP(IELEM) = KETYP(I_TEMPLATE)
          ENDIF
          ELEM_LOCKED(I_TEMPLATE) = .TRUE.
      ENDIF

9999  IF (FATAL_READ_ERROR) ERROR = .TRUE.


      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** ELEMENT NAME EXPECTED'/' ')                 
  920 FORMAT(' ** WARNING ** UNKNOWN ELEMENT KEYWORD "',A,               
     +       '" --- "DRIFT" ASSUMED'/' ')                                
  930 FORMAT(' ** WARNING ** THE ABOVE NAME WAS DEFINED IN LINE ',I5,    
     +       ', IT WILL BE REDEFINED'/' ')                               
  940 FORMAT(' ** WARNING ** NON-"NLC-STANDARD" ELEMENT KEYWORD "',A,
     +	   '" DETECTED, ELEMENT "',A,'".'/' ')
  950 FORMAT(' *** ERROR *** TRYING TO REDEFINE PROTECTED NAME "',A,
     +       '",',/,15X,'OLD VERSION (DEFINED IN LINE ',I5,') KEPT.',/)
C----------------------------------------------------------------------- 
      END                                                                
