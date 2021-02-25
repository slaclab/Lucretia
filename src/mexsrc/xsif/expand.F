      SUBROUTINE EXPAND(JBEAM,JACT,NELM1,NFREE1,IPOS1,IPOS2,ERROR1)         
C
C     member of MAD INPUT PARSER
C
C---- EXPAND A BEAM LINE                                                 
C
C     MOD:
C		 05-DEC-2003, PT:
C			add allocation of ERRFLG and EALIGN arrays.
C		 19-MAY-2003, PT:
C			dynamically allocate ITEM table based on the actual
C			number of elts required.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C          31-july-1998, PT:
C             replaced ERROR with ERROR1 to avoid conflict with 
C             ERROR in DIMAD_INOUT module.  Replaced NELM with
C             NELM1 and NFREE with NFREE1 for same reason.
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
C
C----------------------------------------------------------------------- 
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N) 
      SAVE
C
      LOGICAL           ERROR1        
	INTEGER*4         LOOP_COUNT, ALLSTAT                                    
C----------------------------------------------------------------------- 
      LOGICAL           FLAG                                             
C----------------------------------------------------------------------- 
C---- INITIALIZE   
	LOOP_COUNT = 0
12345 LOOP_COUNT = LOOP_COUNT + 1                                                      
      ERROR1 = .FALSE.                                                    
      NELM1 = 0                                                           
      IPOS = NFREE1                                                       
      IDIR = 1                                                           
      IREP = 1                                                           
      ICELL = 0                                                          
      IBEAM = JBEAM                                                      
      IHEAD = IEDAT(IBEAM,3)                                             
      IACT = JACT                                                        
      IDIREP = 1                                                         
C---- ENTER NAMED BEAM LINE                                              
   90 CALL FRMSET(IBEAM,IACT,FLAG)                                       
      ERROR1 = ERROR1 .OR. FLAG                                            
C---- ENTER BEAM LINE --- STACK TRACKING PARAMETERS                      
  100 ILDAT(IHEAD,4) = IDIR*IREP                                         
      ILDAT(IHEAD,5) = ICELL                                             
      ILDAT(IHEAD,6) = IBEAM                                             
      IDIR = ISIGN(1,IDIREP)                                             
      IREP = IABS(IDIREP)                                                
      ICELL = IHEAD                                                      
C---- BEGIN TRACKING THROUGH BEAM LINE                                   
  110 IF (IBEAM .NE. 0) THEN
	  IPOS = IPOS + 1
	  IF (LOOP_COUNT .GT. 1) THEN                                             
          IF (IPOS .GT. MAXPOS) CALL OVFLOW(4,MAXPOS)                      
          ITEM(IPOS) = IBEAM + MAXELM                                      
        ENDIF
      ENDIF                                                              
C---- STEP THROUGH LINE                                                  
  120 IF (IDIR .LT. 0) ICELL = ILDAT(ICELL,2)                            
      IF (IDIR .GT. 0) ICELL = ILDAT(ICELL,3)                            
C---- SWITCH ON LIST CELL TYPE                                           
      GO TO (150, 200, 250), ILDAT(ICELL,1)                              
C---- END TRACKING THROUGH BEAM LINE                                     
  150 IBEAM = ILDAT(ICELL,6)                                             
      IF (IBEAM .NE. 0) THEN
	  IPOS = IPOS + 1
	  IF (LOOP_COUNT .GT. 1) THEN                                             
          IF (IPOS .GT. MAXPOS) CALL OVFLOW(4,MAXPOS)                                                                        
          ITEM(IPOS) = IBEAM + MAXELM+MXLINE   
	  ENDIF                            
      ENDIF                                                              
C---- ANY MORE REPETITIONS?                                              
      IREP = IREP - 1                                                    
      IF (IREP .GT. 0) GO TO 110                                         
C---- LEAVE CURRENT BEAM LINE --- UNSTACK TRACKING PARAMETERS            
      IHEAD = ICELL                                                      
      IDIR = ISIGN(1,ILDAT(IHEAD,4))                                     
      IREP = IABS(ILDAT(IHEAD,4))                                        
      ICELL = ILDAT(IHEAD,5)                                             
      IBEAM = ILDAT(IHEAD,6)                                             
      ILDAT(IHEAD,4) = 0                                                 
      ILDAT(IHEAD,5) = 0                                                 
      ILDAT(IHEAD,6) = 0                                                 
      IF (ICELL .NE. 0) GO TO 120
C---- Allocate ITEM and PRTFLG now
	IF (LOOP_COUNT.EQ.1) THEN
	  MAXPOS = IPOS - NFREE1
	  IF (ALLOCATED(ITEM))
     &	DEALLOCATE(ITEM,PRTFLG,ERRFLG,ERRPTR)
	  ALLOCATE(ITEM(MAXPOS),PRTFLG(MAXPOS),
     &		   ERRFLG(MAXPOS),ERRPTR(MAXPOS),
     &		   STAT = ALLSTAT)
	  IF (ALLSTAT.NE.0) THEN
		ERROR1 = .TRUE.
		FATAL_ALLOC_ERROR = .TRUE.
		WRITE(IECHO,950)MAXPOS
		WRITE(ISCRN,950)MAXPOS
		GOTO 54321
	  ENDIF
	  GOTO 12345 ! fill the table for real this time
	ENDIF                                       
C---- PRINT ENDING MESSAGE                                               
54321 LE = LEN_TRIM(KELEM(IBEAM))                                          
      IF (ERROR1) THEN                                                    
        WRITE (IECHO,910) KELEM(IBEAM)(1:LE)                             
        WRITE (ISCRN,910) KELEM(IBEAM)(1:LE)                             
        NFAIL = NFAIL + 1                                                
        IPOS1 = 0                                                        
        IPOS2 = 0                                                        
      ELSE                                                               
        NPOS = IPOS - NFREE1                                              
        WRITE (IECHO,920) KELEM(IBEAM)(1:LE),NELM1,NPOS                   
        WRITE (ISCRN,920) KELEM(IBEAM)(1:LE),NELM1,NPOS                   
        IPOS1 = NFREE1 + 1                                                
        IPOS2 = IPOS                                                     
        NFREE1 = IPOS                                                     
      ENDIF                                                              
      RETURN                                                             
C---- CALL UNNAMED BEAM LINE                                             
  200 IDIREP = ILDAT(ICELL,4)*IDIR                                       
      IF (IDIREP .EQ. 0) GO TO 120                                       
      IHEAD = ILDAT(ICELL,5)                                             
      IBEAM = 0                                                          
      GO TO 100                                                          
C---- CALL NAMED BEAM LINE OR BEAM ELEMENT                               
  250 IDIREP = ILDAT(ICELL,4)*IDIR                                       
      IF (IDIREP .EQ. 0) GO TO 120                                       
      IELEM = ILDAT(ICELL,5)                                             
      IF (IETYP(IELEM)) 260,270,300                                      
C---- CALL UNDEFINED ELEMENT                                             
  260 LE = LEN_TRIM(KELEM(IELEM))                                          
      WRITE (IECHO,930) KELEM(IELEM)(1:LE)                               
      WRITE (ISCRN,930) KELEM(IELEM)(1:LE)                               
      NFAIL = NFAIL + 1                                                  
      ERROR1 = .TRUE.                                                     
      GO TO 120                                                          
C---- CALL FORMAL ARGUMENT OR NAMED BEAM LINE                            
  270 IHEAD = IEDAT(IELEM,3)                                             
      IF (ILDAT(IHEAD,4) .NE. 0) GO TO 280                               
      IF (IELEM .GE. IELEM2) THEN                                        
        IACT  = 0                                                        
        IBEAM = 0                                                        
        GO TO 100                                                        
      ELSE                                                               
        IACT  = ILDAT(ICELL,6)                                           
        IBEAM = IELEM                                                    
        GO TO 90                                                         
      ENDIF                                                              
C---- BEAM LINE CALLS ITSELF                                             
  280 LE = LEN_TRIM(KELEM(IELEM))                                          
      WRITE (IECHO,940) KELEM(IELEM)(1:LE)                               
      WRITE (ISCRN,940) KELEM(IELEM)(1:LE)                               
      NFAIL = NFAIL + 1                                                  
      ERROR1 = .TRUE.                                                     
      GO TO 120                                                          
C---- CALL BEAM ELEMENT                                                  
  300 IEREP = IABS(IDIREP)      
	IF (LOOP_COUNT.GT.1) THEN                                          
        IF (IPOS + IEREP .GT. MAXPOS) CALL OVFLOW(4,MAXPOS)                
	ENDIF
      DO 310 J = 1, IEREP                                                
        NELM1 = NELM1 + 1                                                  
        IPOS = IPOS + 1
	  IF (LOOP_COUNT.GT.1)                                                  
     &    ITEM(IPOS) = IELEM                                               
  310 CONTINUE                                                           
      GO TO 120                                                          
C----------------------------------------------------------------------- 
  910 FORMAT('0*** ERROR *** EXPANSION OF "',A,'" FAILED'/' ')           
  920 FORMAT('0... BEAM LINE "',A,'" EXPANDED:',                         
     +       I10,' ELEMENTS,',I10,' POSITIONS')                          
  930 FORMAT('0*** ERROR *** UNDEFINED ELEMENT "',A,                     
     +       '" ENCOUNTERED DURING EXPANSION')                           
  940 FORMAT('0*** ERROR *** THE BEAM LINE "',A,'" REFERS TO ITSELF')    
  950 FORMAT(' *** ERROR *** UNABLE TO ALLOCATE ',I10,' ENTRIES ',
     &		'IN POSITION TABLE'/' ')
C----------------------------------------------------------------------- 
      END                                                                
