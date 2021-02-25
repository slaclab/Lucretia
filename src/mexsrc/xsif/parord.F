      SUBROUTINE PARORD(ERROR1)                                           
C
C     member of MAD INPUT PARSER
C
C---- SET UP ORDERED LIST FOR EVALUATION OF DEPENDENT PARAMETERS         
C----------------------------------------------------------------------- 
C	
C	MOD:
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL           ERROR1                                            
C----------------------------------------------------------------------- 
      ERROR1 = .FALSE.                                                    
C---- FLAG DEPENDENT PARAMETERS                                          
      IPDEP = 0                                                          
      IPLIST = 0                                                         
      DO 10 IPARM = 1, IPARM1                                            
        IF (IPTYP(IPARM) .GT. 0) THEN                                    
          IPDEP = IPDEP + 1                                              
          IPNEXT(IPARM) = -1                                             
        ELSE                                                             
          IPNEXT(IPARM) = 0                                              
        ENDIF                                                            
   10 CONTINUE                                                           
      DO 20 IPARM = IPARM2, MAXPAR                                       
        IF (IPTYP(IPARM) .GT. 0) THEN                                    
          IPDEP = IPDEP + 1                                              
          IPNEXT(IPARM) = -1                                             
        ELSE                                                             
          IPNEXT(IPARM) = 0                                              
        ENDIF                                                            
   20 CONTINUE                                                           
      IF (IPDEP .EQ. 0) RETURN                                           
C---- ORDER THE TABLE JUST CREATED ------------------------------------- 
C---- PASS THROUGH LIST TO FIND PARAMETERS WHOSE OPERANDS ARE DEFINED    
  100 ICOUNT = IPDEP                                                     
C---- DEFINE GLOBAL PARAMETERS, IF POSSIBLE                              
      DO 140 IPARM = 1, IPARM1                                           
        IF (IPNEXT(IPARM) .LT. 0) THEN                                   
          IOP2 = IPDAT(IPARM,2)                                          
          IOP1 = IOP2                                                    
          IF (IPTYP(IPARM) .LE. 10) IOP1 = IPDAT(IPARM,1)                
          IF (IPNEXT(IOP1) .GE. 0 .AND. IPNEXT(IOP2) .GE. 0) THEN        
            IPDEP = IPDEP - 1                                            
            IPNEXT(IPARM) = IPLIST                                       
            IPLIST = IPARM                                               
          ENDIF                                                          
        ENDIF                                                            
  140 CONTINUE                                                           
C---- DEFINE ELEMENT PARAMETERS, IF POSSIBLE                             
      DO 190 IPARM = IPARM2, MAXPAR                                      
        IF (IPNEXT(IPARM) .LT. 0) THEN                                   
          IOP2 = IPDAT(IPARM,2)                                          
          IOP1 = IOP2                                                    
          IF (IPTYP(IPARM) .LE. 10) IOP1 = IPDAT(IPARM,1)                
          IF (IPNEXT(IOP1) .GE. 0 .AND. IPNEXT(IOP2) .GE. 0) THEN        
            IPDEP = IPDEP - 1                                            
            IPNEXT(IPARM) = IPLIST                                       
            IPLIST = IPARM                                               
          ENDIF                                                          
        ENDIF                                                            
  190 CONTINUE                                                           
C---- END OF PASS --- ALL DEFINED?                                       
      IF (IPDEP .LE. 0) GO TO 300                                        
C---- ANY DEFINITIONS ADDED IN LAST PASS?                                
      IF (IPDEP .LT. ICOUNT) GO TO 100                                   
C---- CIRCULAR DEFINITIONS LEFT                                          
      WRITE (IECHO,910)                                                  
      WRITE (ISCRN,910)                                                  
      NFAIL = NFAIL + 1                                                  
      ERROR1 = .TRUE.                                                     
      DO 240 IPARM = 1, IPARM1                                           
        IF (IPNEXT(IPARM) .LT. 0) THEN                                   
          CALL PARPRT(IPARM)                                             
          PDATA(IPARM) = 0.0                                             
          IPNEXT(IPARM) = 0                                              
        ENDIF                                                            
  240 CONTINUE                                                           
      DO 290 IPARM = IPARM2, MAXPAR                                      
        IF (IPNEXT(IPARM) .LT. 0) THEN                                   
          CALL PARPRT(IPARM)                                             
          PDATA(IPARM) = 0.0                                             
          IPNEXT(IPARM) = 0                                              
        ENDIF                                                            
  290 CONTINUE                                                           
C---- REVERSE LIST (IT WAS GENERATED IN REVERSE ORDER!)                  
  300 IF (IPLIST .EQ. 0) RETURN                                          
      IPARM = IPLIST                                                     
      IPLIST = 0                                                         
  310 IPSUCC = IPNEXT(IPARM)                                             
        IPNEXT(IPARM) = IPLIST                                           
        IPLIST = IPARM                                                   
        IPARM = IPSUCC                                                   
      IF (IPARM .NE. 0) GO TO 310                                        
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('0*** ERROR *** CIRCULAR PARAMETER DEFINITION(S):')         
C----------------------------------------------------------------------- 
      END                                                                
