      SUBROUTINE PAREVL
C
C     member of MAD INPUT PARSER
C
C---- EVALUATE COUPLED PARAMETERS                                        
C
C	MOD:
C		 29-oct-1998, PT:
C			added ABS and ASIN for MAD compatibility.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C
C --  Modified by PT on 14-Feb-1996 in the following way:  
C     added ATAN to functions evaluated here for use in tunable bend
C     magnets.  Note that adding functions here also requires 
C     modification of DECEXP in dimad02.f to decode them properly.
C
C----------------------------------------------------------------------- 
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
C----------------------------------------------------------------------- 
      IPARM = IPLIST                                                     
      IF (IPARM .EQ. 0) RETURN                                           
C---- PERFORM OPERATION                                                  
  100   ITYPE = IPTYP(IPARM)                                             
        IF (ITYPE .LE. 0 .OR. ITYPE .GT. 21) GO TO 500                   
        IOP1 = IPDAT(IPARM,1)                                            
        IOP2 = IPDAT(IPARM,2)                                            
        GO TO (110,120,130,140,500,500,500,500,500,500,                  
     +         210,220,230,240,250,260,270,280,290,300,310), ITYPE           
C---- BINARY OPERATIONS                                                  
  110     PDATA(IPARM) = PDATA(IOP1) + PDATA(IOP2)                       
        GO TO 500                                                        
  120     PDATA(IPARM) = PDATA(IOP1) - PDATA(IOP2)                       
        GO TO 500                                                        
  130     PDATA(IPARM) = PDATA(IOP1)*PDATA(IOP2)                         
        GO TO 500                                                        
  140     IF (PDATA(IOP2) .EQ. 0.0) THEN                                 
            WRITE (IECHO,910)                                            
            WRITE (ISCRN,910)                                            
            CALL PARPRT(IPARM)                                           
            NWARN = NWARN + 1                                            
            PDATA(IPARM) = 0.0                                           
          ELSE                                                           
            PDATA(IPARM) = PDATA(IOP1) / PDATA(IOP2)                     
          ENDIF                                                          
        GO TO 500                                                        
C---- UNARY OPERATIONS                                                   
  210     PDATA(IPARM) = PDATA(IOP2)                                     
        GO TO 500                                                        
  220     PDATA(IPARM) = -PDATA(IOP2)                                    
        GO TO 500                                                        
  230     IF (PDATA(IOP2) .LT. 0.0) THEN                                 
            WRITE (IECHO,920)                                            
            WRITE (ISCRN,920)                                            
            CALL PARPRT(IPARM)                                           
            NWARN = NWARN + 1                                            
            PDATA(IPARM) = 0.0                                           
          ELSE                                                           
            PDATA(IPARM) = SQRT(PDATA(IOP2))                             
          ENDIF                                                          
        GO TO 500                                                        
  240     IF (PDATA(IOP2) .LE. 0.0) THEN                                 
            WRITE (IECHO,930)                                            
            WRITE (ISCRN,930)                                            
            CALL PARPRT(IPARM)                                           
            NWARN = NWARN + 1                                            
            PDATA(IPARM) = 0.0                                           
          ELSE                                                           
            PDATA(IPARM) = LOG(PDATA(IOP2))                              
          ENDIF                                                          
        GO TO 500                                                        
  250     PDATA(IPARM) = EXP(PDATA(IOP2))                                
        GO TO 500                                                        
  260     PDATA(IPARM) = SIN(PDATA(IOP2))                                
        GO TO 500                                                        
  270     PDATA(IPARM) = COS(PDATA(IOP2))                                
        GO TO 500
C---- Here's the patch for ATAN. -PT
 280      PDATA(IPARM) = ATAN(PDATA(IOP2))
        GO TO 500
C----	ASIN
 290		PDATA(IPARM) = ASIN(PDATA(IOP2))
	  GOTO 500
C----	ABS
 300	    PDATA(IPARM) = ABS(PDATA(IOP2))
C---- TAN
        GOTO 500
 310      PDATA(IPARM) = TAN(PDATA(IOP2))
	  GOTO 500
C---- NEXT LIST MEMBER                                                   
  500   IPARM = IPNEXT(IPARM)                                            
        IF (IPARM .NE. 0) GO TO 100                                      
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('0** WARNING ** DIVISION BY ZERO ATTEMPTED --- ',           
     +       'RESULT SET TO ZERO')                                       
  920 FORMAT('0** WARNING ** SQUARE ROOT OF A NUMBER < 0.0 --- ',        
     +       'RESULT SET TO ZERO')                                       
  930 FORMAT('0** WARNING ** LOGARITHM OF A NUMBER <= 0.0 --- ',         
     +       'RESULT SET TO ZERO')                                       
C----------------------------------------------------------------------- 
      END                                                                
