      SUBROUTINE RDNUMB(VALUE,FLAG)
C
C     member of MAD INPUT PARSER
C
C---- DECODE A REAL NUMBER                                               
C----------------------------------------------------------------------- 
C
C	MOD:
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL           FLAG                                             
C----------------------------------------------------------------------- 
      LOGICAL           DIG,PNT                                          
C----------------------------------------------------------------------- 
      PARAMETER         (ZERO =  0.0D0)                                  
      PARAMETER         (ONE  =  1.0D0)                                  
      PARAMETER         (TEN  = 10.0D0)                                  
C----------------------------------------------------------------------- 
      FLAG = .FALSE.                                                     
      VALUE = ZERO                                                       
C---- ANY NUMERIC CHARACTER?                                             
      IF (INDEX('0123456789+-.',KLINE(ICOL)) .NE. 0) THEN                
        VAL = ZERO                                                       
        SIG = ONE                                                        
        IVE = 0                                                          
        ISE = 1                                                          
        NPL = 0                                                          
        DIG = .FALSE.                                                    
        PNT = .FALSE.                                                    
C---- SIGN?                                                              
        IF (KLINE(ICOL) .EQ. '+') THEN                                   
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
        ELSE IF (KLINE(ICOL) .EQ. '-') THEN                              
          SIG = -ONE                                                     
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
        ENDIF                                                            
C---- DIGIT OR DECIMAL POINT?                                            
   10   IDIG = INDEX('0123456789',KLINE(ICOL)) - 1                       
        IF (IDIG .GE. 0) THEN                                            
          VAL = TEN * VAL + FLOAT(IDIG)                                  
          DIG = .TRUE.                                                   
          IF (PNT) NPL = NPL + 1                                         
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          GO TO 10                                                       
        ELSE IF (KLINE(ICOL) .EQ. '.') THEN                              
          IF (PNT) FLAG = .TRUE.                                         
          PNT = .TRUE.                                                   
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          GO TO 10                                                       
        ENDIF                                                            
        FLAG = FLAG .OR. (.NOT. DIG)                                     
C---- EXPONENT?                                                          
        IF (INDEX('DE',KLINE(ICOL)) .NE. 0) THEN                         
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          DIG = .FALSE.                                                  
          IF (KLINE(ICOL) .EQ. '+') THEN                                 
            CALL RDNEXT                                                  
            IF (FATAL_READ_ERROR) GOTO 9999
          ELSE IF (KLINE(ICOL) .EQ. '-') THEN                            
            ISE = -1                                                     
            CALL RDNEXT                                                  
            IF (FATAL_READ_ERROR) GOTO 9999
          ENDIF                                                          
   20     IDIG = INDEX('0123456789',KLINE(ICOL)) - 1                     
          IF (IDIG .GE. 0) THEN                                          
            IVE = 10 * IVE + IDIG                                        
            DIG = .TRUE.                                                 
            CALL RDNEXT                                                  
            IF (FATAL_READ_ERROR) GOTO 9999
            GO TO 20                                                     
          ENDIF                                                          
          FLAG = FLAG .OR. (.NOT. DIG)                                   
   30     IF (INDEX('0123456789.DE',KLINE(ICOL)) .NE. 0) THEN            
            CALL RDSKIP('0123456789.DE')                                 
            IF (FATAL_READ_ERROR) GOTO 9999
            FLAG = .TRUE.                                                
          ENDIF                                                          
        ENDIF                                                            
C---- RETURN VALUE                                                       
        IF (FLAG) THEN                                                   
          CALL RDFAIL                                                    
          WRITE (IECHO,910)                                              
          WRITE (ISCRN,910)                                              
        ELSE                                                             
          VALUE = SIG * VAL * TEN ** (ISE * IVE - NPL)                   
        ENDIF                                                            
      ELSE                                                               
        CALL RDFAIL                                                      
        WRITE (IECHO,920)                                                
        WRITE (ISCRN,920)                                                
        FLAG = .TRUE.                                                    
      ENDIF       

9999  CONTINUE

                                                       
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** INCORRECT REAL VALUE'/' ')                  
  920 FORMAT(' *** ERROR *** REAL VALUE EXPECTED'/' ')                   
C----------------------------------------------------------------------- 
      END                                                                
