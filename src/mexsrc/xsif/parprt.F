      SUBROUTINE PARPRT(IPARM)                                           
C
C     member of MAD INPUT PARSER
C
C---- PRINT PARAMETER OPERATION                                          
C**** ROUTINE NOT COMPLETE                                               
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
C----------------------------------------------------------------------- 
      CHARACTER*8       KOPER(20)                                        
C----------------------------------------------------------------------- 
      DATA KOPER(1)     / 'ADD     ' /                                   
      DATA KOPER(2)     / 'SUBTRACT' /                                   
      DATA KOPER(3)     / 'MULTIPLY' /                                   
      DATA KOPER(4)     / 'DIVIDE  ' /                                   
      DATA KOPER(11)    / 'EQUALS  ' /                                   
      DATA KOPER(12)    / 'NEGATE  ' /                                   
      DATA KOPER(13)    / 'SQRT    ' /                                   
      DATA KOPER(14)    / 'LOG     ' /                                   
      DATA KOPER(15)    / 'EXP     ' /                                   
      DATA KOPER(16)    / 'SIN     ' /                                   
      DATA KOPER(17)    / 'COS     ' /                                   
C----------------------------------------------------------------------- 
      IOP = IPTYP(IPARM)                                                 
      IOP1 = IPDAT(IPARM,1)                                              
      IOP2 = IPDAT(IPARM,2)                                              
      WRITE (IECHO,910) KPARM(IPARM),KOPER(IOP),KPARM(IOP1),KPARM(IOP2)  
      WRITE (ISCRN,910) KPARM(IPARM),KOPER(IOP),KPARM(IOP1),KPARM(IOP2)  
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(15X,A,' = ',A,' ( ',A,' , ',A,' )')                         
C----------------------------------------------------------------------- 
      END                                                                
