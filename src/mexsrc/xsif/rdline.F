      SUBROUTINE RDLINE
C
C     member of MAD INPUT PARSER
C
C---- READ INPUT LINE AND PRINT ECHO                                     
C----------------------------------------------------------------------- 
C
C	MOD:
C            31-jan-2001, PT:
C               fix bugs related to read-error status.
C		 19-oct-1998, PT:
C	        add use of KTEXT_ORIG and KLINE_ORIG which preserve
C			original input line capitalization.
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
C----------------------------------------------------------------------- 
      IF (.NOT. ENDFIL) THEN                                             
        READ (IDATA,910,IOSTAT=ISTAT) KTEXT
	  KTEXT_ORIG = KTEXT                              
        DO 1 I=1,80                                                      
          IND = INDEX(LOTOUP, KLINE(I))                                  
          IF(IND.NE.0) KLINE(I) = UPTOLO(IND)                            
 1      CONTINUE                                                         
        ILINE = ILINE + 1                                                
        IMARK = 1                                                        
        ICOL = 0                                                         
C---- READ ERROR?                                                        
        IF (ISTAT .GT. 0) THEN                                           
C     PT, 31-jan-2001
c          WRITE (IECHO,920) ILINE                                        
c          WRITE (ISCRN,920) ILINE                                        
          WRITE (IECHO,920) idata, ILINE                                        
          WRITE (ISCRN,920) idata, ILINE 
          write (iecho, 921) istat
          write (iscrn, 921) istat
          NFAIL = NFAIL + 1                                              
c          CALL PLEND                                                     
          CALL RDEND(102)                                                
C---- END OF FILE?                                                       
        ELSE IF (ISTAT .LT. 0) THEN                                      
          ENDFIL = .TRUE.                                                
          KTEXT = '!!! END OF FILE !!!'                                  
          ICOL = 1                                                       
C---- READ WAS OK                                                        
        ELSE IF (MOD(ILINE,5) .EQ. 0) THEN                               
          IF (NOECHO.EQ.0) WRITE (IECHO,930) ILINE,KTEXT                 
          IF (NOECHO.EQ.0) WRITE (ISCRN,930) ILINE,KTEXT                 
          IF (NOECHO.EQ.0) WRITE (IPRNT,930) ILINE,KTEXT                 
        ELSE                                                             
          IF (NOECHO.EQ.0) WRITE (IECHO,940) KTEXT                       
          IF (NOECHO.EQ.0) WRITE (ISCRN,940) KTEXT                       
          IF (NOECHO.EQ.0) WRITE (IPRNT,940) KTEXT                       
        ENDIF                                                            
        KLINE(81) = ';'                                                  
      ENDIF                                                              
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(A80)                                                        
  920 FORMAT('0*** ERROR *** READ ERROR ON LOGICAL UNIT ',I2,            
     +       ', LINE ',I4,' --- EXECUTION TERMINATED') 
  921 FORMAT('               ERROR STATUS CODE WAS ',I6)
  930 FORMAT(' ',I9,5X,A80)                                              
  940 FORMAT(15X,A80)                                                    
C----------------------------------------------------------------------- 
      END                                                                
