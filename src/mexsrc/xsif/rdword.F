      SUBROUTINE RDWORD(KWORD,LWORD)
C
C     member of MAD INPUT PARSER
C
C---- READ AN IDENTIFIER OR KEYWORD OF UP TO 8 CHARACTERS                
C----------------------------------------------------------------------- 
C
C	MOD:
C          15-DEC-2003, PT:
C             expand potential word size to 16 characters.
C		 08-dec-2003, PT:
C			stop reading if RDNEXT sets WHITESPACE_SKIPPED (ie,
C			allow white space to separate words in XSIF).
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C		 19-oct-1998, PT:
C			allow ._$ in word.
C
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      CHARACTER*(*)       KWORD 
      INTEGER*4 LKWORD                                            
C----------------------------------------------------------------------- 
C
C     determine the length of the character variable passed to the
C     function
C
      LKWORD = LEN(KWORD)
      KWORD =  REPEAT(' ',LKWORD)                                               
      LWORD = 0                                                          
      IF (INDEX('ABCDEFGHIJKLMNOPQRSTUVWXYZ',                            
     +          KLINE(ICOL)) .NE. 0) THEN                                
   10   CONTINUE                                                         
          IF (LWORD .LT. LKWORD) THEN                                         
            LWORD = LWORD + 1                                            
            KWORD(LWORD:LWORD) = KLINE(ICOL)                             
          ENDIF                                                          
          CALL RDNEXT  
		IF (WHITESPACE_SKIPPED) GOTO 9999                                                  
          IF (FATAL_READ_ERROR) GOTO 9999
        IF (INDEX('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$._''',              
     +            KLINE(ICOL)) .NE. 0) GO TO 10                          
      ENDIF   

9999  CONTINUE

                                                           
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                
