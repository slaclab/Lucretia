      SUBROUTINE RDNEXT
C
C     member of MAD INPUT PARSER
C
C---- FIND NEXT NON-BLANK INPUT CHARACTER                                
C----------------------------------------------------------------------- 
C
C	MOD:
C		 08-DEC-2003, PT:
C			set WHITESPACE_SKIPPED if whitespace is skipped.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C		 07-OCT-1998, PT:
C			 allow comments to break up input sequences which
C			 exceed 1 line (ie, a line ending in '&' followed
C			 by a comment line followed by more input is okay).
C			 IMPLICIT NONE.
C
C     modules:
C
      USE XSIF_INOUT
C
C      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
	IMPLICIT NONE
      SAVE
C
	LOGICAL*4 :: CONTINUE_FLAG = .FALSE.
C
C----------------------------------------------------------------------- 
	WHITESPACE_SKIPPED = .FALSE.
    5 IF (ICOL .GT. 80) CALL RDLINE 
      IF (FATAL_READ_ERROR) GOTO 9999
   10 CONTINUE                                                           
        IF (ENDFIL) THEN
		GOTO 9999
	  ENDIF                                               
        IMARK = ICOL + 1                                                 
   20   CONTINUE                                                         
          ICOL = ICOL + 1                                                
        IF (KLINE(ICOL) .EQ. ' ') THEN
		  WHITESPACE_SKIPPED = .TRUE.
		  GO TO 20
	  ENDIF                               
      IF (KLINE(ICOL) .EQ. '&') THEN                                     
	  CONTINUE_FLAG = .TRUE.
        CALL RDLINE                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        GO TO 10                                                         
      ENDIF                                                              
      IF (ICOL .LE. 80) IMARK = ICOL                                     
      IF (KLINE(ICOL) .EQ. '!' .OR.                                      
     +    KLINE(1) .EQ. '@' .OR.                                         
     +    KLINE(1) .EQ. '*' .OR.                                         
     +    KLINE(1) .EQ.'('          ) THEN 
     			ICOL = 81
	        IF (CONTINUE_FLAG) THEN
				GOTO 5
			ENDIF
	ENDIF
	                           
9999	CONTINUE_FLAG = .FALSE.
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                
