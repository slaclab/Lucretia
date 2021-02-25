      SUBROUTINE DUMP                                                           
C
C     member of MAD INPUT PARSER
C
C---- DECODE "DUMP" COMMAND                                                     
C
C-----------------------------------------------------------------------        
C
C	MOD:
C          15-DEC-2003, PT:
C             display pointers to misalignments.  Expand element names
C             to 16 characters.
C          14-may-2003, PT:
C             assorted improvements in output format, plus output of
C             the xsif open stack (opened files).
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C
C-----------------------------------------------------------------------    

      USE XSIF_INOUT
    
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE

C-----------------------------------------------------------------------        
      CALL RDFIND(';')                                                          
      IF (FATAL_READ_ERROR) GOTO 9999
      CALL DUMPEX              

9999  CONTINUE

                                                 
      RETURN                                                                    
C-----------------------------------------------------------------------        
      END                                                                       
C
      SUBROUTINE DUMPEX                                                         
C---- EXECUTE "DUMP" COMMAND
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
      USE XSIF_INTERFACES
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
      INTEGER*4 ILOOP, IPOINT1, IPOINT, IEALIGN
      CHARACTER(ENAME_LENGTH) BNAME
      TYPE (XSIF_FILETYPE), POINTER :: THIS_FILE
C-----------------------------------------------------------------------        
C---- BEAM ELEMENTS AND BEAM LINES                                              
      IF (IELEM1 .GT. 0) THEN                                                   
        WRITE (IPRNT,910)                                                       
        WRITE (IPRNT,920) (I,KELEM(I),KETYP(I),IETYP(I),                        
     +    IEDAT(I,1),IEDAT(I,2),IEDAT(I,3),IELIN(I),I=1,IELEM1)                 
      ENDIF                                                                     
C---- FORMAL ARGUMENTS                                                          
      IF (IELEM2 .LE. MAXELM) THEN                                              
        WRITE (IPRNT,930)                                                       
        WRITE (IPRNT,920) (I,KELEM(I),KETYP(I),IETYP(I),                        
     +     IEDAT(I,1),IEDAT(I,2),IEDAT(I,3),IELIN(I),I=IELEM2,MAXELM)           
      ENDIF                                                                     
C---- GLOBAL PARAMETERS                                                         
      IF (IPARM1 .GT. 0) THEN                                                   
        WRITE (IPRNT,940)                                                       
        WRITE (IPRNT,950) (I,KPARM(I),IPTYP(I),                                 
     +     IPDAT(I,1),IPDAT(I,2),IPLIN(I),PDATA(I),I=1,IPARM1)                  
      ENDIF                                                                     
C---- ELEMENT PARAMETERS                                                        
      IF (IPARM2 .LE. MAXPAR) THEN                                              
        WRITE (IPRNT,960)                                                       
        WRITE (IPRNT,950) (I,KPARM(I),IPTYP(I),                                 
     +     IPDAT(I,1),IPDAT(I,2),IPLIN(I),PDATA(I),I=IPARM2,MAXPAR)             
      ENDIF                                                                     
C---- LINK TABLE                                                                
      IF (IUSED .GT. 0) THEN                                                    
        WRITE (IPRNT,970)                                                       
        WRITE (IPRNT,980) (I,(ILDAT(I,J),J=1,6),I=1,IUSED)                      
      ENDIF 
C---- POSITION TABLE
      IF (NPOS2 .GT. 0) THEN
c        WRITE(IPRNT,990)
c        WRITE(IPRNT,995) (I,ITEM(I),I=NPOS1,NPOS2)
          WRITE(IPRNT,990)
          DO ILOOP = NPOS1,NPOS2
              IPOINT1 = ITEM(ILOOP)
	        IEALIGN = ERRPTR(ILOOP)
              IF (IPOINT1 .GT. MAXELM+MXLINE) THEN
                  IPOINT = IPOINT1 - MAXELM - MXLINE
                  BNAME = KELEM(IPOINT)
                  WRITE(IPRNT,995)ILOOP,IPOINT1,IPOINT,BNAME
              ENDIF
              IF ( (IPOINT1 .GT. MAXELM)  .AND.
     &             (IPOINT1 .LE. MAXELM+MXLINE) ) THEN
                  IPOINT = IPOINT1 - MAXELM
                  BNAME = KELEM(IPOINT)
                  WRITE(IPRNT,996)ILOOP,IPOINT1,IPOINT,BNAME
              ENDIF
              IF ( IPOINT1 .LE. MAXELM ) THEN
                  WRITE(IPRNT,997)ILOOP,IPOINT1,IEALIGN
              ENDIF
          ENDDO
      ENDIF
C---- OPEN FILE STACK
      IF (.NOT.ASSOCIATED(XSIF_OPEN_STACK_HEAD)) THEN
        WRITE(IPRNT,*)' No files presently in OPEN stack.'
      ELSE
        WRITE(IPRNT,*)'DUMP OPEN STACK: UNIT # FILENAME'
        THIS_FILE => XSIF_OPEN_STACK_HEAD
        DO
          WRITE(IPRNT,998)THIS_FILE%UNIT_NUMBER,
     &      ARR_TO_STR(THIS_FILE%FILE_NAME)
          IF (ASSOCIATED(THIS_FILE%NEXT_FILE)) THEN
            THIS_FILE => THIS_FILE%NEXT_FILE
          ELSE
            EXIT
          ENDIF
        ENDDO
      ENDIF                                                                   
      RETURN                                                                    
C-----------------------------------------------------------------------        
c  910 FORMAT('1DUMP OF ELEMENT TABLE (1)')                                      
  910 FORMAT('ELEMENT TABLE (1): KELEM KETYP IETYP IEDAT(:,1)',
     & ' IEDAT(:,2) IEDAT(:,3) IELIN')                                      
  920 FORMAT(' ',I10,5X,A16,5X,A4,5I8)                                           
C  930 FORMAT('1DUMP OF ELEMENT TABLE (2)')                                      
  930 FORMAT('ELEMENT TABLE (2): KELEM KETYP IETYP IEDAT(:,1)',
     & ' IEDAT(:,2) IEDAT(:,3) IELIN')                                      
C  940 FORMAT('1DUMP OF PARAMETER TABLE (1)')                                    
  940 FORMAT('PARAMETER TABLE (1): KPARM IPTYP IPDAT(:,1)',
     &  'IPDAT(:,2) IPLIN PDATA')                                    
  950 FORMAT(' ',I10,5X,A16,4I8,F15.6)                                           
C  960 FORMAT('1DUMP OF PARAMETER TABLE (2)')                                    
  960 FORMAT('PARAMETER TABLE (2): KPARM IPTYP IPDAT(:,1)',
     &  'IPDAT(:,2) IPLIN PDATA')                                    
  970 FORMAT('DUMP OF LINK TABLE')                                             
  980 FORMAT(' ',I10,6I8)          
  990 FORMAT('DUMP OF POSITION TABLE:  ITEM   POINTER    MESSAGE'
     &       '/ALIGNPTR')
  995 FORMAT(' ',I10,I10,I10,' END OF LINE:    ',A16)                                          
  996 FORMAT(' ',I10,I10,I10,' START OF LINE:  ',A16)
  997 FORMAT(' ',I10,I10,I10)        
  998 FORMAT(' ',I10,' ',A)                                    
C-----------------------------------------------------------------------        
      END                                                                       
