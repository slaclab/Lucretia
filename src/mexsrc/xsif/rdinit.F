      SUBROUTINE RDINIT(LDATA,LPRNT,LECHO)
C     
C     member of MAD INPUT PARSER
C
C     MOD:
C          05-jan-2001, PT:
C             call XSIF_HEADER to write info about xsif parser to
C             various locales upon I/O init.
C
C---- INITIALIZE READ PACKAGE                                            
C----------------------------------------------------------------------- 
C
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
      IDATA = LDATA                                                      
      IPRNT = LPRNT                                                      
      IECHO = LECHO                                                      
c      CALL SETUP                                                         
      ILINE = 0                                                          
      ILCOM = 0                                                          
      ICOL  = 81                                                         
      IMARK = 1                                                          
      NWARN = 0                                                          
      NFAIL = 0                                                          
      ENDFIL = .FALSE.                                                   
      KTEXT = ' '                                                        
      KLINE(81) = ';'   
c     PT, 05-jan-2001:
      CALL XSIF_HEADER( ISCRN )
      CALL XSIF_HEADER( IPRNT )
      CALL XSIF_HEADER( IECHO )
C     end new block 05-jan-2001.                                                 
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                
