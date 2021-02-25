      SUBROUTINE xsif_header(IFILE)
C
C     member of MAD INPUT PARSER
C
C---- PRINT PAGE HEADER                                                  
C----------------------------------------------------------------------- 
C
C     MOD:
C          17-DEC-2003, PT:
C             new version number and date.
c          18-nov-2002, PT:
c             new version number and date
c          15-May-2001, PT:
C             new version date.
C          20-MAR-2001, PT:
C             move initialization of version date and version # here
C             from DIMAD_INOUT.
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
C      WRITE (IFILE,910) KTIT
      XSIF_VERSION = '2.1'
      XSIF_VERS_DATE = '01-JAN-2004'
      write (ifile,*)' '
      WRITE (IFILE,920) XSIF_VERSION
      WRITE (IFILE,930) XSIF_VERS_DATE
      WRITE (IFILE,940) KDATE,KTIME   
      WRITE (IFILE,*)'  XSIF Parser developed by NLC Department,'
      WRITE (IFILE,*)'      Stanford Linear Accelerator Center.' 
      WRITE (IFILE,*)' '                       
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('1',A80)
  920 FORMAT('   XSIF Parser Version  ',A)
  930 FORMAT('   Version Date:        ',A)
  940 FORMAT('   Run: ',A,'  ',A)         
C----------------------------------------------------------------------- 
      END                                                                
