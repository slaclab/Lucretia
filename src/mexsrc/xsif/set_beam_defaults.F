      SUBROUTINE SET_BEAM_DEFAULTS( IEP1, IEP2 )
C
C     sets the MAD8 default values for parameters of the BEAM
C     statement.
C
C     AUTH:  PT, 15-nov-2002
C
C     MOD:
C
C========1=========2=========3=========4=========5=========6=========7=C

      USE XSIF_ELEMENTS
      USE XSIF_ELEM_PARS
      USE XSIF_CONSTANTS

      IMPLICIT NONE
      SAVE

      INTEGER*4 IEP1, IEP2

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C             C  O  D  E                                               C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

      PDATA(IEP1)    = MAD_POSI
      PDATA(IEP1+1)  = 0.
      PDATA(IEP1+2)  = 0.
      PDATA(IEP1+3)  = 1.0
      PDATA(IEP1+4)  = 0. 
      PDATA(IEP1+5)  = 0.
      PDATA(IEP1+6)  = 1.0
      PDATA(IEP1+7)  = 0.
      PDATA(IEP1+8)  = 1.0
      PDATA(IEP1+9)  = 0.
      PDATA(IEP1+10) = 1.0
      PDATA(IEP1+11) = 0.
      PDATA(IEP1+12) = 0.
      PDATA(IEP1+13) = 1.0
      PDATA(IEP1+14) = 0.
      PDATA(IEP1+15) = 0.
      PDATA(IEP1+16) = MAD_RFALSE
      PDATA(IEP1+17) = MAD_RFALSE

      RETURN
      END