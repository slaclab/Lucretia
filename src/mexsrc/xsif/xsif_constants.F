      MODULE XSIF_CONSTANTS
C
C     This module contains shared constants used by the XSIF
C     parser library, and other values related to the constants.
C     It is descended from DIMAD_CONSTANTS module.
C
C     Auth: PT, 05-jan-2001
C
C     Mod:
C
C     Modules:
C
      USE XSIF_SIZE_PARS

      IMPLICIT NONE
      SAVE
C
C========1=========2=========3=========4=========5=========6=========7=C
C
C     physical  and  numerical  constants; note that IPTYP (particle
C     type) has been renamed to IPRTYP to avoid conflicting with an
C     array in DIMAD_ELEMENTS module (part of MAD input parser).

      REAL*8    PI, TWOPI            ! self-explanatory
      REAL*8    CRDEG                ! conversion radians to degrees
      REAL*8    CLIGHT               ! speed of light
      REAL*8    CMAGEN               ! convert GeV to kGm
      REAL*8    EMASS, ERAD, ECHG    ! electron mass, radius, and
                                     ! charge in GeV, m, and C, resp.

C	data and constants related to interchange of physical and
C	numerical constants between DIMAT and MAD.  The way this
C	works is that MAD_LOCATION(PI_INDEX) is the position in PDATA
C	which contains the value of PI, etc.

	INTEGER*4 MAD_LOCATION(MX_SHARECON)

	INTEGER*4 PI_INDEX, TWOPI_INDEX, DEGRAD_INDEX
	INTEGER*4 RADDEG_INDEX, E_INDEX, EMASS_INDEX
	INTEGER*4 PMASS_INDEX, CLIGHT_INDEX

	PARAMETER ( PI_INDEX = 1,
     &			TWOPI_INDEX = 2,
     &			DEGRAD_INDEX = 3,
     &			RADDEG_INDEX = 4,
     &			E_INDEX = 5,
     &			EMASS_INDEX = 6,
     &			PMASS_INDEX = 7,
     &			CLIGHT_INDEX = 8 )

C	initial values

	REAL*8 PI_INIT, DEGRAD_INIT, E_INIT, EMASS_INIT, PMASS_INIT,
     &	   CLIGHT_INIT

	PARAMETER ( PI_INIT =      3.1415926535898D0 ,
     &			DEGRAD_INIT = 57.295779513082D0  ,
     &			E_INIT =       2.7182818284590D0 ,
     &			EMASS_INIT =   0.51099906D-03    ,
     &			PMASS_INIT =   0.93827231D0      ,
     &			CLIGHT_INIT =  2.99792458D+08      )

      END MODULE XSIF_CONSTANTS