c
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c+                                                             +
c+    LIAR:   D i m e n s i o n a l   d e f i n i t i o n s.   +
c+                                                             +
c+    Note:   -  This file must be included FIRST!             +
c+            -  Due to history there are multiple parameters  +
c+               for some quantities. Change ALL!              +
c+                                                             +
c+                                                             +
c+                                                             +
c+                                         RA, SLAC 3/12/97    +
c+       +
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c	MOD:
C                 23-Jan-2005, Linda Hendrickson (LJH):
C                   Expand STRUC_TYPE_MAX from 10 to 100.
c                07-Dec-2001, Linda Hendrickson (LJH):
C                   Increase STRUC_TYPE_MAX  from 3 to 10 .  
c                06-Sep-2001, Linda Hendrickson (LJH):
C                       Increase nslice_max from 30 to 51.
C		 20-OCT-1998, PT:
C			changed to max of 200 bunches.
c
      MODULE DEFINITIONS_MOD
      IMPLICIT NONE
      SAVE
c
c___________________________________________________________
c++  Maximum number of beamline elements
c
c        INTEGER*8    NELEM_MAX
c        PARAMETER   (NELEM_MAX = 9000)
c
c___________________________________________________________
c++  Maximum number of quadrupoles
c
c        INTEGER*8    NQUAD_MAX
c        PARAMETER   (NQUAD_MAX = 900)
c
c___________________________________________________________
c++  Maximum number of bending magnets
c
C        INTEGER*8    NBEND_MAX
C        PARAMETER   (NBEND_MAX = 20)
c
c___________________________________________________________
c++  Maximum number of BPM's
c
C        INTEGER*8    NBPM_MAX
C        PARAMETER   (NBPM_MAX = 900)
c
c___________________________________________________________
c++  Maximum number of RF-structures
c
c        INTEGER*8    NSTRUCT_MAX
c        PARAMETER   (NSTRUCT_MAX = 5000)
c
c___________________________________________________________
c++  Maximum number of pieces per RF-structure
c
        INTEGER*4    NPIECE_PER_STRUC_MAX
        INTEGER*4    NPIECE_MAX
c        PARAMETER   (NPIECE_PER_STRUC_MAX = 5)
c        PARAMETER   (NPIECE_MAX           = 5)
        PARAMETER   (NPIECE_PER_STRUC_MAX = 2)
        PARAMETER   (NPIECE_MAX           = 2)
c
c___________________________________________________________
c++  Maximum number of X-correctors
c
C        INTEGER*8    NXCOR_MAX
C        PARAMETER   (NXCOR_MAX = 900)
c
c___________________________________________________________
c++  Maximum number of Y-correctors
c
C        INTEGER*8    NYCOR_MAX
C        PARAMETER   (NYCOR_MAX = 900)
c
c___________________________________________________________
c++  Maximum number of markers
c
        INTEGER*4    NMARKER_MAX
        PARAMETER   (NMARKER_MAX = 50)
c
c___________________________________________________________
c++  Maximum number of bunches for BPM readings and markers
c++  Must agree with NBUNCH_MAX (see "beam.inc").
c
        INTEGER*4    NB_MAX
        INTEGER*4    NBUNCH_MAX
c        PARAMETER   (NB_MAX     = 180)
c        PARAMETER   (NBUNCH_MAX = 180)
c        PARAMETER   (NB_MAX     = 90)
c        PARAMETER   (NBUNCH_MAX = 90)
        PARAMETER   (NB_MAX     = 200)
        PARAMETER   (NBUNCH_MAX = 200)
c
c___________________________________________________________
c++  Maximum number of slices for markers. Must agree with 
c++  NSLICE_MAX (see "beam.inc")!
c
        INTEGER*4    NS_MAX
        INTEGER*4    NSLICE_MAX
        INTEGER*4    NSMAX
        PARAMETER   (NS_MAX     = 51)
        PARAMETER   (NSLICE_MAX = 51)
        PARAMETER   (NSMAX      = 51)
c
c___________________________________________________________
c++  Maximum number of beam ellipses per slice
c
        INTEGER*4     NMP_MAX
        PARAMETER   (NMP_MAX = 20)
c
c___________________________________________________________
c++  Maximum number of ellipses in beam 
c++  This is:     NBUNCH_MAX*NSLICE_MAX*NMP_MAX
c
C        INTEGER*8    NBEAMP_MAX
c        PARAMETER   (NBEAMP_MAX = 15270)
C        PARAMETER   (NBEAMP_MAX = 300)       
c
c___________________________________________________________
c++  Maximum number of reference trajectories
c
        INTEGER*4    MAX_NREF
        PARAMETER   (MAX_NREF = 20)
c
c___________________________________________________________
c++  Maximum number of support girders
c
        INTEGER*4    NSUPPORT_MAX
C        PARAMETER   (NSUPPORT_MAX = 3500)
        PARAMETER   (NSUPPORT_MAX = 20000)
c
c___________________________________________________________
c++  Maximum number of elements attached to a single support
c
        INTEGER*4    NATTACH_MAX
        PARAMETER   (NATTACH_MAX = 10)
c
c___________________________________________________________
c++  Maximum number of RF-BPM's per single RF-structure
c
        INTEGER*4    NRFBPM_MAX
        PARAMETER   (NRFBPM_MAX = 2)
c
c___________________________________________________________
c++  Maximum number of different structure types
c
        INTEGER*4    STRUC_TYPE_MAX
        PARAMETER   (STRUC_TYPE_MAX = 100)
c
c___________________________________________________________
c++  Maximum number of different LRWF frequency errors
c
c    The error wakefields will be allocated dynamically in
c    SET_ERROR_WF_TRANSV_LR subroutine. We don't need this max
c    value any more
c
        INTEGER*4    WF_ERROR_MAX
        PARAMETER   (WF_ERROR_MAX = 50)
c
c___________________________________________________________
c++  Maximum number of position feedback loops
c
        INTEGER*4    NFDBK_MAX
        PARAMETER   (NFDBK_MAX = 20)
c
c___________________________________________________________
c++  Maximum number of emittance bumps
c
        INTEGER*4    NEBUMPMAX
        PARAMETER   (NEBUMPMAX = 5)
c
c___________________________________________________________
c++  Maximum number of multi-knobs
c
        INTEGER*4    NMULTI_MAX
        PARAMETER   (NMULTI_MAX = 50)
c
c___________________________________________________________
c++  Maximum number of elements per multi-knob
c
        INTEGER*4    NMULTI_ELEMENT_MAX
        PARAMETER   (NMULTI_ELEMENT_MAX = 50)
c
c___________________________________________________________
c++  Maximum number of different seeds in logbook file
c
        INTEGER*4    NSEED_MAX
        PARAMETER   (NSEED_MAX = 1000)
c
c___________________________________________________________
c++  Maximum number of different sets in logbook
c
        INTEGER*4    NSET_MAX
        PARAMETER   (NSET_MAX = 50)
c
c___________________________________________________________
c++  Maximum number of parameters for a LIAR command
c
        INTEGER*4    NPAR_MAX
        PARAMETER   (NPAR_MAX = 30)
c
c___________________________________________________________
      END MODULE
