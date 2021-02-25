c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c+                                                             +
c+    LIAR: some control parameters.                           +
c+                                                             +
c+                                         RA, SLAC 9/17/95    +
c+       +
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c     MOD:
C          17-May-2002, Linda Hendrickson (LJH):
C             Add SEED_CONSTVAL to allow setting a constant seed.
C          24-jan-2002, PT:
C             add DIMAD_SEED_ADVANCE logical, which is
C             initialized to FALSE.
C          21-JAN-2002, PT:
C             add logical DO_TRACKANA to indicate that a call to
C             TRACKANA is in the cards.
c          10-Aug-2001, Linda Hendrickson (LJH):
C             Add deflun for default output lun (instead of 6 or *).
c             This is to support matlab-liar.
C          05-jul-2001, PT:
C             add logical DIMAD_READY to indicate that
C             DIMAD mode tracking is ready to go.
c          30-apr-2001, PT:
c             add variables SEED_FILE (character string),
c                           SEED_LUNIT (INTEGER*4)
c             to support new features in RESEED_SELECT.
C             Add TRACKANA_TERSE logical to suppress some output
C             in TRACKANA.
C          17-apr-2001, PT:
C             add INTEGER*4s RTRACK_VERBOSE_START,
C                          RTRACK_VERBOSE_STOP,
C                          RTRACK_VERB_START_USR,
C                          RTRACK_VERB_STOP_USR
C             for switching on/off fully-detailed calculations
C             of BPMs and MARKers during RTRACK.
C             Add ORDER_DEFAULT default tracking order.
C          28-NOV-2000, PT:
C             add logical RESEED_SWITCH, initialized to TRUE, to
C             permit user to control whether reseed commands 
C             are disabled.
C          16-NOV-2000, PT:
C             add L_TRACKOMETER for decision whether or not
C             to display the trackometer in RTRACK
C		 23-nov-1998, PT:
C			moved XSIF error flags to DIMAD_INOUT.
C		 11-NOV-1998, PT:
C			added XSIF_PARSE_NOLINE and XSIF_PARSE_ERROR
C			error messages.
C          12-OCT-1998, PT:
C             added OUTOPT for supressing output.
C
      MODULE CONTROL_MOD

      IMPLICIT NONE
      SAVE
c
c++  File unit for output file
c
        INTEGER*4       outlun
c
c++  File unit for std output (equivalent to 6 or *)
c
        INTEGER*4  ::   deflun = 6
c
c++  Debug level for debug information
c
        INTEGER*4       debug
c
c++   output suppression flag
c
        INTEGER*4  ::   outopt = 0
c
c++   Flag to matlab-compatible code.
c
        INTEGER*4  ::   matlabopt = 0

        LOGICAL*4  ::   L_TRACKOMETER = .TRUE.

        LOGICAL*4  ::   RESEED_SWITCH = .TRUE.

      INTEGER*4 :: RTRACK_VERBOSE_START = 0
      INTEGER*4 :: RTRACK_VERBOSE_STOP = -1
      INTEGER*4 :: RTRACK_VERB_START_USR = 0
      INTEGER*4 :: RTRACK_VERB_STOP_USR = -1
      INTEGER*4 :: ORDER_DEFAULT = 2

      LOGICAL*4 :: TRACKANA_TERSE = .FALSE.

      CHARACTER*80 SEED_FILE
      INTEGER*4 :: SEED_LUNIT = 0
      INTEGER*4 :: SEED_CONSTVAL = 0

      LOGICAL*4 :: DIMAD_READY = .FALSE.
      LOGICAL*4 :: DO_TRACKANA = .FALSE.

      LOGICAL*4 :: DIMAD_ADVANCE_SEED = .FALSE.

c	error flags for XSIF_PARSE

c	  INTEGER*4  ::  XSIF_PARSE_NOLINE = -191
c	  INTEGER*4  ::  XSIF_PARSE_ERROR  = -193
c	  INTEGER*4  ::  XSIF_PARSE_NOOPEN = -195
c
c++  Common block
c
C        COMMON /CONTROL/ outlun, debug
c
      END MODULE CONTROL_MOD
