      MODULE XSIF_ELEM_PARS
C
C     This module contains various definitions related to element
C     types and their permitted keywords.  It is descended from
C     DIMAD_ELEM_PARS.
C
C     AUTH: PT, 05-JAN-2001
C
C     MOD:
C          17-DEC-2003, PT:
C             change HKICK/VKICK to HKICKER/VKICKER to match MAD
C             format.  Change SROT/YROT to SROTATIO/YROTATIO.  Add
C             MAD KICKER type (as opposed to H/VKICKER).
C          14-nov-2002, PT:
C             add BETA0 (emulate MAD initial twiss element); this
C             element IS an accepted part of the NLC coding standard,
C             so NLC_KEYW will be TRUE.  Includes ENERGY for 
C             compatibility with MAD8Acc.  Added BEAM element as well,
C             similar constraints.
C          02-MAR-2001, PT:
C             removed cross-reference stuff (it's really DIMAD stuff,
C             not used by XSIF).
C
      IMPLICIT NONE
      SAVE

C========1=========2=========3=========4=========5=========6=========7=C

C     number of element parameters associated with each MAD
C     element type

      INTEGER*4 NKEYW    ! number of different element types
      INTEGER*4 NDRFT    ! drift pars
      INTEGER*4 NBEND    ! bend pars
      INTEGER*4 NQUAD    ! quad pars
      INTEGER*4 NSEXT    ! sextupole pars
      INTEGER*4 NOCT     ! octupole pars
      INTEGER*4 NMULT    ! multipole pars
      INTEGER*4 NSOLO    ! solenoid pars
      INTEGER*4 NCVTY    ! RF cavity pars
      INTEGER*4 NLCAV    ! Linac cavity pars
      INTEGER*4 NSEPA    ! separator pars
      INTEGER*4 NROTA    ! roll parameters
      INTEGER*4 NKICK    ! kick pars
      INTEGER*4 NMON     ! monitor pars
      INTEGER*4 NCOLL    ! collimator pars
      INTEGER*4 NQUSE    ! quad-sext pars
      INTEGER*4 NGKIK    ! gkick pars
      INTEGER*4 NARBI    ! ARBITELM pars
      INTEGER*4 NTWIS    ! mtwiss pars
      INTEGER*4 NMATR    ! matrix pars
      INTEGER*4 NINST    ! instrument pars
      INTEGER*4 NBET0    ! BETA0 pars
      INTEGER*4 NBEAM    ! BEAM pars
	INTEGER*4 NKICKMAD ! MAD Kicker pars

      PARAMETER         (NKEYW = 39)                                     
      PARAMETER         (NDRFT =  1)                                     
      PARAMETER         (NBEND = 13)                                     
      PARAMETER         (NQUAD =  4)                                     
      PARAMETER         (NSEXT =  4)                                     
      PARAMETER         (NOCT  =  4)                                     
      PARAMETER         (NMULT = 50)                                     
      PARAMETER         (NSOLO =  5)                                     
      PARAMETER         (NCVTY = 12)                                     
      PARAMETER         (NLCAV = 13)                                     
      PARAMETER         (NSEPA =  3)                                     
      PARAMETER         (NROTA =  1)                                     
      PARAMETER         (NKICK =  3)                                     
      PARAMETER         (NMON  =  5)                                     
      PARAMETER         (NCOLL =  3)                                     
      PARAMETER         (NQUSE =  5)                                     
      PARAMETER         (NGKIK = 11)                                     
      PARAMETER         (NARBI = 21)                                     
      PARAMETER         (NTWIS =  7)                                     
      PARAMETER         (NMATR =163)
      PARAMETER         (NINST =  1)
      PARAMETER         (NBET0 = 27)
      PARAMETER         (NBEAM = 18)
	PARAMETER         (NKICKMAD = 4)

C===========================================================C

C     character arrays containing the element keywords

      CHARACTER*8       DKEYW(NKEYW)                                     
      CHARACTER*8       DDRFT(NDRFT)                                     
      CHARACTER*8       DBEND(NBEND)                                     
      CHARACTER*8       DQUAD(NQUAD)                                     
      CHARACTER*8       DSEXT(NSEXT)                                     
      CHARACTER*8       DOCT (NOCT )                                     
      CHARACTER*8       DMULT(NMULT)      
	CHARACTER*8		  DDIMU(NMULT)                               
      CHARACTER*8       DSOLO(NSOLO)                                     
      CHARACTER*8       DCVTY(NCVTY)                                     
      CHARACTER*8       DLCAV(NLCAV)                                     
      CHARACTER*8       DSEPA(NSEPA)                                     
      CHARACTER*8       DROTA(NROTA)                                     
      CHARACTER*8       DKICK(NKICK)                                     
      CHARACTER*8       DMON (NMON)                                      
      CHARACTER*8       DCOLL(NCOLL)                                     
      CHARACTER*8       DQUSE(NQUSE)                                     
      CHARACTER*8       DGKIK(NGKIK)                                     
      CHARACTER*8       DARBI(NARBI)                                     
      CHARACTER*8       DTWIS(NTWIS)                                     
      CHARACTER*8       DMATR(NMATR)
      CHARACTER*8       DINST(NINST)
      CHARACTER*8       DBET0(NBET0)
      CHARACTER*8       DBEAM(NBEAM)
	CHARACTER*8       DKICKMAD(NKICKMAD)

C     logical arrays indicating whether keywords are or are not
C     NLC-standard

      LOGICAL*4       NLC_KEYW(NKEYW)

      DATA DKEYW                                                         
     +   / 'DRIFT   ','RBEND   ','SBEND   ','WIGGLER ','QUADRUPO',       
     +     'SEXTUPOL','OCTUPOLE','MULTIPOL','SOLENOID','RFCAVITY',       
     +     'SEPARATO','ROLL    ','ZROT    ','HKICKER ','VKICKER ',       
     +     'HMONITOR','VMONITOR','MONITOR ','MARKER  ','ECOLLIMA',       
     +     'RCOLLIMA','QUADSEXT','GKICK   ','ARBITELM','MTWISS  ',       
     +     'MATRIX  ','LCAVITY ','INSTRUME','BLMONITO','PROFILE ',
     +	 'WIRE    ','SLMONITO','IMONITOR','DIMULTIP','YROTATIO',
     +	 'SROTATIO','BETA0   ','BEAM    ','KICKER  '/                   
     
      DATA NLC_KEYW
     &    / .TRUE.  , .FALSE. , .TRUE.  , .FALSE. , .TRUE.  ,
     &      .TRUE.  , .TRUE.  , .TRUE.  , .TRUE.  , .TRUE.  ,
     &      .FALSE. , .FALSE. , .FALSE. , .TRUE.  , .TRUE.  ,
     &      .TRUE.  , .TRUE.  , .TRUE.  , .TRUE.  , .TRUE.  ,
     &      .TRUE.  , .FALSE. , .FALSE. , .FALSE. , .FALSE. ,
     &      .FALSE. , .TRUE.  , .TRUE.  , .TRUE.  , .TRUE.  ,
     &      .TRUE.  , .TRUE.  , .TRUE.  , .FALSE. , .TRUE.  ,
     &      .TRUE.  , .TRUE.  , .TRUE.  , .FALSE. /                  

C     parameters for MAD element-type number

      INTEGER*4 MAD_DRIFT,    MAD_RBEND,     MAD_SBEND, 
     &          MAD_WIGG,     MAD_QUAD,      MAD_SEXT,
     &          MAD_OCTU,     MAD_MULTI,     MAD_SOLN,
     &          MAD_RFCAV,    MAD_SEPA,      MAD_ROLL,
     &          MAD_ZROT,     MAD_HKICK,     MAD_VKICK,
     &          MAD_HMON,     MAD_VMON,      MAD_MONI,
     &          MAD_MARK,     MAD_ECOLL,     MAD_RCOLL,
     &          MAD_QUSE,     MAD_GKICK,     MAD_ARBIT,
     &          MAD_MTWIS,    MAD_MATR,      MAD_LCAV,
     &          MAD_INST,     MAD_BLMO,      MAD_PROF,
     &		  MAD_WIRE,     MAD_SLMO,      MAD_IMON,
     &		  MAD_DIMU,     MAD_YROT,      MAD_SROT,
     &          MAD_BET0,     MAD_BEAM,      MAD_KICKMAD

      PARAMETER ( MAD_DRIFT = 1 ,  MAD_RBEND = 2 , MAD_SBEND = 3 ,
     &            MAD_WIGG  = 4 ,  MAD_QUAD  = 5 , MAD_SEXT  = 6 ,
     &            MAD_OCTU  = 7 ,  MAD_MULTI = 8 , MAD_SOLN  = 9 ,
     &            MAD_RFCAV = 10,  MAD_SEPA  = 11, MAD_ROLL  = 12,
     &            MAD_ZROT  = 13,  MAD_HKICK = 14, MAD_VKICK = 15,
     &            MAD_HMON  = 16,  MAD_VMON  = 17, MAD_MONI  = 18, 
     &            MAD_MARK  = 19,  MAD_ECOLL = 20, MAD_RCOLL = 21,
     &            MAD_QUSE  = 22,  MAD_GKICK = 23, MAD_ARBIT = 24,
     &            MAD_MTWIS = 25,  MAD_MATR  = 26, MAD_LCAV =  27,
     &            MAD_INST =  28,  MAD_BLMO  = 29, MAD_PROF =  30,
     &			MAD_WIRE =  31,  MAD_SLMO  = 32, MAD_IMON =  33,
     &			MAD_DIMU =  34,  MAD_YROT  = 35, MAD_SROT =  36,
     &            MAD_BET0 =  37,  MAD_BEAM  = 38, MAD_KICKMAD = 39 )

      DATA DDRFT( 1)                                                     
     +   / 'L       ' /  
      DATA DBEND                                                         
     +   / 'L       ','ANGLE   ','K1      ','E1      ','E2      ',       
     +     'TILT    ','K2      ','H1      ','H2      ','HGAP    ',       
     +     'FINT    ','HGAPX   ','FINTX   '/
      DATA DQUAD                                                         
     +   / 'L       ','K1      ','TILT    ','APERTURE' / 
      DATA DSEXT                                                         
     +   / 'L       ','K2      ','TILT    ','APERTURE' /  
      DATA DOCT                                                          
     +   / 'L       ','K3      ','TILT    ','APERTURE' / 
      DATA DDIMU                                                         
     +   / 'L       ','K0      ','T0      ','K1      ','T1      ',       
     +     'K2      ','T2      ','K3      ','T3      ','K4      ',       
     +     'T4      ','K5      ','T5      ','K6      ','T6      ',       
     +     'K7      ','T7      ','K8      ','T8      ','K9      ',       
     +     'T9      ','K10     ','T10     ','K11     ','T11     ',       
     +     'K12     ','T12     ','K13     ','T13     ','K14     ',       
     +     'T14     ','K15     ','T15     ','K16     ','T16     ',       
     +     'K17     ','T17     ','K18     ','T18     ','K19     ',       
     +     'T19     ','K20     ','T20     ','SCALEFAC','APERTURE',       
     +     'KZL     ','KRL     ','THETA   ','Z       ','TILT    ' /                                                  
      DATA DMULT                                                         
     +   / 'L       ','K0L     ','T0      ','K1L     ','T1      ',       
     +     'K2L     ','T2      ','K3L     ','T3      ','K4L     ',       
     +     'T4      ','K5L     ','T5      ','K6L     ','T6      ',       
     +     'K7L     ','T7      ','K8L     ','T8      ','K9L     ',       
     +     'T9      ','K10L    ','T10     ','K11L    ','T11     ',       
     +     'K12L    ','T12     ','K13L    ','T13     ','K14L    ',       
     +     'T14     ','K15L    ','T15     ','K16L    ','T16     ',       
     +     'K17L    ','T17     ','K18L    ','T18     ','K19L    ',       
     +     'T19     ','K20L    ','T20     ','SCALEFAC','APERTURE',       
     +     'KZL     ','KRL     ','THETA   ','Z       ','TILT    ' /                                                  
      DATA DSOLO                                                         
     +   / 'L       ','KS      ','K1      ','TILT    ','APERTURE' /  
      DATA DCVTY                                                         
     +   / 'L       ','VOLT    ','LAG     ','HARMON  ','ENERGY  ',
     +     'LFILE   ','TFILE   ','ELOSS   ','NBIN    ','BINMAX  ',
     +	 'FREQ    ','APERTURE' /  
      DATA DLCAV                                                         
     +   / 'L       ','E0      ','DELTAE  ','PHI0    ','FREQ    ',       
     +     'KICKCOEF','T       ','LFILE   ','TFILE   ','ELOSS   ',
     +	 'NBIN    ','BINMAX  ','APERTURE' / 
      DATA DSEPA                                                         
     +   / 'L       ','E       ','TILT    ' /                            
      DATA DROTA                                                         
     +   / 'ANGLE   ' /                                                  
      DATA DKICK                                                         
     +   / 'L       ','KICK    ','TILT    ' /                            
      DATA DMON                                                          
     +   / 'L       ','XSERR   ','YSERR   ','XRERR   ','YRERR   ' /      
      DATA DCOLL                                                         
     +   / 'L       ','XSIZE   ','YSIZE   ' /                            
      DATA DQUSE                                                         
     +   / 'L       ','K1      ','K2      ','TILT    ','APERTURE' / 
      DATA DGKIK                                                         
     +   / 'L       ','DX      ','DXP     ','DY      ','DYP     ',       
     +     'DL      ','DP      ','ANGLE   ','DZ      ','V       ',       
     +     'T       ' /                                                  
      DATA DARBI                                                         
     +   / 'L       ','P1      ','P2      ','P3      ','P4      ',       
     +     'P5      ','P6      ','P7      ','P8      ','P9      ',       
     +     'P10     ','P11     ','P12     ','P13     ','P14     ',       
     +     'P15     ','P16     ','P17     ','P18     ','P19     ',       
     +     'P20     ' /                                                  
      DATA DTWIS                                                         
     +   / 'L       ','MUX     ','BETAX   ','ALPHAX  ','MUY     ',       
     +     'BETAY   ','ALPHAY  ' /                                       
      DATA DMATR /                                                       
     +'L   ','R11','R12','R13','R14','R15','R16','T111','T112',          
     +'T113','T114','T115','T116','T122','T123','T124','T125','T126',    
     +'T133','T134','T135','T136','T144','T145','T146','T155','T156',    
     +'T166','R21','R22','R23','R24','R25','R26','T211','T212',          
     +'T213','T214','T215','T216','T222','T223','T224','T225','T226',    
     +'T233','T234','T235','T236','T244','T245','T246','T255','T256',    
     +'T266','R31','R32','R33','R34','R35','R36','T311','T312',          
     +'T313','T314','T315','T316','T322','T323','T324','T325','T326',    
     +'T333','T334','T335','T336',                                       
     +'T344','T345','T346','T355','T356','T366','R41','R42','R43',       
     +'R44','R45','R46','T411','T412','T413','T414','T415','T416',       
     +'T422','T423','T424','T425','T426','T433','T434','T435','T436',    
     +'T444','T445','T446','T455','T456','T466','R51','R52','R53',       
     +'R54','R55','R56','T511','T512','T513','T514','T515','T516',       
     +'T522','T523','T524','T525','T526','T533','T534','T535','T536',    
     +'T544','T545','T546','T555','T556','T566','R61','R62','R63',       
     +'R64','R65','R66','T611','T612','T613','T614','T615','T616',       
     +'T622','T623','T624','T625','T626','T633','T634','T635','T636',    
     +'T644','T645','T646','T655','T656','T666' /                        
      DATA DINST /
     &'L       ' /
      DATA DBET0 /
     &'BETX    ','ALFX    ','MUX     ',
     &'BETY    ','ALFY    ','MUY     ',
     &'DX      ','DPX     ','DY      ','DPY     ',
     &'X       ','PX      ','Y       ','PY      ','T       ','PT      ',
     &'WX      ','PHIX    ','DMUX    ',
     &'WY      ','PHIY    ','DMUY    ',
     &'DDX     ','DDY     ','DDPX    ','DDPY    ',
     &'ENERGY  ' /
      DATA DBEAM /
     &'PARTICLE','MASS    ','CHARGE  ',
     &'ENERGY  ','PC      ','GAMMA   ',
     &'EX      ','EXN     ','EY      ','EYN     ',
     &'ET      ','SIGT    ','SIGE    ',
     &'KBUNCH  ','NPART   ','BCURRENT',
     &'BUNCHED ','RADIATE ' /
	DATA DKICKMAD /
     &'L       ','HKICK   ','VKICK   ','TILT    ' /

C     array for NLC-std. or nonstd. parameters of standard keywords.
C     This is initialized to FALSE here, and appropriate TRUE values
C     are set in subroutine CLEAR.

      LOGICAL*4, DIMENSION(NKEYW,NBEND) :: NLC_PARAM = .FALSE.

C     constants related to BEAM parameters that come in as non-numeric:

      REAL*8 MAD_POSI, MAD_ELEC, MAD_PROTON, MAD_PBAR
      PARAMETER( MAD_POSI = 1., MAD_ELEC = 2., MAD_PROTON = 3.,
     &           MAD_PBAR = 4. )

C     now put them into appropriate arrays

      INTEGER*4 NMADPART
      PARAMETER ( NMADPART = 4 )

      CHARACTER*8 DMADPART( NMADPART )
      DATA DMADPART /
     &'POSITRON','ELECTRON','PROTON  ','ANTI-PRO' /

C     real values for TRUE and FALSE

      REAL*8 MAD_RTRUE, MAD_RFALSE
      PARAMETER( MAD_RTRUE = 1., MAD_RFALSE = 0.)

      END MODULE XSIF_ELEM_PARS