/* Lucretia Dictionary -- home of data which is needed for figuring
   out what parameters are needed for any particular element on any
	particular operation */

/* AUTH:  PT, 16-dec-2004 */
/* MOD:
     23-Apr-2014, GW:
         Support for LSC flags
     12-Feb-2012, GW:
         Support for CSR and Split flags
		 21-May-2007, PT:
			   support for XYCOR (combined function horizontal-
			   vertical corrector dipole).
         10-mar-2006, PT:
			   support for solenoids.
			10-feb-2006, PT:
			  dictionary for COORD coordinate-change element.
			09-dec-2005, PT:
			   move enum for SR flag settings to LucretiaCommon.h.
			06-oct-2005, PT:
				create enum for SR flag settings; allow TCAVs to have
				SR flag set; add Lrad for correctors.
         18-oct-2005, PT:
			   dictionary entry for BPM scale factor.
		   30-sep-2005, PT:
			   dictionary for transverse deflecting cavity (TCAV).
			29-sep-2005, PT:
			   Additional support for devices with multiple power supplies.
			21-sep-2005, PT:
			   sector bend PS and dB can be either scalar or 1 x 2 vectors.

  */

#define LUCRETIA_DICTIONARY
   #ifndef LUCRETIA_COMMON
#include "LucretiaCommon.h"
#endif

/* Version string */

char LucretiaDictionaryVers[] = "LucretiaDictionary Version = 23-Apr-2014" ;

/* useful constants */

#define Required  1
#define Optional  0
#define Ignored  -1

/* define indexing of requirement array */

const int RmatPars   = 0 ;
const int TrackPars  = 1 ;
const int VerifyPars = 2 ;

/* define indices of the various tables */

#define ElementTable 0 
#define PSTable 1 
#define GirderTable 2 
#define KlystronTable 3 
#define TLRTable 4
#define TLRErrTable 5

/* max number of power supplies any device can have */

#define MaxPSPerDevice 2 
#define MaxGirderPerDevice 1
#define MaxKlysPerDevice 1

/*=====================================================================*/

/* tracking flags information */

   #define NUM_TRACK_FLAGS 22

	enum TrackFlagIndex{
		SynRad, Aper, GetBPMData, GetBPMBeamPars, MultiBunch,
	   GetInstData, SRWF_T, SRWF_L, LRWF_T, LRWF_ERR, 
	   ZMotion, LorentzDelay, GetSBPMData, CSR, CSR_SmoothFactor, Split, CSR_DriftSplit,
     LSC, LSC_cutoffFreq, LSC_smoothFactor, LSC_npow2Bins, LSC_storeData
	} ;

	const char* TrackFlagNames[] = {	
		"SynRad","Aper","GetBPMData","GetBPMBeamPars","MultiBunch",
		"GetInstData","SRWF_T", "SRWF_Z","LRWF_T","LRWF_ERR",
		"ZMotion","LorentzDelay","GetSBPMData", "CSR", "CSR_SmoothFactor", "Split", "CSR_DriftSplit",
    "LSC", "LSC_cutoffFreq", "LSC_smoothFactor", "LSC_npow2Bins", "LSC_storeData"
	} ;

	int TrackFlagSet[NUM_TRACK_FLAGS] ;

	int TrackFlagMinValue[NUM_TRACK_FLAGS] = {
		SR_None, 0, 0, 0, 0, 
		      0, 0, 0, 0, 0,
		      0, 0, 0, -1, 0, 0, 0,
          0, 0, 0, 1, 0
	} ;

	int TrackFlagMaxValue[NUM_TRACK_FLAGS] = {
		SR_HB, 1, 1, 1, 1, 
		    1, 1, 1, 1, 1,
		    1, 1, 1, 1000000, 1000000, 1000, 1000,
        1, 2, 1000000000, 20, 1
	} ;


/*=====================================================================*/

/* Initialized structures for defining element parameters */

/* Note:  Each dictionary is paired with an enumerated data type which
   shows the order of the parameters in the dictionary.  If a dictionary
	is changed, the changer must be careful that the order of the
	corresponding enumerated data type is still consistent with the
	dictionary! */

/* TMap */

#define nTMapPar 13

enum TMapOrdinal {
	TMapS, TMapP, TmapL, TMapOffset, TMapR, TMapT, TMapU, TMapV, TMapW, TMapTinds, TMapUinds, TMapVinds, TMapWinds } ;

struct LucretiaParameter TMapPar[nTMapPar] = {
  {"S",           {Required,Ignored, Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL              },
  {"P",           {Required,Optional,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL              },
  {"L",           {Required,Required,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL              },
  {"Offset",      {Required,Required,Optional},{Required,Ignored, Ignored}, 6,6,0,0,NULL             },
  {"R",           {Required,Required,Optional},{Required,Ignored, Ignored}, 36,36,0,0,NULL           },
  {"T",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,216,0,0,NULL            },
  {"U",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,1296,0,0,NULL           },
  {"V",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,7776,0,0,NULL           },
  {"W",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,46656,0,0,NULL          },
  {"Tinds",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,216,0,0,NULL        },
  {"Uinds",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,1296,0,0,NULL       },
  {"Vinds",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,7776,0,0,NULL       },
  {"Winds",           {Optional,Optional,Optional},{Ignored,Ignored, Ignored}, 0,46656,0,0,NULL      }
} ;

int TMapTrackFlag[NUM_TRACK_FLAGS] = {
	0, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;  
  
/* Drift space */

#define nDrifPar 3

enum DriftOrdinal {
	DriftS, DriftP, DriftL } ;

struct LucretiaParameter DrifPar[nDrifPar] = {
  {"S",      {Required,Ignored, Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
  {"P",      {Required,Optional,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
  {"L",      {Required,Required,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    }
} ;

int DrifTrackFlag[NUM_TRACK_FLAGS] = {
	0, 0, 0, 0, 0,
   0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* Quadrupole, also used for sextupole and octupole. 
   Note that any changes made to the Solenoid pars,\ need
   to be made also in quad parameters. */

#define nQuadPar 10

enum QuadOrdinal {
	QuadS, QuadP, QuadL, QuadB, QuadPS, QuadGirder, QuadOffset,
	QuaddB, Quadaper, QuadTilt } ;

struct LucretiaParameter QuadPar[nQuadPar] = {
	{"S",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",      {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"B",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"PS",     {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder", {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset", {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"dB",     {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"aper",   {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Tilt",   {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
} ;

int QuadTrackFlag[NUM_TRACK_FLAGS] = {
	1, 1, 0, 0, 0, 
   0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* Solenoid: note that any changes made to the Quad pars, above, need
   to be made also in solenoid parameters, below. */

#define nSolePar 9

enum SoleOrdinal {
	SoleS, SoleP, SoleL, SoleB, SolePS, SoleGirder, SoleOffset,
	SoledB, Soleaper } ;

struct LucretiaParameter SolePar[nSolePar] = {
	{"S",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",      {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"B",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"PS",     {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder", {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset", {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"dB",     {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"aper",   {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
} ;

int SoleTrackFlag[NUM_TRACK_FLAGS] = {
	1, 1, 0, 0, 0, 
   0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* Thin-lens multipole */

#define nMultPar 13

enum MultOrdinal {MultS, MultP, MultL, MultLrad, MultB, MultTilt, MultPoleIndex,
    MultAngle, MultPS, MultGirder, MultOffset, MultdB, Multaper} ; 

struct LucretiaParameter MultPar[nMultPar] = {
	{"S",        {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",        {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",        {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Lrad",     {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"B",        {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Tilt",     {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"PoleIndex",{Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Angle",    {Required,Required,Optional},{Ignored,Required,Ignored },2,2,0,0,NULL    },
	{"PS",       {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder",   {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",   {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"dB",       {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"aper",     {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
} ;


int MultTrackFlag[NUM_TRACK_FLAGS] = {
	1, 1, 0, 0, 0, 
   0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* sector bend */

#define nSBendPar 14

enum SBendOrdinal {
	SBendS, SBendP, SBendL, SBendB, SBendAngle, SBendTilt, SBendEdgeAngle,
	SBendEdgeCurvature, SBendHGAP, SBendFINT, SBendPS, SBendGirder,
	SBendOffset, SBenddB } ;

struct LucretiaParameter SBendPar[nSBendPar] = {
	{"S",            {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",            {Required,Optional,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",            {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"B",            {Required,Required,Optional},{Optional,Optional,Optional},1,2,0,0,NULL    },
	{"Angle",        {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Tilt",         {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"EdgeAngle",    {Optional,Optional,Optional},{Optional,Optional,Optional},1,2,0,0,NULL    },
	{"EdgeCurvature",{Optional,Optional,Optional},{Optional,Optional,Optional},1,2,0,0,NULL    },
	{"HGAP",         {Optional,Optional,Optional},{Optional,Optional,Optional},1,2,0,0,NULL    },
	{"FINT",         {Optional,Optional,Optional},{Optional,Optional,Optional},1,2,0,0,NULL    },
	{"PS",           {Optional,Optional,Optional},{Optional,Optional,Optional},1,2,0,0,NULL    },
  {"Girder",       {Ignored ,Optional,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",       {Ignored ,Optional,Optional},{Ignored ,Required,Optional},6,6,0,0,NULL    },
	{"dB",           {Ignored ,Optional,Optional},{Ignored ,Optional,Optional},1,2,0,0,NULL    },
  } ;

int SBendTrackFlag[NUM_TRACK_FLAGS] = {
	1, 1, 0, 0, 0, 
   0, 0, 0, 0, 0,
	0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* accelerating structures */

#define nLcavPar 15

enum LcavOrdinal {
	LcavS, LcavP, LcavL, LcavVolt, LcavEgain, LcavPhase, LcavFreq, LcavKlystron,
	LcavGirder, LcavOffset, Lcavaper, LcavWakes, LcavKloss, LcavdV, LcavdPhase } ;

struct LucretiaParameter LcavPar[nLcavPar] = {
	{"S",       {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",       {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",       {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Volt",    {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Egain",   {Required,Ignored ,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Phase",   {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Freq",    {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Klystron",{Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder",  {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",  {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"aper",    {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Wakes",   {Ignored ,Optional,Optional},{Ignored,Optional,Optional},0,4,0,0,NULL    },
	{"Kloss",   {Ignored ,Ignored ,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"dV",      {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"dPhase",  {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
} ;

int LcavTrackFlag[NUM_TRACK_FLAGS] = {
	0, 1, 0, 0, 0, 
   0, 1, 1, 1, 1,
	1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;
/* deflecting structures */

#define nTcavPar 16

enum TcavOrdinal {
	TcavS, TcavP, TcavL, TcavVolt, TcavEgain, TcavPhase, TcavFreq, TcavKlystron,
	TcavGirder, TcavOffset, Tcavaper, TcavWakes, TcavKloss, TcavdV, TcavdPhase,
	TcavTilt } ;

struct LucretiaParameter TcavPar[nTcavPar] = {
	{"S",       {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",       {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",       {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Volt",    {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Egain",   {Required,Ignored ,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Phase",   {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Freq",    {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Klystron",{Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder",  {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",  {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"aper",    {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Wakes",   {Ignored ,Optional,Optional},{Ignored,Optional,Optional},0,4,0,0,NULL    },
	{"Kloss",   {Ignored ,Ignored ,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"dV",      {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"dPhase",  {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Tilt",    {Ignored ,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
} ;

int TcavTrackFlag[NUM_TRACK_FLAGS] = {
	1, 1, 0, 0, 0, 
   0, 1, 1, 1, 1,
	1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* dipole correctors -- note that these are treated as drifts by the RMAT code.
   There are two kinds of dipole correctors -- standard XCOR or YCORs on the one
   hand, and XYCORs (combined-function horizontal-vertical correctors) on the other.
   The only dictionary differences between them are that XYCORs have requirements on 
   the length of the B/dB fields in their data structures, and they also have requirements
   on the length of the PS vector.  So XCORs and YCORs use one dictionary, while XYCORs
   use another, almost identical, dictionary. */

#define nCorrectorPar 10

enum CorrectorOrdinal {
	CorS, CorP, CorL, CorB, CorTilt, CorPS, CorGirder, CorOffset, CordB,
	CorLrad} ;

struct LucretiaParameter CorrectorPar[nCorrectorPar] = {
	{"S",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",      {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"B",      {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Tilt",   {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"PS",     {Optional,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder", {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset", {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"dB",     {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Lrad",   {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
}; 

struct LucretiaParameter XYCorrectorPar[nCorrectorPar] = {
	{"S",      {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",      {Required,Optional,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",      {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"B",      {Required,Required,Optional},{Required,Required,Ignored },2,2,0,0,NULL    },
	{"Tilt",   {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"PS",     {Optional,Optional,Optional},{Required,Required,Ignored },2,2,0,0,NULL    },
	{"Girder", {Ignored ,Optional,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset", {Ignored ,Optional,Optional},{Ignored ,Required,Optional},6,6,0,0,NULL    },
	{"dB",     {Ignored ,Optional,Optional},{Required,Required,Ignored },2,2,0,0,NULL    },
	{"Lrad",   {Ignored ,Optional,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    }
}; 

int CorrectorTrackFlag[NUM_TRACK_FLAGS] = {
	1, 0, 0, 0, 0, 
   0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* beam position monitor -- BPMs are treated as drifts by the RMAT code */

#define nBPMPar 8

enum BPMOrdinal {
	BPMS, BPMP, BPML, BPMGirder, BPMOffset, BPMElecOffset, BPMdScale, BPMResolution } ;

struct LucretiaParameter BPMPar[nBPMPar] = {
	{"S",         {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",         {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",         {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder",    {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",    {Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"ElecOffset",{Ignored ,Optional,Optional},{Ignored,Optional,Optional},2,2,0,0,NULL    },
	{"dScale",    {Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Resolution",{Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    }
} ;

int BPMTrackFlag[NUM_TRACK_FLAGS] = {
	0, 0, 1, 1, 1,
	0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* instrument -- treated as a drift by the RMAT code */

#define nInstPar 5

enum InstOrdinal {
	InstS, InstP, InstL, InstGirder, InstOffset } ;

struct LucretiaParameter InstPar[nInstPar] = {
	{"S",     {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",     {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",     {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder",{Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",{Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    }
} ;

int InstTrackFlag[NUM_TRACK_FLAGS] = {
	0, 0, 0, 0, 1,
	1, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* collimator -- treated as a drift by the RMAT code.  Note that collimators also have
   a geometry parameter, which is a text string ("Rectangle" or "Ellipse"), and which
   is obtained by GetCollimatorGeometry function */

#define nCollPar 8

enum CollOrdinal {
	CollS, CollP, CollL, CollLrad, CollGirder, CollOffset, CollTilt, Collaper } ;

struct LucretiaParameter CollPar[nCollPar] = {
	{"S",     {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",     {Required,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"L",     {Required,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Lrad",  {Ignored ,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Girder",{Ignored ,Optional,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Offset",{Ignored ,Optional,Optional},{Ignored,Required,Optional},6,6,0,0,NULL    },
	{"Tilt",  {Ignored ,Required,Optional},{Ignored,Ignored ,Ignored },0,0,0,0,NULL    },
	{"aper",  {Ignored ,Required,Optional},{Ignored,Required,Ignored },2,2,0,0,NULL    }
} ;

int CollTrackFlag[NUM_TRACK_FLAGS] = {
	0, 1, 0, 0, 0,
    0, 0, 0, 0, 0,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;

/* coordinate change element -- has an R-matrix used by the RMAT code */

#define nCoordPar 3

enum CoordOrdinal {
	CoordS, CoordP, CoordChange } ;

struct LucretiaParameter CoordPar[nCoordPar] = {
	{"S",     {Required,Required,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"P",     {Required,Optional,Optional},{Ignored ,Ignored ,Ignored },0,0,0,0,NULL    },
	{"Change",{Required,Required,Optional},{Required,Required,Optional},6,6,0,0,NULL    }
} ;

int CoordTrackFlag[NUM_TRACK_FLAGS] = {
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
} ;


/*=====================================================================*/

/* Power supply parameters */

#define nPSPar 5

enum PSOrdinal {
	PSAmpl, PSdAmpl, PSSetPt, PSStep, PSElem } ;

struct LucretiaParameter PSPar[nPSPar] = {
	{"Ampl",   {Required,Required,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"dAmpl",  {Required,Optional,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"SetPt",  {Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"Step",   {Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"Element",{Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    }
} ;

/*=====================================================================*/

/* Klystron parameters -- note that klystrons also have a Status parameter,
   which is a char array (not numeric), and so not in this list */

#define nKlystronPar 9

enum KlysOrdinal {
	KlysAmpl,  KlysAmplSetPt,  KlysAmplStep, 
	KlysPhase, KlysPhaseSetpt, KlysPhaseStep, 
	KlysElem,  KlysdAmpl,      KlysdPhase } ;

struct LucretiaParameter KlystronPar[nKlystronPar] = {
   {"Ampl",      {Required,Required,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
   {"AmplSetPt", {Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
   {"AmplStep",  {Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"Phase",     {Required,Required,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
   {"PhaseSetPt",{Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
   {"PhaseStep", {Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
   {"Element",   {Ignored ,Ignored ,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"dAmpl",     {Ignored ,Optional,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    },
	{"dPhase",    {Ignored ,Optional,Optional},{Ignored,Ignored, Ignored}, 0,0,0,0,NULL    }
} ;

/*=====================================================================*/

/* Girder parameters */

#define nGirderPar 7

enum GirderOrdinal {
	GirderS, GirderElem, GirderOffset, GirderMover, GirderMoverPos,
		GirderMoverSetpt, GirderMoverStep } ;

struct LucretiaParameter GirderPar[nGirderPar] = {
	{"S",         {Ignored,Required,Optional},{Ignored,Required,Optional},1,2,0,0,NULL  },
	{"Element",   {Ignored,Ignored ,Optional},{Ignored,Ignored ,Optional},0,0,0,0,NULL  },
	{"Offset",    {Ignored,Required,Optional},{Ignored,Required,Optional},6,6,0,0,NULL  },
	{"Mover",     {Ignored,Optional,Optional},{Ignored,Required,Optional},1,6,0,0,NULL  },
	{"MoverPos",  {Ignored,Optional,Optional},{Ignored,Required,Optional},1,6,0,0,NULL  },
	{"MoverSetPt",{Ignored,Ignored ,Optional},{Ignored,Ignored ,Optional},1,6,0,0,NULL  },
	{"MoverStep", {Ignored,Ignored ,Optional},{Ignored,Ignored ,Optional},1,6,0,0,NULL  }
} ;

/*=====================================================================*/

/* LRWF frequency mode parameters */

#define nLRWFFreqPar 5

enum LRWFFreqOrdinal {
	LRWFFreqFreq, LRWFFreqQ, LRWFFreqK, 
		LRWFFreqTilt, LRWFFreqBinWidth } ;

struct LucretiaParameter LRWFFreqPar[nLRWFFreqPar] = {
	{"Freq",    {Ignored,Required,Optional},{Ignored,Ignored,Optional},0,0,0,0,NULL },
	{"Q",       {Ignored,Required,Optional},{Ignored,Ignored,Optional},0,0,0,0,NULL },
	{"K",       {Ignored,Required,Optional},{Ignored,Ignored,Optional},0,0,0,0,NULL },
	{"Tilt",    {Ignored,Required,Optional},{Ignored,Ignored,Optional},0,0,0,0,NULL },
	{"BinWidth",{Ignored,Required,Optional},{Ignored,Ignored,Optional},0,0,0,0,NULL }
} ;
