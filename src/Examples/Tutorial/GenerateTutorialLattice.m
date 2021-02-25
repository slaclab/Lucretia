function statall = GenerateTutorialLattice()
%
% GenerateTutorialLattice -- instantiate the BEAMLINE cell array needed for
%    the Lucretia Tutorial.
%
% GenerateTutorialLattice takes no arguments, and returns a master status
%    cell array detailing the success or failure of various subsidiary
%    functions called by GenerateTutorialLattice.  At the end of execution,
%    the global arrays BEAMLINE and WF are fully populated.
% 
% Version date:  11-Mar-2008.

% Modification History:
%
%==========================================================================

  global BEAMLINE GIRDER KLYSTRON PS WF %#ok<NUSED>
  statall = InitializeMessageStack() ;

% Ordinarily one would start with a lattice which is defined in an XSIF
% file, and use XSIFToLucretia to generate the lattice.  However, it is
% possible (if not particularly sensible, IMO) to do it "by hand" in
% Matlab.  We'll do that here, partially to demonstrate the correct use of
% the lattice generation tools...

% define a standard RF cavity with no wakefields or central energy loss...

  ilccav = RFStruc( 1.0362, 1.0362*31.5, 0, 1300, 0, 0, 0, 0.039, 'ILCCav' ) ;
  
% pick up the wakefields from external files, store them in the WF data
% structure, and point the ilccav at them

  [stat,WF.ZSR(1)] = ParseSRWF('ilc2005.lwake.sr.data',0.1) ;
  statall = AddStackToMasterStack(statall,stat,'ParseSRWF longitudinal') ;
  [stat,WF.TSR(1)] = ParseSRWF('ilc2005.twake.sr.data',0.1) ;
  statall = AddStackToMasterStack(statall,stat,'ParseSRWF transverse') ;
  [stat,WF.TLR{1}] = ParseFrequencyDomainTLR('LBand_Freq_Wake_XY_0.dat',0) ;
  statall = AddStackToMasterStack(statall,stat,...
            'ParseFrequencyDomainTLR') ;
  WF.TLRErr = cell(0) ;
  ilccav.Wakes = [1 1 1 0] ;
  
% define a couple of other components:  a standard drift space of 10 cm and
% one of 30 cm (the former for inter-element, the latter for inter-girder),
% a BPM of 10 cm length, and a combined- function xy corrector of 10 cm
% length.  Note that we are not using standard ILC / TESLA separations, but
% simply whatever comes to mind and seems convenient

  bpm = BPMStruc(0.1, 'BPM') ; bpm.Resolution = 3e-6 ;
  drif10cm = DrifStruc(0.1,'Drift10cm') ; drif30cm = DrifStruc(0.3,'Drift30cm') ;
  xycor = CorrectorStruc(0.1,[0 0],0,3,'XYCor') ;
  
% now to define the quads.  The quads will be 20 cm long, with 3.5 cm
% aperture radius, and will be instantiated as 2 longitudinal "slices". The
% phase advance per FODO cell will be set to 120 degrees in each plane,
% which requires that the focal length of the quads be equal to the
% inter-quad spacing over sqrt(3).  First compute the length of one
% half-cell:

  lcellov2 = 8*ilccav.L + 8 * drif10cm.L + xycor.L + drif10cm.L + ...
      0.2 + bpm.L + drif30cm.L ;
  
% now the integrated strength of the whole quad is given by brho times
% sqrt(3) divided by length of the half-cell, and we're starting at 5 GeV

  Bq = sqrt(3) * (5/0.299792458) / lcellov2 ;
  
% create the QF and QD quads, bearing in mind that each instance will be
% 1/2 of the total quad (split quad).  Set the momenta of the QF and QD to
% 5 GeV, so that we can scale them later...

  QF = QuadStruc(0.1,Bq/2,0,0.035,'QF')  ; 
  QD = QuadStruc(0.1,-Bq/2,0,0.035,'QD') ; 
  
% construct the first FODO cell, and for now set the momentum to 5 GeV at
% each element (we'll get the momentum profile right later)

  for count = 1:8
      BEAMLINE{2*count-1} = ilccav ; BEAMLINE{2*count} = drif10cm ;
  end
  BEAMLINE{17} = xycor ; BEAMLINE{18} = drif10cm ; 
  BEAMLINE{19} = QF ; BEAMLINE{20} = QF ; 
  BEAMLINE{21} = bpm ; BEAMLINE{22} = drif30cm ;
  
  for count = 1:22
      BEAMLINE{count+22} = BEAMLINE{count} ;
  end
  BEAMLINE{19+22} = QD ; BEAMLINE{20+22} = QD ;
  
  icell1end = length(BEAMLINE) ; 
  for count = 1:icell1end
      BEAMLINE{count}.P = 5 ;
  end
  
% compute the # of FODO cells needed to reach 10 GeV, starting from 5 GeV

  voltage = ilccav.Volt ; % this is in MV
  ncell = ceil(5000/(16*voltage)) ;
  
% generate the remaining cells needed for full energy

  for ccount = 1:ncell-1
      for ecount = 1:icell1end
          BEAMLINE{ecount+ccount*icell1end} = BEAMLINE{ecount} ;
      end
  end
  ifodoend = length(BEAMLINE) ;
  
% now we need the 4-bend chicane -- let's use 1 meter bend magnets for
% this, and aim for a total deflection of about 5 cm at the center of the
% chicane.  This means that about 50 mrad of bend is needed per magnet.
% We'll do the bend magnets as unsplit...

  bangle = 5e-2 ; bbend = (5/.299792458) * bangle ; 
  bplus = SBendStruc(1, [bbend 0], bangle, [bangle/2 bangle/2], ...
      [0 0], [0.035 0.035], [0.5 0.5], 0, 'BPlus') ;
  bminus = SBendStruc(1, -[bbend 0], -bangle, -[bangle/2 bangle/2], ...
      [0 0], [0.035 0.035], [0.5 0.5], 0, 'BMinus') ;
  
% finally, a wire scanner will be needed at the center of the chicane...

  wire = InstStruc(0.1,'WIRE','EsprWire') ;
  
% and we can construct the chicane  
  
  BEAMLINE{ifodoend+1} = bplus     ; BEAMLINE{ifodoend+2} = drif10cm ;
  BEAMLINE{ifodoend+3} = bminus    ; BEAMLINE{ifodoend+4} = drif10cm ;
  BEAMLINE{ifodoend+5} = wire      ; BEAMLINE{ifodoend+6} = bpm      ;
  BEAMLINE{ifodoend+7} = drif10cm  ; BEAMLINE{ifodoend+8} = bminus   ;
  BEAMLINE{ifodoend+9} = drif10cm  ; BEAMLINE{ifodoend+10} = bplus   ;
  BEAMLINE{ifodoend+11} = drif10cm ; BEAMLINE{ifodoend+12} = bpm     ;
  
  for count = ifodoend+1:ifodoend+12 
      BEAMLINE{count}.P = 5 ;
  end
  
% now set the correct no-load momentum profile and scale the magnets to it,
% and set the S positions as well

  stat = SetDesignMomentumProfile(1,length(BEAMLINE),0,5) ;
  statall = AddStackToMasterStack(statall,stat,'SetDesignMomentumProfile') ;
  SetSPositions(1,length(BEAMLINE),0) ;
  
% find the blocks and slices and set the data on them into the lattice

  stat = SetElementBlocks(1,length(BEAMLINE)) ;
  statall = AddStackToMasterStack(statall,stat,'SetElementBlocks') ;
  stat = SetElementSlices(1,length(BEAMLINE)) ;
  statall = AddStackToMasterStack(statall,stat,'SetElementSlices') ;
  
