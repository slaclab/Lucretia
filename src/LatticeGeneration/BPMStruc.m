function B = BPMStruc( L, name ) 
%
% BPMSTRUC create Lucretia data structure for a beam position monitor.
%
% B = BPMStruc( L, Name ) returns a Lucretia data structure for a beam
%    position monitor of a given length with a given name.  The BPM's
%    resolution is set to zero by default.

% MOD:
%      18-Oct-2005, PT:
%         Add a scale factor error (dScale) which is initialized to zero.

%=========================================================

% start with the name, length, energy, S position, and
% class
%
  B.Name = name ; B.S = 0 ; B.P = 0 ; B.Class = 'MONI' ;
  B.L = L ;
%
% Resolution, electrical offsets, and scale factor
%
  B.Resolution = 0 ;
  B.ElecOffset = [0 0] ;
  B.dScale = 0 ;
%
% mechanical offset wrt girder and girder # 
%
  B.Offset = [0 0 0 0 0 0] ;
  B.Girder = 0 ;
%
% set the tracking flags to their off conditions
%
  B.TrackFlag.MultiBunch = 0 ;
  B.TrackFlag.GetBPMData = 0 ;
  B.TrackFlag.GetBPMBeamPars = 0 ;
  B.TrackFlag.ZMotion = 0 ;
  B.TrackFlag.LorentzDelay = 0 ;