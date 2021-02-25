function W = InstStruc( L , Class, Name ) 
%
% INSTSTRUC Create a Lucretia data structure for a beam instrument.
%
% W = InstStruc( L, Class, Name ) returns a Lucretia data structure for a
%    beamline instrument with a given length and name, and which is
%    associated with a suppored instrument class ('INST', 'BLMO', 'PROF',
%    'WIRE', 'SLMO', 'IMON').
%

  W.Name = Name ; W.S = 0 ; W.P = 1 ; W.Class = Class ;
  W.L = L ;

  W.Offset = [0 0 0 0 0 0] ;
  W.Girder = 0 ;

  W.TrackFlag.GetInstData  = 0 ;
  W.TrackFlag.ZMotion      = 0 ;
  W.TrackFlag.LorentzDelay = 0 ;
  W.TrackFlag.MultiBunch   = 0 ;