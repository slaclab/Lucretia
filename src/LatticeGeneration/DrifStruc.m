function D = DrifStruc( length, Name )
%
% DRIFSTRUC Create a Lucretia drift data structure.
%
% D = DrifStruc( L, Name ) creates a Lucretia drift-space structure with a
%    given length and name.
%

%================================================================

D.Name = Name ;
D.S = 0 ; D.P = 0 ;
D.Class = 'DRIF' ;
D.L = length ;
D.TrackFlag.ZMotion = 0 ;
D.TrackFlag.LorentzDelay = 0 ;