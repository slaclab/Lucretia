function S = MarkerStruc( name )
%
% MARKERSTRUC return a Lucretia data structure for a marker.
%
%   S = MarkerStruc( name ) returns the data structure for a Lucretia
%      marker element.  Such an element has only a name, an S position,
%      and a design momentum P.
%

%================================================================

S.Name = name ; S.Class = 'MARK' ;
S.S = 0 ; S.P = 0 ;

