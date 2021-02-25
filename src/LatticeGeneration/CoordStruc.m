function C = CoordStruc( dx, dtheta, dy, dphi, dz, dpsi, name )
%
% COORDSTRUC Return the data structure for a change in coordinates.
%
% C = CoordStruc( dx, dtheta, dy, dphi, dz, dpsi, name ) returns a Lucretia 
%    data structure for an element which represents a change in the
%    reference coordinates.
%
% Version date:  22-Mar-2007.
%

% MOD:
%      22-Mar-2007, PT:
%         changed comments / help to show name argument.

%=========================================================================

  C.Class = 'COORD' ; C.Name = name ;
  C.P = 0 ; C.S = 0 ;
  C.Change = [dx dtheta dy dphi dz dpsi] ;
  C.TrackFlag.ZMotion = 0 ;