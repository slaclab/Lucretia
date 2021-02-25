function [stat,G] = GirderStruc( varargin )
%
% GIRDERSTRUC Return data structure for an element girder
%
% [stat,G] = GirderStruc( ) returns the data structure for an element
%    girder with no associated mover.  
% [stat,G] = GirderStruc( dofs ) returns the data structure for an element
%    girder with a mover, where the mover degrees of freedom are determined
%    by calling argument dofs (a 1 x n vector where n<=1<=6).  For example,
%    a mover with x,y, roll degrees of freedom would be specified by dofs
%    == [1 3 6].
% Return argument Stat is a Lucretia status and message cell array (type
% help LucretiaStatus for more information).
%
% Status return values:  +1 for successful completion, 0 if argument dofs
% is not valid.
%
% See also:  AssignToGirder, AddMoverToGirder.
%

%==========================================================================

  stat = InitializeMessageStack( ) ;
  
% make a blank girder structure

  G.S = 0 ; G.Element = [] ; G.Offset = [0 0 0 0 0 0] ;
  
% if there are DOF arguments, check them out

  if (nargin==0)
      return ;
  end
  dofs = varargin{1} ; dofs = sort(dofs(:)) ; dofs = dofs' ;
  
  [statcall,G] = AddMoverToGirder( dofs, G ) ;
  if (statcall{1} ~= 1)
    stat{1} = 0 ;
    stat = AddStackToStack(stat,statcall) ;
  end
%  