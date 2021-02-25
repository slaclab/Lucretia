function [stat,G] = AddMoverToGirder( dofs, Gin )
%
% ADDMOVERTOGIRDER Add fields for a mover to the data structure for a
% girder.
%
% [stat,G_out] = AddMoverToGirder( dofs, G_in ) takes an existing girder
%    data structure and adds fields required for a mover to it.  G_in and
%    G_out are the original and modified girder data structure.  Return
%    argument stat is a Lucretia status and message cell array (type help
%    LucretiaStatus for more information).  Calling argument dofs is a
%    row-vector list of the degrees of freedom of the mover, where 1==x
%    offset, 2==x angle, 3==y offset, 4==y angle, 5==S offset, 6==xy
%    rotation; for example, dofs == [1 3 6] corresponds to a mover with x,
%    y and xy rotation degrees of freedom.
%
% Return status value:  +1 if successful, 0 if invalid degrees of freedom
% specified in dofs.
%
% See also:  AssignToGirder, GirderStruc.
%

%==========================================================================

  stat = InitializeMessageStack( ) ;

% check dofs first

  dofs = sort(dofs(:)) ; dofs = dofs' ;
  if ( (~isnumeric(dofs)) | (~isvector(dofs)) | (length(dofs)>6) | ...
       (min(dofs)<1)      | (max(dofs)>6)                              )
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'Invalid DOF assignment requested in AddGirderToMover') ;
    return ;
  end
  
% eliminate duplicates

  valold = 0 ;
  for count = 1:length(dofs)
    if (dofs(count)==valold)
      dofs(count) = 0 ;
    else
      valold = dofs(count) ;
    end
  end

  dvec = dofs(find(dofs>0)) ;
  
% add the mover 

  G = Gin ;
  G.Mover = dvec ;
  G.MoverPos = zeros(1,length(dvec))   ;
  G.MoverSetPt = zeros(1,length(dvec)) ;
  G.MoverStep = zeros(1,length(dvec))  ;
  
%  