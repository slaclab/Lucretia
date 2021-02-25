function [stat,errdat] = ErrorGroupGaussErrors( group, meanval, rmsval, ...
  keepold, randTrunc )
%
% ERRORGROUPGAUSSERRORS Apply Gaussian-distributed errors to the members of
% an error group.
%
% [stat,ErrorData] = ErrorGroupGaussErrors( group, Meanval, RMSVal,
%    KeepOld, RandTrunc ) applies Gaussian-distributed errors or misalignments to the
%    members of an error group (defined using MakeErrorGroup).  Return
%    argument stat is a Lucretia status message list (type help
%    LucretiaStatus for more information); return argument ErrorData tells
%    the mean and RMS of the applied errors, along with the actual applied
%    errors for each group member and the total errors for each group
%    member.  Calling arguments are defined as follows;
%
%    group is an error group, generated using MakeErrorGroup.
%
%    Meanval is the desired mean error or offset.  Meanval must be a scalar
%       for dB, dAmpl, dPhase, or dV error groups, a 1 x 2 vector for
%       ElecOffset or BPMOffset error groups, or a 1 x 6 vector for Offset
%       error groups.
%
%    RMSVal is the desired RMS of the errors, and must have the same shape
%       as Meanval.
%
%    KeepOld is a flag that indicates whether existing errors are to be
%       overridden (KeepOld == 0) or added to (KeepOld == 1) by the new
%       errors.  KeepOld must have the same shape as Meanval and RMSVal.
%
%    RandTrunc: if provided, assign a value to this argument to truncate
%       the random number distribution in units of sigma, if this is a
%       scalar, apply same truncation to all rmsval, else Truncate should
%       have the same dimensionality as rmsval
%
% Return Arguments: +1 if successful completion, or 0 if the arguments are
% not consistent with one another.
%
% See Also:  MakeErrorGroup, SetGaussianErrors.
%
% Version date:  25-September-2007.
%

% MOD:
%      14-Feb-2012, GW:
%         Add distributedLucretia support
%      25-sep=2007, GRW:
%         Add truncate functionality
%      24-feb-2006, PT:
%         return the complete applied error data.

%==========================================================================

global BEAMLINE KLYSTRON PS GIRDER %#ok<NUSED>
errdat = [] ;
stat = InitializeMessageStack( ) ;

% first make sure the arguments are consistent with one another
if ~isfield(group,'error')
  error('Incorrectly formatted error group')
end
statarg = EGGEVerifyArgs( group.error, meanval, rmsval, keepold ) ;
stat = AddStackToStack( stat, statarg ) ;
if (statarg{1} ~= 1)
  stat{1} = 0 ;
  return ;
end

% if truncate argument given, check that it is either a scalar or of the
% same dimensionality as rmsval
% NB: randTrunc in group.gener structure to be eval'd, => has to exist (0=
% no truncation)
if exist('randTrunc','var') && length(randTrunc)>1 && ~isequal(size(randTrunc),size(rmsval))
  error('RandTrunc should either be scalar or of the same dimensionality and type as rmsval!');
elseif ~exist('randTrunc','var')
  randTrunc=0; %#ok<NASGU>
end % if truncate arg wrong

% if everything is all right, apply the errors

% pre-allocate the accumulation table for execution speed

eval(group.dimension) ;

% loop over clusters

% If specify distributed ops then apply errors to each MC machine
isdist=false;
if ~isempty(group.comment) && isa(group.comment,'distributedLucretia')
  isdist=true;
  DL=group.comment;
  dlLoop=DL.workers;
else
  dlLoop=1;
end

indxList=[];
for iloop=1:length(dlLoop)
  
  for ClusterCount = 1:length(group.ClusterList) ;
    
    % generate the errors
    
    if (~isempty(group.gener))
      eval(group.gener) ;
    end
    eval(group.accum) ;
    
    % loop over cluster members
    
    for count = 1:length(group.ClusterList(ClusterCount).index)
      indx = group.ClusterList(ClusterCount).index(count) ;
      ds = group.ClusterList(ClusterCount).dS(count) ; %#ok<NASGU>
      
      % execute the adjustment string
      
      if (~isempty(group.adjust))
        eval(group.adjust) ;
      end
      
      % execute the application string
      if isdist
        eval(regexprep(group.apply,'.+=','temp='));
        DL.latticeSyncVals(DL.workers(iloop)).(group.error)(indx,:)=temp;
        indxList=[indxList indx];
      else
        eval(group.apply) ;
      end
      
    end
    
  end
  eval(group.statistics) ;
  errdat.Mean = ErrMean ;
  errdat.std = ErrStd ;
  errdat.AppliedErrors = ErrAccum ;
  
end

% Apply errors to MC machines
if isdist
  DL.setError(group.error,indxList);
end

%==========================================================================
%==========================================================================
%==========================================================================

% subfunction to perform verification of the arguments

function statarg = EGGEVerifyArgs( errstring, meanvec, rmsvec, keepold )

statarg = InitializeMessageStack( ) ;

% if we are doing offsets, verify that the mean, RMS, and keep flags are 1 x 6;
% if BPMOffset or ElecOffset, that they are 1 x 2;
% otherwise verify that they are scalar

switch errstring
  case 'Offset'
    lenvec = 6 ;
  case {'BPMOffset','ElecOffset'}
    lenvec = 2 ;
  otherwise
    lenvec = 1 ;
end
if ( (sum(size(meanvec)==[1 lenvec]) ~= 2) || ...
    (sum(size(rmsvec) ==[1 lenvec]) ~= 2) || ...
    (sum(size(keepold)==[1 lenvec]) ~= 2)       )
  statarg{1} = 0 ;
  statarg = AddMessageToStack(statarg,...
    'Arguments 3,4,7 in SetGaussianErrors have incorrect size') ;
  return ;
end

% verify that mean and rms, and keepold are numeric

if ( (~isfloat(meanvec)) || (~isfloat(rmsvec)) || (~isnumeric(keepold)) )
  statarg{1} = 0 ;
  statarg = AddMessageToStack(statarg, ...
    'Arguments 3,4,7 in SetGaussianErrors must be numeric') ;
  return ;
end

% verify that all entries in keepold are either one or zero

if ( length(find(keepold==1))+length(find(keepold==0)) ~= length(keepold) )
  statarg{1} = 0 ;
  statarg = AddMessageToStack(statarg, ...
    'Argument 7 in SetGaussianErrors must be 1''s or 0''s') ;
  return ;
end
