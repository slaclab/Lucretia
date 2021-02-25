function [stat,ErrorData] = SetGaussianErrors( tablecell, range, ...
                              meanval, rmsval, errname, clusters, keepold )
% 
% SETGAUSSIANERRORS Apply normal-distributed misalignments and errors to
%    Lucretia data arrays.
%
% [stat,ErrorData] = SetGaussianErrors( Table, Range, Mean, Std, ErrName,
%    Clusters, KeepOld ) applies Gaussian-distributed misalignments and
%    errors to entries in the BEAMLINE, KLYSTRON, GIRDER, or PS data
%    tables.  Return argument stat is a Lucretia status cell array (type
%    help LucretiaStatus for more information), return argument ErrorData
%    tells how many sets of errors were generated, as well as mean and RMS
%    values of the generated errors.  Calling arguments are defined as
%    follows:
% 
%      Table:  either a string or a cell array of strings.  The first
%         string must be the name of the table of interest ('BEAMLINE',
%         'KLYSTRON', 'PS', 'GIRDER').  If the first string is BEAMLINE,
%         strings 2 and 3 are the element class and element name,
%         respectively, which are to be given errors.
%      Range:  1 x 2 vector with the initial and final table entries which
%         are to receive errors.
%      ErrName:  string which names the error which is to be set ('dB',
%         'dV', 'dPhase', 'dAmpl', 'dPhase', 'Offset', 'ElecOffset',
%         'BPMOffset').
%      Clusters: scalar, if clusters == 2 then all slices (for errors) or
%         block members (for misalignments) will receive a common error set
%         (with appropriate adjustments to the misalignments for the fact
%         that block members with different S positions need different
%         position offsets if they receive angle offsets).  If clusters ==
%         0, each entry in BEAMLINE gets a unique error.  If clusters == 1,
%         the result is similar to cluster == 2, except that only block
%         members with the name and class requested will be misaligned and
%         all other block members will not be moved.
%      Mean, Std, KeepOld are scalars (for errors) or 1 x 6 vectors (for
%         misalignments) specifying the mean value, RMS value, and whether
%         to keep or replace existing values of the error.  A value of 1
%         indicates that old errors are kept, 0 mandates replacement.  For
%         offsets, whether each of the 6 entries is kept or replaced must
%         be specified.  Similarly BPM offsets and electrical offsets
%         require 2 values for each of Mean, Std, KeepOld.
%
% Return status:  +1 if executed successfully, 0 if an invalid combination
%    of arguments is supplied.
%
% See also:  MakeErrorGroup, ErrorGroupGaussErrors.
%
% Version Date:  24-Feb-2006.
%

% MOD:
%
%      24-Feb-2006, PT:
%         return full applied errors to user.

%==========================================================================

  stat = InitializeMessageStack( ) ;
  
% build an error group if possible...

  [statcall,group] = MakeErrorGroup( tablecell, range, errname, ...
                        clusters, ' ' ) ;
  stat = AddStackToStack( stat, statcall ) ;
  if (statcall{1} ~= 1)
      stat{1} = 0 ;
      return ;
  end
  
% apply the desired error

  [statcall,errdat] = ErrorGroupGaussErrors( group, meanval, rmsval, ...
                                                keepold ) ;
  stat = AddStackToStack( stat, statcall ) ;
  if (statcall{1} ~= 1)
      stat{1} = 0 ;
      return ;
  end
  
% construct the return argument

  ErrorData.NumErrors = length(group) ;
  ErrorData.Mean = errdat.Mean ;
  ErrorData.std = errdat.std ;
  ErrorData.AppliedErrors = errdat.AppliedErrors ;

%  
