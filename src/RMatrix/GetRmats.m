% GETRMATS Compute R-matrices over a range.
%
% [STAT,R] = GetRmats( START , FINISH ) Returns the linear 6 x 6 maps
%    ("R-matrix") of BEAMLINE elements from START to FINISH inclusive.  R
%    is a data structure containing one field, RMAT, a 6 x 6 real (double
%    precision) matrix. Return variable STAT is a Lucretia status and
%    message cell array (type help LucretiaStatus for more information).
%
% Return status values:  +1 for successful completion, -1 for completion
%    but some element R matrices could not be computed due to errors, 0 for
%    failure due to errors.
%
% V = GetRmats( "version" ) Returns the version dates of all key components
%    in Lucretia.
%
% See also RmatAtoB.