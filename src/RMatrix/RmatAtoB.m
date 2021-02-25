% RMATATOB Use Lucretia to compute R-matrix between 2 points.
%
% [STAT,R] = RmatAtoB( START, FINISH ) Returns R, the linear 6 x 6 map
%    ("R-matrix") from the upstream face of the element indexed by START to
%    the downstream face of the element indexed by FINISH.  STAT is a
%    Lucretia status and message cell array (type help LucretiaStatus for
%    more information).
%
% Return status values: +1 for successful completion, 0 for failure due to
%    errors.
%
% V = RmatAtoB( "version" ) Returns the version dates of all key components
%    in Lucretia.
%
% See also GetRmats.