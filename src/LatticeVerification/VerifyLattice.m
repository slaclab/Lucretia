% VERIFYLATTICE Verify the internal consistency and well-formedness of a
%    Lucretia lattice.
%
% [stat,error,warning,info] = VerifyLattice( ) verifies that the Lucretia
%    lattice is correctly loaded and internally consistent.  Specifically,
%    it verifies the following:
%
%    All elements are of known classes, have all required fields and all
%      valid tracking flags for that class, and have nonzero design 
%      momentum values
%    All pointers to klystrons, power supplies, girders, and wakefields are
%      valid (ie, they point to existing data structures, and the data
%      structures point back at the correct elements)
%    All wakefields have the correct fields and that all field length 
%      constraints are obeyed
%    All klystrons have valid status values.
%
%    In addition, a number of checks of optional features are performed,
%    for example whether girders have movers and whether the KLYSTRON, PS,
%    WF, and GIRDER tables are present (even if no element points at them).
%
% Return arguments stat, error, warning, and info are all Lucretia message
%    and status cell arrays (type help LucretiaStatus for more
%    information).  Argument stat contains the status information for the
%    VerifyLucretia function itself; error, warning, info contain lattice
%    errors, lattice warnings, and informational messages, respectively,
%    with the first cell telling the total number of messages in each
%    array.
%   