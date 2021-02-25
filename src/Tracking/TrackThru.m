% TRACKTHRU Track bunches of a beam through a lattice.
%
%   [STAT,BEAMOUT] = TrackThru(E1,E2,BEAMIN,B1,B2) tracks bunches
%      B1 through B2 of beam BEAMIN through elements E1 through E2
%      inclusive. The resulting beam is stored in BEAMOUT.  Note that only
%      the bunches which are tracked will be copied to BEAMOUT, any bunches
%      in BEAMIN outside the range B1 to B2 inclusive will not be copied,
%      and BEAMIN will populate bunches 1 through (B2-B1+1). STAT is a
%      Lucretia status and message cell array (type help LucretiaStatus for
%      more information). STAT{1} is a number indicating overall status (1
%      = completed without incident; 0 = forced to stop partway through ;
%      -1 = tracked all bunches through all elements but encountered
%      unexpected conditions).
%
%   [STAT,BEAMOUT,DATA] = TrackThru(E1,E2,BEAMIN,B1,B2) returns
%      BPM or instrument information (readings and/or beam parameters
%      which are not generally available in the "real world") to 
%      cell array DATA.  DATA{1} is a structure array of BPM data;
%      DATA{2} is a structure array of data from INST, WIRE, PROF, IMON,
%      BLMO, SLMO elements; DATA{3} is a structure array of data from
%      RF-structure BPMs.  Only BPMs etc. which are selected for the
%      return of information (via their tracking flags) will return
%      information to DATA.
%
%   [...] = TrackThru(E1,E2,BEAMIN,B1,B2,FLAG) uses FLAG to indicate
%      the order of the tracking loops.  When FLAG == 0 or is absent, the
%      outer loop is over elements and the inner loop is over bunches; when
%      FLAG == 1 the outer loop is over bunches and the inner loop is over
%      elements.  When FLAG==1, any long-range wakefield kicks in any
%      elements will be preserved from one call of TrackThru to the next,
%      to permit simulation of beamlines with intra-train tuning
%      capabilities.  In this case wakefields are deleted when TrackThru is
%      called with a B1 value which is <= B1 from the previous call.
%
%   TrackThru('version') returns version information.
%
%   TrackThru('clear') clears internal dynamic memory which would 
%      otherwise be used on multiple calls to TrackThru to speed execution.
%      There is no particular reason to want to do this, but it may prove
%      useful in applications which use a lot of dynamic allocation for a
%      brief period and then want to get rid of it before continuing.  This
%      option also allows the user to force deletion of existing long-range
%      wakefield kicks.
%


