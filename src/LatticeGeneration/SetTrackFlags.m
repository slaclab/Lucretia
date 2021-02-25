function iss = SetTrackFlags( FlagName, FlagValue, istart, varargin )
%
% SETTRACKFLAGS Set selected tracking flags over a selected range of
%    elements.
%
% list = SetTrackFlags( Name, Value, start, end ) will identify all
%    elements between start and end, inclusive, which have tracking flag
%    Name (ie, Name is a field of TrackFlag, which is a field of the
%    element), and set the flag to Value on all those elements.  A list of
%    the elements which have had their tracking flags altered is returned.
%
% list = SetTrackFlags( Name, Value, elemlist ) will identify all elements
%    in elemlist which have tracking flag Name, and set the flag to Value
%    on all those elements.
%
% Version date:  03-Apr-2006.
%

% MOD:
%       03-Apr-2006, PT:
%           if SR is turned on and zero-length magnets are present, don't
%           turn SR on for them and issue a warning.
%       13-Jan-2006, PT:
%           support for list-based selection of elements.

global BEAMLINE
persistent SRWarnIssued
iss = [] ;
CheckSRLength = 0 ;

if ( (strcmp(FlagName,'SynRad')) & (FlagValue ~= 0) )
  CheckSRLength = 1 ;
end

if (nargin==3)
    elemlist = istart ;
else
    elemlist = linspace(istart,varargin{1},varargin{1}-istart+1) ;
end

for count = elemlist
    
    BadLength = 0 ;
    
    if ( isfield(BEAMLINE{count},'TrackFlag') )
        
        if ( isfield(BEAMLINE{count}.TrackFlag,FlagName) )
            if (CheckSRLength == 1)
              if (BEAMLINE{count}.L == 0)
                BadLength = 1 ;
                if (isfield(BEAMLINE{count},'Lrad'))
                  if (BEAMLINE{count}.Lrad > 0)
                    BadLength = 0 ;
                  end
                end
                if BadLength == 1 && ~isempty(SRWarnIssued)
                  warning('SR not switched on in elements with non-zero length/Lrad') ;
                  SRWarnIssued = 1 ;
                end
              end
            end
            if (BadLength == 0) 
              BEAMLINE{count}.TrackFlag = setfield(BEAMLINE{count}.TrackFlag,FlagName,FlagValue) ;
              iss = [iss count] ;
            end
        end
        
    end
    
end