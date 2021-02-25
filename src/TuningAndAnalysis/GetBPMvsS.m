function [S,x,y] = GetBPMvsS( bpmdatastruc ) 
%
% GETBPMVSS Extract BPM readings in vector form from data returned from
% tracking.
%
% [S,x,y] = GetBPMvsS( bpmdatastruc ) returns vectors of S, x, and
%    y readings (including BPM offsets, rotations, and resolution limits)
%    from the data which is returned from tracking.  All vectors have
%    dimensions of meters.  Argument bpmdatastruc is the first cell in the
%    third returned argument from TrackThru, ie, if your TrackThru call
%    looks like:
%        [stat,beam2,data] = TrackThru(...)
%    then bpmdatastruc = data{1}.
%
% See also BPMZPlot, PlotOrbit.
%

  S = zeros(1,length(bpmdatastruc)) ;
  x = S ; y = S ;
  
  for count = 1:length(bpmdatastruc)
      
    S(count) = bpmdatastruc(count).S ;
    x(count) = bpmdatastruc(count).x ;
    y(count) = bpmdatastruc(count).y ;
    
  end
  
