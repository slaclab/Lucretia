function [x,y,S] = PlotOrbit( bpmdatastruc, varargin )
%
% PlotOrbit Plot the beam orbit from BPMs as a function of S along the beamline.
%
% [x,y,S] = PlotOrbit( bpmdata ) plots the x vs S and y vs S BPM readings
%    on separate axes.  Argument bpmdata is the data structure of BPM
%    information returned by TrackThru (ie, bpmdata == the first cell of
%    the 3rd return from TrackThru).
%
% [x,y,S] = PlotOrbit( bpmdata, ustring ) and
% [x,y,S] = PlotOrbit( bpmdata, uxstring, uystring ) allow selection of a
%    unit other than meters for x and y.  Recognized strings are "mm"
%    (millimeters) and "um" (micrometers).  In the former case the same
%    unit is used for both x and y.
%
% [x,y,S] = PlotOrbit( bpmdata, uxstring, uystring, usstring ) allows
%    selection of a different unit for the S coordinate.  Recognized values
%    are "km" (kilometers).  
%
% In any case, the returned x,y, and S vectors are in meters.
%
% See also: BPMZplot, TrackThru.

%=========================================================================%


  xf = 1 ; yf = 1 ; sf = 1 ;
  if (nargin==2)
    if (strcmpi(varargin{1},'mm'))
      xf = 1e3 ;
      yf = 1e3 ;
    end
    if (strcmpi(varargin{1},'um'))
      xf = 1e6 ;
      yf = 1e6 ;
    end
  end
  if (nargin >= 3)
    if (strcmpi(varargin{1},'mm'))
      xf = 1e3 ;
    end
    if (strcmpi(varargin{1},'um'))
      xf = 1e6 ;
    end
    if (isempty(varargin{2}))
      yf = xf ;
    elseif (strcmpi(varargin{2},'mm'))
      yf = 1e3 ;
    elseif (strcmpi(varargin{2},'um'))
      yf = 1e6 ;
    end
  end
  if (nargin == 4)
    if (strcmpi(varargin{3},'km'))
      sf = 1e-3 ;
    end 
  end
    
    
  S = zeros(1,length(bpmdatastruc)) ;
  x = S ; y = S ;
  
  for count = 1:length(bpmdatastruc)
      
    S(count) = bpmdatastruc(count).S ;
    x(count) = bpmdatastruc(count).x ;
    y(count) = bpmdatastruc(count).y ;
    
  end
  
  figure ; subplot(2,1,1) ;
  plot(S*sf,x*xf) ;
  subplot(2,1,2) ;
  plot(S*sf,y*yf) ;