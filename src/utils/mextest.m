function [tt,K,N,L,P,A,T,E,FDN,twss,orbt,S]=mextest(fname)
%
% [tt,K,N,L,P,A,T,E,FDN,twss,orbt,S]=mextest(fname);
%
% Outputs:
%
%   tt   = run title
%   K    = element keyword
%   N    = element name
%   L    = element length
%   P    = element parameter
%   A    = aperture
%   T    = engineering type
%   E    = energy
%   FDN  = NLC Formal Device Name
%   twss = twiss (mux,betx,alfx,dx,dpx,muy,bety,alfy,dy,dpy)
%   orbt = orbit (x,px,y,py,t,pt)
%   S    = suml

% check the input/output arguments

if (nargin~=1)
  error('File name input argument required')
end
if (nargout~=12)
  error('12 output arguments required')
end
if (~ischar(fname))
  error('File name must be a string')
end
[nrow,ncol]=size(fname);
if (nrow~=1)
  error('File name must be a single row')
end

% use the mex-file to download the data

[tt,keyw,name,L,P,A,etyp,E,fnam,twss,orbt,S]=mextest_mex(fname);

% set the string attribute for the appropriate arrays

K=char(keyw);
N=char(name);
T=char(etyp);
FDN=char(fnam);
