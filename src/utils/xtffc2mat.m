function [tt,K,N,L,P,A,T,E,FDN,ctwss,cnorm,orbt,S]=xtffc2mat(fname)
%
% [tt,K,N,L,P,A,T,E,FDN,ctwss,cnorm,orbt,S]=xtffc2mat(fname);
%
% Outputs:
%
%   tt    = run title
%   K     = element keyword
%   N     = element name
%   L     = element length
%   P     = element parameter
%   A     = aperture
%   T     = engineering type
%   E     = energy
%   FDN   = NLC Formal Device Name
%   ctwss = coupled twiss (mux,betx,alfx,dx,dpx,muy,bety,alfy,dy,dpy)
%   cnorm = coupling terms of normalizing transform
%           (n13,n14,n23,n24,n31,n32,n41,n42)
%   orbt  = orbit (x,px,y,py,t,pt)
%   S     = suml

% check the input/output arguments
if (nargin~=1)
  error('File name input argument required')
end
if (nargout~=13)
  error('13 output arguments required')
end
if (~ischar(fname))
  error('File name must be a string')
end
[nrow,ncol]=size(fname);
if (nrow~=1)
  error('File name must be a single row')
end

if (exist('xtffc2mat_mex')==3)
  try
    % use the mex-file to download the data
    [tt,keyw,name,L,P,A,etyp,E,fnam,ctwss,cnorm,orbt,S]=xtffc2mat_mex(fname);
    % set the string attribute for the appropriate arrays
    K=char(keyw);
    N=char(name);
    T=char(etyp);
    FDN=char(fnam);
  catch
    % use the (much slower) script to download the data
    [tt,K,N,L,P,A,T,E,FDN,ctwss,cnorm,orbt,S]=xtffc2mat_nomex(fname);
  end
else
  % use the (much slower) script to download the data
  [tt,K,N,L,P,A,T,E,FDN,ctwss,cnorm,orbt,S]=xtffc2mat_nomex(fname);
end

end