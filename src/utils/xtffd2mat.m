function [tt,K,N,L,P,A,T,E,FDN,twss,orbt,S]=xtffd2mat(tname,oname)
%
% [tt,K,N,L,P,A,T,E,FDN,twss,orbt,S]=xtffd2mat(tname,oname);
%
% Inputs:
%
%   tname = name of file containing TAPE-format DIMAT MACHINE output
%   oname = (optional) name of file containing DIMAT REFERENCE ORBIT output
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

% check the input/output arguments; handle defaults

if (nargin<1),error('One or two file name input arguments required'),end
if (nargout~=12),error('12 output arguments required'),end
if (~ischar(tname)),error('Twiss file name must be a string'),end
[nrow,ncol]=size(tname);
if (nrow~=1),error('Twiss file name must be a single row'),end
if (exist(tname)~=2),error('Twiss file not found'),end

getorbit=0;
if (nargin>1)
  if (~ischar(oname)),error('Orbit file name must be a string'),end
  [nrow,ncol]=size(oname);
  if (nrow~=1),error('Orbit file name must be a single row'),end
  if (exist(oname)~=2),error('Orbit file not found'),end
  getorbit=1;
end

% prepare the twiss file

awkname=which('dim2mat_twiss.awk');
cmd=['gawk -f ',awkname,' ',tname,' > tmp.txt'];
[status,result]=system(cmd);

% use the mex-file to download the twiss data

[tt,keyw,name,L,P,A,etyp,E,fnam,twss,orbt,S]=xtfft2mat_mex('tmp.txt');

% set the string attribute for the appropriate arrays

K=char(keyw);
N=char(name);
T=char(etyp);
FDN=char(fnam);

if (getorbit)
  
% prepare the orbit file

  awkname=which('dim2mat_orbit.awk');
  cmd=['gawk -f ',awkname,' ',oname,' > tmp.txt'];
  [status,result]=system(cmd);

% download the orbit

  tmp=load('tmp.txt');
  orbt(:,1)=tmp(:,1);
  orbt(:,3)=tmp(:,2);
end

% cleanup

if (exist('tmp.txt')==2),delete tmp.txt,end

% done ...

return
