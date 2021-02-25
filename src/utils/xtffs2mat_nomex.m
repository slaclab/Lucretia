function [tt,K,N,L,P,A,T,E,FDN,coor,S]=xtffs2mat_nomex(fname)
%
% [tt,K,N,L,P,A,T,E,FDN,coor,S]=xtffs2mat_nomex(fname);
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
%   coor = survey coordinates (X,Y,Z,yaw,pitch,roll)
%   S    = suml

% open the XTFF file
fid=fopen(fname);
if (fid==-1)
  error(['  Failed to open ',fname])
end

% read in the header ... check that XTFF file is a SURVEY file
line=fgetl(fid);
xtff=line(9:16);
if (~strcmp(xtff,'  SURVEY'))
  error(['  Unexpected XTFF type (',xtff,') encountered ... abort'])
end

% read in the run title
tt=deblank(fgetl(fid));

% read in the INITIAL data
p=zeros(1,8);
line=fgetl(fid); % line 1
K=line(1:4);
N=line(5:20);
L=str2double(line(21:32));
p(1)=str2double(line(33:48));
p(2)=str2double(line(49:64));
p(3)=str2double(line(65:80));
A=str2double(line(81:96));
T=line(98:113);
E=str2double(line(115:130));
line=fgetl(fid); % line 2
if (length(line)<105),line=[line,blanks(105-length(line))];end
p(4)=str2double(line(1:16));
p(5)=str2double(line(17:32));
p(6)=str2double(line(33:48));
p(7)=str2double(line(49:64));
p(8)=str2double(line(65:80));
FDN=line(82:105);
line=fgetl(fid); % line 3
x=str2double(line(1:16));
y=str2double(line(17:32));
z=str2double(line(33:48));
S=str2double(line(49:64));
line=fgetl(fid); % line 4
yaw=str2double(line(1:16));
pitch=str2double(line(17:32));
roll=str2double(line(33:48));

% define arrays
P=p;
coor=[x,y,z,yaw,pitch,roll];

% read in the data ... break at end of the file
while(1)
  line=fgetl(fid); % line 1
  if (isempty(line)||~isletter(line(1:1))),break,end
  K=[K;line(1:4)];
  N=[N;line(5:20)];
  L=[L;str2double(line(21:32))];
  p(1)=str2double(line(33:48));
  p(2)=str2double(line(49:64));
  p(3)=str2double(line(65:80));
  A=[A;str2double(line(81:96))];
  T=[T;line(98:113)];
  E=[E;str2double(line(115:130))];
  line=fgetl(fid); % line 2
  if (length(line)<105), line=[line,blanks(105-length(line))]; end
  p(4)=str2double(line(1:16));
  p(5)=str2double(line(17:32));
  p(6)=str2double(line(33:48));
  p(7)=str2double(line(49:64));
  p(8)=str2double(line(65:80));
  FDN=[FDN;line(82:105)];
  line=fgetl(fid); % line 3
  x=str2double(line(1:16));
  y=str2double(line(17:32));
  z=str2double(line(33:48));
  S=[S;str2double(line(49:64))];
  line=fgetl(fid); % line 4
  yaw=str2double(line(1:16));
  pitch=str2double(line(17:32));
  roll=str2double(line(33:48));
  P=[P;p];
  coor=[coor;x,y,z,yaw,pitch,roll];
end

% close the XTFF file
fclose(fid);

end