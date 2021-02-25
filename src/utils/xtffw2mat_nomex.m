function [tt,K,N,L,P,A,T,E,FDN,chrom,orbt,S]=xtffw2mat_nomex(fname)
%
% [tt,K,N,L,P,A,T,E,FDN,chrom,orbt,S]=xtffw2mat_nomex(fname);
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
%   chrom = (wx,phix,dmux,ddx,ddpx,wy,phiy,dmuy,ddy,ddpy)
%   orbt  = orbit (x,px,y,py,t,pt)
%   S     = suml

% open the XTFF file
fid=fopen(fname);
if (fid==-1)
  error(['  Failed to open ',fname])
end

% read until CHROM is found ... error if not found
found=0;
while(~found)
  line=fgetl(fid);
  if (~ischar(line)),break,end
  if ((length(line)>16)&&strcmp(line(9:16),'   CHROM'))
    found=1;
  end
end
if (~found),error('CHROM not found'),end

% read in the run title
tt=deblank(fgetl(fid));

% read in the INITIAL data
p=zeros(1,8);
w=zeros(1,10);
o=zeros(1,4);
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
w(1)=str2double(line(1:16));
w(2)=str2double(line(17:32));
w(3)=str2double(line(33:48));
w(4)=str2double(line(49:64));
w(5)=str2double(line(65:80));
line=fgetl(fid); % line 4
w(6)=str2double(line(1:16));
w(7)=str2double(line(17:32));
w(8)=str2double(line(33:48));
w(9)=str2double(line(49:64));
w(10)=str2double(line(65:80));
line=fgetl(fid); % line 5
o(1)=str2double(line(1:16));
o(2)=str2double(line(17:32));
o(3)=str2double(line(33:48));
o(4)=str2double(line(49:64));
S=str2double(line(65:80));

% define arrays
P=p;
chrom=w;
orbt=o;

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
  if (length(line)<105),line=[line,blanks(105-length(line))];end
  p(4)=str2double(line(1:16));
  p(5)=str2double(line(17:32));
  p(6)=str2double(line(33:48));
  p(7)=str2double(line(49:64));
  p(8)=str2double(line(65:80));
  FDN=[FDN;line(82:105)];
  line=fgetl(fid); % line 3
  w(1)=str2double(line(1:16));
  w(2)=str2double(line(17:32));
  w(3)=str2double(line(33:48));
  w(4)=str2double(line(49:64));
  w(5)=str2double(line(65:80));
  line=fgetl(fid); % line 4
  w(6)=str2double(line(1:16));
  w(7)=str2double(line(17:32));
  w(8)=str2double(line(33:48));
  w(9)=str2double(line(49:64));
  w(10)=str2double(line(65:80));
  line=fgetl(fid); % line 5
  o(1)=str2double(line(1:16));
  o(2)=str2double(line(17:32));
  o(3)=str2double(line(33:48));
  o(4)=str2double(line(49:64));
  S=[S;str2double(line(65:80))];
  P=[P;p];
  chrom=[chrom;w];
  orbt=[orbt;o];
end

% close the XTFF file
fclose(fid);

end