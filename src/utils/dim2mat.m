function [N,S,twss]=dim2mat(fname)

N=[];
S=[];
twss=[];
fid=fopen(fname);
while (1)
  s=fgetl(fid);
  if (~ischar(s)),break,end
  name=s(2:9);
  betx=str2num(s(16:26));
  alfx=str2num(s(27:37));
  bety=str2num(s(38:48));
  alfy=str2num(s(49:59));
  dx=str2num(s(60:69));
  dpx=str2num(s(70:78));
  dy=str2num(s(79:87));
  dpy=str2num(s(88:96));
  nux=str2num(s(97:105));
  nuy=str2num(s(106:114));
  suml=str2num(s(115:124));
  N=[N;name];
  S=[S;suml];
  twss=[twss;nux,betx,alfx,dx,dpx,nuy,bety,alfy,dy,dpy];
end
fclose(fid);
