function planview(fname,z1,z2,sfac)

if (~exist('z1','var')),z1=0;end
if (~exist('z2','var')),z2=0;end
if (~exist('sfac','var')),sfac=1;end

[tt,K,N,L,P,A,T,E,FDN,coor,S]=xtffs2mat(fname);
if (z1==z2)
  id=[1:length(S)]';
else
  id=find((coor(:,3)>=min([z1,z2]))&(coor(:,3)<=max([z1,z2])));
end

plot(coor(id,3),coor(id,1),'b-')

wa=0.05*sfac;
wb=0.125*sfac;
wq=0.25*sfac;
ws=0.2*sfac;
wo=0.15*sfac;

ida=id(find_name('LCAV',K(id,:)));
if (~isempty(ida))
  hold on
  for m=1:length(ida)
    ida1=ida(m)-1;
    ida2=ida(m);
    coor0=mean(coor([ida1;ida2],:));
    z0=coor0(3);
    x0=coor0(1);
    lu=L(ida2)/2;
    ld=lu;
    z=[-lu,ld,ld,-lu,-lu];
    x=(wa/2)*[-1,-1,1,1,-1];
    S=sin(coor0(4));
    C=cos(coor0(4));
    t=[C,-S;S,C]*[z;x];
    z=z0+t(1,:)';
    x=x0+t(2,:)';
    plot(z,x,'b-')
  end
  hold off
end

idb=id(find_name('SBEN',K(id,:)));
hold on
for m=1:length(idb)/2
  idb0=idb(2*m-1);
  idb1=idb(2*m-1)-1;
  idb2=idb(2*m);
  z0=coor(idb0,3);
  x0=coor(idb0,1);
  lu=L(idb0);
  ld=L(idb2);
  z=[-lu,ld,ld,-lu,-lu];
  x=(wb/2)*[-1,-1,1,1,-1];
  S=sin(coor(idb0,4));
  C=cos(coor(idb0,4));
  t=[C,-S;S,C]*[z;x];
  z=z0+t(1,:)';
  x=x0+t(2,:)';
  plot(z,x,'b-')
end
hold off

idq=id(find_name('QUAD',K(id,:)));
hold on
for m=1:length(idq)/2
  idq0=idq(2*m-1);
  idq1=idq(2*m-1)-1;
  idq2=idq(2*m);
  z0=coor(idq0,3);
  x0=coor(idq0,1);
  lq=L(idq0);
  z=lq*[-1,1,1,-1,-1];
  x=(wq/2)*[-1,-1,1,1,-1];
  S=sin(coor(idq0,4));
  C=cos(coor(idq0,4));
  t=[C,-S;S,C]*[z;x];
  z=z0+t(1,:)';
  x=x0+t(2,:)';
  plot(z,x,'b-')
end
hold off

ids=id(find_name('SEXT',K(id,:)));
hold on
for m=1:length(ids)/2
  ids0=ids(2*m-1);
  ids1=ids(2*m-1)-1;
  ids2=ids(2*m);
  z0=coor(ids0,3);
  x0=coor(ids0,1);
  ls=L(ids0);
  z=ls*[-1,1,1,-1,-1];
  x=(ws/2)*[-1,-1,1,1,-1];
  S=sin(coor(ids0,4));
  C=cos(coor(ids0,4));
  t=[C,-S;S,C]*[z;x];
  z=z0+t(1,:)';
  x=x0+t(2,:)';
  plot(z,x,'b-')
end
hold off

ido=id(find_name('OCTU',K(id,:)));
hold on
for m=1:length(ido)/2
  ido0=ido(2*m-1);
  ido1=ido(2*m-1)-1;
  ido2=ido(2*m);
  z0=coor(ido0,3);
  x0=coor(ido0,1);
  lo=L(ido0);
  z=lo*[-1,1,1,-1,-1];
  x=(wo/2)*[-1,-1,1,1,-1];
  S=sin(coor(ido0,4));
  C=cos(coor(ido0,4));
  t=[C,-S;S,C]*[z;x];
  z=z0+t(1,:)';
  x=x0+t(2,:)';
  plot(z,x,'b-')
end
hold off

ylabel('X (m)')
xlabel('Z (m)')
