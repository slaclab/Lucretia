function [x,px,r21,y,py,r43]=beam(E,exn,eyn,bx,ax,by,ay,units);
g=E/0.51099906e-3;
ex=exn/g;
ey=eyn/g;
gx=(1+ax^2)/bx;
gy=(1+ay^2)/by;
sigx=ex*[bx,-ax;-ax,gx];
sigy=ey*[by,-ay;-ay,gy];
sig=[sigx,zeros(2);zeros(2),sigy];
x=sqrt(sig(1,1));
px=sqrt(sig(2,2));
r21=sig(2,1)/sqrt(sig(1,1)*sig(2,2));
y=sqrt(sig(3,3));
py=sqrt(sig(4,4));
r43=sig(4,3)/sqrt(sig(3,3)*sig(4,4));
disp(' ')
disp(sprintf('%.6f %.6f %.6f %.6f',units*sqrt(diag(sig))))
disp(sprintf('%.6f 0 0 0 0 %.6f',r21,r43))
disp(' ')
