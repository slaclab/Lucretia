function [ax0p,ax0m,ay0p,ay0m]=alpha0(Rx,Ry,b0)
Mx=r2m(Rx);My=r2m(Ry);
a=Mx(1,3)*ones(size(b0))./b0;
b=Mx(1,2)*ones(size(b0));
c=Mx(1,1)*b0+Mx(1,3)*ones(size(b0))./b0-b0;
ax0p=(-b+sqrt(b.^2-4*a.*c))./(2*a);
ax0m=(-b-sqrt(b.^2-4*a.*c))./(2*a);
a=My(1,3)*ones(size(b0))./b0;
b=My(1,2)*ones(size(b0));
c=My(1,1)*b0+My(1,3)*ones(size(b0))./b0-b0;
ay0p=(-b+sqrt(b.^2-4*a.*c))./(2*a);
ay0m=(-b-sqrt(b.^2-4*a.*c))./(2*a);
