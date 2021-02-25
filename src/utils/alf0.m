function [a0p,a0m]=alf0(R,b0)
M=r2m(R);
a=M(1,3)*ones(size(b0))./b0;
b=M(1,2)*ones(size(b0));
c=M(1,1)*b0+M(1,3)*ones(size(b0))./b0-b0;
a0p=(-b+sqrt(b.^2-4*a.*c))./(2*a);
a0m=(-b-sqrt(b.^2-4*a.*c))./(2*a);
