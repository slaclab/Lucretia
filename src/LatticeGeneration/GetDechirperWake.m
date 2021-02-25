function [ZSR, TSR] = GetDechirperWake(a,p,h)
% [Wz, Wt] = chirpWake(a,p,h)
% Generate longitudinal and transverse wake estimates for periodic
% dechirper structure. From PRSTAB 18, 010702 (2015): Zhang et. al.
% a=half-gap
% p=period
% h=periodic struture height
% Conditions assumed:
% h/p > 0.8
% p,h < a
% applicable for bunch length << a
% -----------
% Returns structures suitable for adding to Lucretia WF global structure
% ZSR= longitudinal wake Lucretia structure [K= V/C/m]
% TSR= dipole wake Lucretia structure [K= V/C/m^2]

Z0=377;
clight=2.99792458e8;

b1=0.1483;
b2=0.1418;
b3=-0.0437;
b4=0.146;
b5=0.5908;
c1=1.7096;
c2=-0.5026;
d1=3.2495;
d2=-9.183;
d3=10.223;
k=(1/a)*(c1/sqrt(h/a) + c2);
F=b1*(1-p/a)*(1-h/a) + b2*(1-h/a) + b3*(1-p/a)^2 + b4*(1-p/a) + b5;
Q=d1*(h/a)^2 + d2*(h/a) + d3;

zmax=pi/k;
z=linspace(0,zmax,1000);

wz=(pi^2/16)*((Z0*clight)/(pi*a^2)).*cos(k.*z).*F.*exp(-(k.*z)/(2*Q));
wt=((2*Z0*clight)/(pi*a^4))*(pi/4)^4;

ZSR.z=z; ZSR.K=wz; ZSR.BinWidth=0.01;
TSR.z=z; TSR.K=ones(size(z)).*wt; TSR.BinWidth=0.01;
