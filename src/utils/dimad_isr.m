function [sigx,sigy,DX,DPX,DY,DPY,dexn,deyn]=dimad_isr(energy,exn0,eyn0)
%
% [sigx,sigy,DX,DPX,DY,DPY,dexn,deyn]=dimad_isr(energy,exn0,eyn0);
%
% Compute relative emittance growth from DIMAD output rays

Me=0.51099906e-3;
egamma=energy/Me;

Rays=load('fort.8');
x=Rays(:,2);
px=Rays(:,3);
y=Rays(:,4);
py=Rays(:,5);
z=Rays(:,6);
dp=Rays(:,7);

coef=polyfit(dp,x,1);DX=coef(1);
coef=polyfit(dp,px,1);DPX=coef(1);
sigx=cov([x-DX*dp,px-DPX*dp]);
exn=egamma*sqrt(det(sigx));
dexn=100*(exn-exn0)/exn0;

coef=polyfit(dp,y,1);DY=coef(1);
coef=polyfit(dp,py,1);DPY=coef(1);
sigy=cov([y-DY*dp,py-DPY*dp]);
eyn=egamma*sqrt(det(sigy));
deyn=100*(eyn-eyn0)/eyn0;

disp(' ')
disp(sprintf('   Energy   = %6.1f GeV',energy))
disp(sprintf('   Sigma X  = %6.1f nm',1e9*std(x)))
disp(sprintf('   Sigma PX = %6.1f um',1e6*std(px)))
disp(sprintf('   Sigma Y  = %6.1f nm',1e9*std(y)))
disp(sprintf('   Sigma PY = %6.1f um',1e6*std(py)))
disp(' ')
disp(sprintf('   DX  = %7.3f mm',1e3*DX))
disp(sprintf('   DPX = %7.3f mrad',1e3*DPX))
disp(sprintf('   Horizontal emittance growth = %6.1f %%',dexn))
disp(' ')
disp(sprintf('   DY  = %7.3f mm',1e3*DY))
disp(sprintf('   DPY = %7.3f mrad',1e3*DPY))
disp(sprintf('   Vertical emittance growth = %6.1f %%',deyn))
disp(' ')
