function cm=rainbow(Ncm)
%
% cm=rainbow(Ncm);
%
% Generate the "rainbow" colormap with N+1 rows, where N is the nearest
% multiple of 5 which is greater than Ncm (if not supplied, Ncm=5; if
% supplied, Ncm must be >= 5)

if (nargin==0)
  N=5;
else
  N=5*ceil(Ncm/5);
  N=max([N,5]);
end
dcm1=2.5/N;
dcm2=2*dcm1;

Nr=N/5;
cm=[1,0,0.5; ...
    ones(Nr,1),zeros(Nr,1),[0.5-dcm1:-dcm1:0]'; ...
    ones(Nr,1),[dcm2:dcm2:1]',zeros(Nr,1); ...
    [1-dcm2:-dcm2:0]',ones(Nr,1),zeros(Nr,1); ...
    zeros(Nr,1),[1-dcm2:-dcm2:0]',[dcm2:dcm2:1]'; ...
    [dcm1:dcm1:0.5]',zeros(Nr,1),[1-dcm1:-dcm1:0.5]'];

return
