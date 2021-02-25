function [stat, Bcorrect, skewps] = correct_coupling(otruse, BEAMLINE, PS, ~, FL, method)
FlHwUpdate;
stat{1}=1;
warning('off','optimlib:fwdFinDiffInsideBnds:StepReduced')
if ~exist('method','var')
  method=1;
end

%intensities to scan to obtain response matrix
intensities=[-18 -10 -5 0 5 10 18];

%intensities to B
IBlookup=[-20 -16 -12 -8 -4 0 4 8 12 16 20;
  -0.2227 -0.1779 -0.1331 -0.0884 -0.0439 0 0.0439 0.0884 0.1331 0.1779 0.2227];
B = interp1(IBlookup(1,:),IBlookup(2,:),intensities);

%Find OTRs
for i = 0:3
  otrind(i+1) = findcells(BEAMLINE,'Name',['OTR',num2str(i),'X']);
end

%Find skew indices
for i = 1:4
  temp = findcells(BEAMLINE,'Name',['QK',num2str(i),'X']);
  skewind(i)=temp(2);
end

%find the skewpowersupplies indices
for n=1:4
  skewps(n)=BEAMLINE{skewind(n)}.PS;
end

% Get OTR data
l=0;
dd=dir(fullfile('userData','emit2dOTRp*'));
ld=load(fullfile('userData',dd(end).name));
Dx=ld.data.DX;
Dy=ld.data.DY;
dp=ld.data.dp;
for i=find(otruse)
  l=l+1;
  dt=ld.rawotrdata{l};
  sig13(l)=dt.sig13.*1e-12-sign(dt.sig13)*abs(Dx(l)*Dy(l)*dp^2);
  %   sig13(l)=dt.sig13;
  sig11(l)=(dt.projx.*1e-6)^2;
  sig33(l)=(dt.projy.*1e-6)^2;
end

FL.SimModel.Initial_IEX.Momentum=FL.SimModel.Initial.Momentum;
beam_EXT=MakeBeam6DGauss(FL.SimModel.Initial_IEX,10001,5,0);
if method==2
  % Get Initial beam matrix that fits data
  [~,S]=GetBeamPars(beam_EXT,1);
  for ir=1:4
    [~,RO{ir}]=RmatAtoB(FL.SimModel.extStart,otrind(ir));
  end
  x0=[S(1,1) S(1,2) S(1,3) S(1,4) S(2,2) S(2,3) S(2,4) S(3,3) S(3,4) S(4,4)];
  SF=10;
  minx=[S(1,1)/SF -1e-6 -1e-6 -1e-6 S(2,2)/SF -1e-6 -1e-6 S(3,3)/SF -1e-6 S(4,4)/SF]';
  maxx=[S(1,1)*SF   1e-6  1e-6  1e-6 S(2,2)*SF   1e-6  1e-6 S(3,3)*SF   1e-6 S(4,4)*SF]';
  opts = optimset('TolX',1e-7,'MaxFunEvals',50000,'MaxIter',300) ;
  problem = createOptimProblem('fmincon','x0',x0,'objective',@(x) minS(x,S,RO,sig11,sig33,sig13),...
    'nonlcon',@(x) funNLCON(x),'lb',minx,'ub',maxx,'options',opts);
  %   xmin=fmincon(@(x) minS(x,S,RO,sig11,sig33,sig13),x0,[],[],[],[],minx,maxx,@(x) funNLCON(x),...
  %     optimset('Display','iter','TolX',1e-7,'MaxFunEvals',50000,'MaxIter',300));
  gs = GlobalSearch('Display','iter','TolX',1e-16) ;
  %   ms = MultiStart('Display','iter','TolX',1e-16);
  [xmin,ff]=run(gs,problem);
  S(1,1)=xmin(1); S(1,2)=xmin(2); S(1,3)=xmin(3); S(1,4)=xmin(4);
  S(2,1)=xmin(2); S(2,2)=xmin(5); S(2,3)=xmin(6); S(2,4)=xmin(7);
  S(3,1)=xmin(3); S(3,2)=xmin(6); S(3,3)=xmin(8); S(3,4)=xmin(9);
  S(4,1)=xmin(4); S(4,2)=xmin(7); S(4,3)=xmin(9); S(4,4)=xmin(10);
  % Get emittances this implies
  [nx, ny] = GetNEmitFromSigmaMatrix( FL.SimModel.Initial.Momentum, S );
  [nx_int, ny_int] = GetNEmitFromSigmaMatrix( FL.SimModel.Initial.Momentum, S, 'normalmode' );
  gamma=FL.SimModel.Initial.Momentum/0.511e-3;
  fprintf('EMIT_X: %.2f (nm) EMIT_Y: %.2f (pm)\n',1e9*nx/gamma,1e12*ny/gamma)
  fprintf('INTRINSIC: EMIT_X: %.2g EMIT_Y: %.2g\n',nx_int/gamma,ny_int/gamma)
  % Get skew quad settings that will minimize emittance
  Bcorrect=fmincon(@(x) minemit(x,skewps,S,FL.SimModel.extStart,otrind(1)),[PS(skewps).Ampl],...
    [],[],[],[],[-0.2227 -0.0557 -0.0557 -0.2227],[0.2227 0.0557 0.0557 0.2227],[],optimset('Display','iter'));
  return
end

latticeversion=FL.SimModel.opticsVersion;
latticename=FL.SimModel.opticsName;
filename=sprintf('RespMatrix4CoupCorr_%s_%s',latticeversion,latticename);
%load response matrix it if exists
% if exist(sprintf('/home/atf2-fs/ATF2/FlightSim/userData/%s.mat',filename),'file')
%   load(sprintf('/home/atf2-fs/ATF2/FlightSim/userData/%s.mat',filename),'R');
% else
%GET for different B for each of the 4 skews the coupling terms in each OTR
for i=1:4%for the different skews
  for j=1:length(B)%for the different B
    
    %all skews off except i
    for n=1:4
      initPS(n)=PS(skewps(n)).Ampl;
      PS(skewps(n)).Ampl=0;
      PS(skewps(n)).SetPt=0;
    end
    PS(skewps(i)).Ampl=1;
    PS(skewps(i)).SetPt=1;
    
    %put B in the skew
    BEAMLINE{skewind(i)-1}.B=B(j);
    BEAMLINE{skewind(i)}.B=B(j);
    
    [~,beam_OTR0]=TrackThru(FL.SimModel.extStart,otrind(1),beam_EXT,1,1,0);
    [~,sigma_OTR0] = GetBeamPars(beam_OTR0,1);sigma_OTR0=sigma_OTR0(1:4,1:4);
    [~,beam_OTR1]=TrackThru(otrind(1),otrind(2),beam_OTR0,1,1,0);
    [~,sigma_OTR1] = GetBeamPars(beam_OTR1,1);sigma_OTR1=sigma_OTR1(1:4,1:4);
    [~,beam_OTR2]=TrackThru(otrind(2),otrind(3),beam_OTR1,1,1,0);
    [~,sigma_OTR2] = GetBeamPars(beam_OTR2,1);sigma_OTR2=sigma_OTR2(1:4,1:4);
    [~,beam_OTR3]=TrackThru(otrind(3),otrind(4),beam_OTR2,1,1,0);
    [~,sigma_OTR3] = GetBeamPars(beam_OTR3,1);sigma_OTR3=sigma_OTR3(1:4,1:4);
    coup(i,1,j)=sigma_OTR0(1,3);
    coup(i,2,j)=sigma_OTR1(1,3);
    coup(i,3,j)=sigma_OTR2(1,3);
    coup(i,4,j)=sigma_OTR3(1,3);
  end
 end
  
  % Restore PS settings
  for n=1:4
    PS(skewps(n)).Ampl=initPS(n);
    PS(skewps(n)).SetPt=initPS(n);
  end
  
  %Do linear fit and build response matrix
  for i2=1:4 %for each skew quad
    for k=1:4
      temp(1:length(B))=coup(i2,k,:);
      c=polyfit(B(:),temp(:),1);
      R(k,i2)=c(1);
    end
  end
  save(sprintf('/home/atf2-fs/ATF2/FlightSim/userData/%s.mat',filename),'R')


% sig13=sig13*1e-12;%stored data was in um

if any(isnan(sig13))
  stat{1}=-1;
  stat{2}=sprintf('getOTRsize gave a NaN return');
  return
end

%Calculate INT to correct
Bcorrect=-(R(find(otruse),:)\sig13'); %#ok<FNDSB>

%if any(abs(Bcorrect)>0.2227)%if intensity exceeds +-20A put +-20A
%   Bcorrect(Bcorrect>0.2227)=0.2227;
%   Bcorrect(Bcorrect<-0.2227)=-0.2227;
%end

function [C,Ceq]=funNLCON(x)

Ceq=0;
C=-(x(1)*x(5)-x(2)^2);
if C>0; return; end;
C=-(x(8)*x(10)-x(9)^2);

function ret=minS(x,S,R,sig11,sig33,sig13)

S(1,1)=x(1); S(1,2)=x(2); S(1,3)=x(3); S(1,4)=x(4);
S(2,1)=x(2); S(2,2)=x(5); S(2,3)=x(6); S(2,4)=x(7);
S(3,1)=x(3); S(3,2)=x(6); S(3,3)=x(8); S(3,4)=x(9);
S(4,1)=x(4); S(4,2)=x(7); S(4,3)=x(9); S(4,4)=x(10);

ret=zeros(1,12);
for ind=1:4
  S2=R{ind}*S*R{ind}';
  ret((ind-1)*3+1)=(sig11(ind)-S2(1,1))/sig11(ind);
  ret((ind-1)*3+2)=(sig33(ind)-S2(3,3))/sig33(ind);
  ret((ind-1)*3+2)=(sig13(ind)-S2(1,3))/sig13(ind);
end
ret=sum(ret.^2);

function ret=minemit(x,sqind,S,i1,i2)
global PS
for ips=1:4
  PS(sqind(ips)).Ampl=x(ips);
end
[~,R]=RmatAtoB(i1,i2);
S2=R*S*R';
ret=abs(S2(1,3)+S(2,3)+S(1,4)+S(2,4))/1e-14;
