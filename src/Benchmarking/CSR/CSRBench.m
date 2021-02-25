% CSRBENCH Benchmark CSR code
%  Benchmark against example case in paper "CSR Wake for a short magnet in ultrarelativistic limit", EPAC2002, Stupakov, Emma

global BEAMLINE

use2d=true; % Use 2D CSR model?

% Setup single bend and drift section, readout CSR fields at fixed locations to compare against results of paper
lb = [2 5 10 14 18 50].*1e-2; % Distances into bend to readout fields
ld = [0 2 5 10 20 50].*1e-2; % Distances into drift past bend to access fields

% Beam and magnet description, constants etc
I=InitCondStruc;
I.Momentum=0.15;
I.Q=1e-9;
I.sigz=50e-6;
I.x.NEmit=1e-6;
I.y.NEmit=1e-6;
Rb=1.5; % Bend radius
CLIGHT=299792458; % speed of light in m/s
qe=1.60217662e-19; % electron charge / C

% Make beam
nray=1e5;
Beam=MakeBeam6DGauss(I,nray,10,1);
% Ensure span of macro particles extends +/-10 sigma
zlinear=linspace(-10*I.sigz,10*I.sigz,200);
zind=randperm(nray,200);
Beam.Bunch.x(5,zind)=zlinear;

% Generate beamline and track for different readout locations
clear applyCSR
Theta = lb(end) / Rb ; % bend angle
B = Theta * (I.Momentum.*1e9) ./ CLIGHT ;
BEAMLINE{1}=SBendStruc(lb(end),B,Theta,[0 Theta],0,0.2,0.5,0,'B1');
BEAMLINE{1}.P=I.Momentum;
BEAMLINE{1}.TrackFlag.CSR=-1;
BEAMLINE{1}.TrackFlag.Split=ceil(lb(end)/0.01);
BEAMLINE{1}.TrackFlag.CSR_SmoothFactor=1;
BEAMLINE{1}.TrackFlag.Diagnostics=-1; % Instruct CSR routine not to apply kicks, just calculate and store fields
BEAMLINE{1}.TrackFlag.CSR_DriftSplit=100;
BEAMLINE{2,1}=DrifStruc(0.5,'D1'); BEAMLINE{2}.P=I.Momentum;
BEAMLINE{2}.TrackFlag.Diagnostics=-1;
if use2d
  BEAMLINE{1}.TrackFlag.CSR_2D=1;
  BEAMLINE{1}.TrackFlag.CSR_USEGPU=true;
end
SetSPositions(1,2,0);
[~,bo]=TrackThru(1,2,Beam,1,1);
dat=applyCSR('GetDiagData');
figure;
subplot(1,2,1);
for iL=1:length(lb)
  [~,ind]=min(abs(dat.S-lb(iL)));
  if use2d
    plot((dat.z{ind}-mean(dat.z{ind}))./I.sigz,1e-6.*dat.Ws{ind},'.'),xlabel('Z/\sigma_z');ylabel('qW(s) [MV/m]'); hold on
  else
    plot((dat.z{ind}-mean(dat.z{ind}))./I.sigz,1e-6.*dat.W{ind},'.'),xlabel('Z/\sigma_z');ylabel('qW(s) [MV/m]'); hold on %#ok<*UNRCH>
  end
end
hold off
legend({'2cm' '5cm' '10cm' '14cm' '18cm' '50cm'})
ax=axis; axis([-10 10 ax(3:4)]);
subplot(1,2,2);
if use2d
  for iL=1:length(lb)
    [~,ind]=min(abs(dat.S-lb(iL)));
    plot((dat.z{ind}-mean(dat.z{ind}))./I.sigz,dat.Wx{ind},'.'),xlabel('Z/\sigma_z');ylabel('qW(x) [rad/m]'); hold on
  end
  hold off
  legend({'2cm' '5cm' '10cm' '14cm' '18cm' '50cm'})
  ax=axis; axis([-10 10 ax(3:4)]);
else
  for iL=1:length(ld)
    [~,ind]=min(abs((dat.S-0.5)-ld(iL)));
    plot((dat.z{ind}-mean(dat.z{ind}))./I.sigz,1e-6*dat.W{ind},'.'),xlabel('Z/\sigma_z');ylabel('qW(s) [MV/m]'); hold on
  end
  hold off
  legend({'0cm' '2cm' '5cm' '10cm' '20cm' '50cm'})
  ax=axis; axis([-10 10 ax(3:4)]);
end