function b = bfact(Beam,ibunch,dz,sf,zcut)
%BFACT Calculate longitudinal bunching factor for provided bunch

% Phys constants
qe=1.60217653e-19; % electron charge / C
clight=299792458; % speed of light in vacuum / ms^-1
me=9.10938356e-31*clight^2/qe/1e9; % electron rest mass / GeV

% Generate current histogram with given bin width
Q = Beam.Bunch(ibunch).Q(~Beam.Bunch(ibunch).stop) ;
E = Beam.Bunch(ibunch).x(6,~Beam.Bunch(ibunch).stop) ;
z = Beam.Bunch(ibunch).x(5,~Beam.Bunch(ibunch).stop) ;
if isempty(dz)
  [~,ed,BIN]=histcounts(z);
  nbin=length(ed);
  dz=abs(diff(ed(1:2)));
else
  nbin = ceil( range(z) / dz ) ;
  [~,ed,BIN]=histcounts(z,nbin);
end
zv=ed(1:end-1)+diff(ed(1:2))/2;
qv=accumarray(BIN',Q');
fprintf('nbins= %d\n',nbin);

% Cut 1% charge beam tails
ns=cumsum(qv); n1=find(ns<ns(end)*0.01,1,'last'); n2=find(ns>ns(end)*0.99,1);
zv=zv(n1:n2); qv=qv(n1:n2);

% Apply smoothing function
if sf<0
  qvs=smoothn(qv,'robust');
elseif sf>0
  qvs=smoothn(qv,sf);
else
  qvs=qv;
end

% Only fit / calc bunching factor over zcut
if ~exist('zcut','var')
  zcut=[min(zv) max(zv)];
end
zi=zv>=zcut(1) & zv<=zcut(2) ;

% De-trend data
qvs=qvs(:)';
[P,~,MU]=polyfit(zv(zi),qvs(zi),9);
qvf=polyval(P,zv(zi),[],MU);
qvt=qvs(zi)-qvf;

% Calculate bunching factor
binwid=abs(diff(zv(1:2)));
relgamma=mean(E)/me;
relbeta=sqrt(1-relgamma^-2);
It=qvt./(binwid/(relbeta*clight)); % convert histogram normalization to current in A
Is=qvs./(binwid/(relbeta*clight));
Iv=qv./(binwid/(relbeta*clight));
If=qvf./(binwid/(relbeta*clight));
L=range(zv(zi));
kmin=(2*pi)/dz/2;
kmax=(2*pi)/L/2;
k=linspace(kmin,kmax,2^14)';
b = (1/L) * sum( binwid .* It .* exp(-1i.*k.*zv(zi)),2 );


% Diagnostics plots
figure
subplot(3,1,1)
plot(zv,Iv,zv,Is,zv(zi),If);xlabel('Z [m]'); ylabel('I [A]'); grid on
ax=axis;
subplot(3,1,2)
plot(zv(zi),It);xlabel('Z [m]'); ylabel('\DeltaI(z) [A]'); grid on
ax2=axis; axis([ax(1:2) ax2(3:4)]);
subplot(3,1,3)
semilogx((2*pi)./k,abs(b));xlabel('\lambda [m]'); ylabel('Bunching factor |b(k)|'); grid on