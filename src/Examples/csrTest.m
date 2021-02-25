function csrTest
% CSRTEST Test generation of 1d and 2d CSR wakes
% 1-d using paper by G. Stupakov and P. Emma in the EPAC2002 paper "CSR Wake for a Short Magnet in Ultrarelativistic Limit"
% 2-d using calculations by Y. Cai: Phys. Rev. Accel. Beams 23, 014402
%  (all as implemented in Lucretia applyCSR function)

% Generate uncorrelated gaussian macro-particle distributions
%  Use example from Y.Cai paper to reconstruct wakes shown in Figs 5 & 6
nray=1e6; % number of macro particles to generate
gamma=500; % normalized beam energy
R=1; % bending radius / m
Lb=0.5; % distance into bend / m
dL=0.01; %#ok<NASGU> % Length of "slice" of bend to use for energy loss calculation / m
Q=1e-9; % Bunch charge / C
PHI=Lb/R; % bend angle
% Particle distribution here is according to Lucretia convention: beam is [x,x',y,y',z,P]
%   RHS co-ordinate system, -ve z being the head of the bunch, particle momenta in GeV
sigma=10e-6; % spherical bunch with rms scale of 50um
beam(1,:)=randn(1,nray).*sigma; % x distribution
beam(2,:)=PHI+randn(1,nray).*1e-12; % x' distribution
beam(3,:)=randn(1,nray).*sigma; % y distribution
beam(4,:)=randn(1,nray).*1e-12; % y' distribution
beam(5,:)=randn(1,nray).*sigma; % z distribution
beam(6,:)=ones(1,nray).*(gamma*0.511e-3); % momentum
qpart=ones(1,nray).*(Q/nray); % use equal weighting here, can use variable charge weighting if desired however

% Simulation parameters
nbin=150;          % histogram binning (<=0: auto, >0: use this number of bins (in both dimensions for 2d))
smoothVal=1;     % smoothing algorithm value (0=auto, larger value-> more smoothing)
gscale=[2 2];    % Scales granularity of 2d grids used for wake calculations. To perform the requried 2d overlap integral, the problem is subdivided into a number of sub-grids to optimize the compute time by providing higher-detail grids for the portions of the 2d integral with fast-changing wake potentials. The gridding requirements change non trivially with bend angle, beam energy and bunch distributions. The CSR algorithm attempts to sensibly auto-scale the integral grids, but a specific problem may benefit from finer gridding or speed up without accuracy loss with coarser gridding depending on the specifics.
                 % The first GSCALE element scales the number of grids used for the fine-detail areas of the 2d integration.
                 % The second GSCALE element scales the number of grids used for the coarser areas of the 2d integration.
                 % The computation time scales approximately as the square of each GSCALE element.
dogpu=false;     % Use GPU for 2-d calculations (requires Nvidia GPU and Matlab parallel computing toolbox)
xmean0=0;        % mean x position of bunch at bend entrance
diagplots=true; % show additional plots from internals of 2-d calculation

%Wake conversion constants
re=2.8179403227e-15; % classical electron radius
qe=1.60217662e-19; % electron charge
cv=((Q/qe).*re)./gamma; %#ok<NASGU> % conversion factor for wakes-> relative energy loss

% Generate longitudinal grid
zvals=-beam(5,:); % make -ve z the tail of the beam for this calculation
zvals=zvals-mean(zvals); % locally centered z distribution
zmin=min(zvals); %#ok<NASGU>
zmax=max(zvals); %#ok<NASGU>
if nbin<=0 % Use automatic binning
  [~,z,bininds]=histcounts(zvals);
  nbin=length(z)-1;
  z=z(1:end-1)+diff(z(1:2))/2;
else % user-defined binning
  [~,z,bininds]=histcounts(zvals,nbin);
  z=z(1:end-1)+diff(z(1:2))/2;
end
[Z, ZSP]=meshgrid(z,z);

% Bin beam particle longitudinal direction and take derivative
q = accumarray(bininds',qpart')';
bw=abs(z(2)-z(1));
q=q./bw; 
if smoothVal==0
  q=smoothn(q,'robust','MaxIter',10000);
else
  q=smoothn(q,smoothVal,'MaxIter',10000);
end
dq=gradient(q,bw);

% 1-d wake potential calculation
SL=(R*PHI^3)/24;
% Loop over particle distribution and form wakefield function and
% calculate energy loss for each bin
ZINT=zeros(nbin,1);
for is=1:nbin
  isp=z>=(z(is)-SL) & (1:length(z))<is ;
  ZINT(is)=sum((1./(z(is)-z(isp)).^(1/3)).*dq(isp).*bw);
end
IND1=abs(Z-(ZSP-(R*PHI^3)/6));
IND2=abs(Z-(ZSP-(R*PHI^3)/24));
[~, I1]=min(IND1,[],2);
[~, I2]=min(IND2,[],2);
W=-(4/(R*PHI)).*q(I1)' + (4/(R*PHI)).*q(I2)' + (2/((3*R^2)^(1/3))).*ZINT;
% dE=-cv.*(W'.*dL)./Q; % dE/E kick for each macro particle

% 2-d wake potential calculation
[dE2d,dXP,Ws,Wx,zb,xb]=csr2dfun(gscale,R,PHI,gamma,beam,qpart,nbin,smoothVal,diagplots,xmean0,dogpu); %#ok<ASGLU>
% dE2d = dE2d .* dL ; % dE/E kick for each macro particle
% dXP = dXP .* dL ; % x kick for each macro particle

% Plot wake functions
[~,xi0]=min(abs(xb));
[~,xip1]=min(abs(xb-sigma));
[~,xim1]=min(abs(xb+sigma));
figure
subplot(1,2,1), plot(z./sigma,-W/Q,'m.');
hold on; plot(zb./sigma,Ws(xi0,:),'b'); plot(zb./sigma,Ws(xip1,:),'r'); plot(zb./sigma,Ws(xim1,:),'k');
grid
legend({'1D' '2D x=0' '2D x=+\sigma_x' '2D x=-\sigma_x'});
xlabel('z / \sigma_z'); ylabel('W_s [m^{-2}]');
subplot(1,2,2), plot(zb./sigma,Wx(xi0,:),'b'); hold on; plot(zb./sigma,Wx(xip1,:),'r'); plot(zb./sigma,Wx(xim1,:),'k');
grid
legend({'x=0' 'x=+\sigma_x' 'x=-\sigma_x'});
xlabel('z / \sigma_z'); ylabel('W_x [m^{-2}]');
return

function [dE,dXP,Ws,Wx,zb,xb]=csr2dfun(gscale,R,theta,gamma,beam,beamQ,nbin,smoothVal,dodiag,xmean0,usegpu)
% Calculate 2D CSR wake function (in x, z planes)
%  - Implements calculations by Y. Cai: Phys. Rev. Accel. Beams 23, 014402

% Some constants
beta=sqrt(1-gamma^-2); % lorentz beta-function
Q=sum(beamQ); % total beam charge [C]
re=2.8179403227e-15; % classical electron radius
qe=1.60217662e-19; % electron charge
cv=((Q/qe).*re)./gamma; % conversion factor for wakes-> relative energy loss

% Beam should be centered and this calculation assumes head at +ve z values
beam(5,:)=beam(5,:)-median(beam(5,:)); beam(5,:)=-beam(5,:);
beam(1,:)=beam(1,:)-xmean0;

% Assumed co-ordinate system for horizontal plane is with +ve x being the
% outside of the bend direction. Lucretia co-ordinate system is a RHS
% co-ordinate system with +ve z as beam direction always. A negative bend
% angle then requires a change in x co-ordinate.
if theta<0
  beam(1,:)=-beam(1,:);
end

% Perform 2-D current-weighted histogram
if nbin(1)<=0 % use automatic binning
  [~,zb,xb,bcz,bcx]=histcounts2(beam(5,:),beam(1,:));
elseif length(nbin)==1 % use user-specified number of bins same in both planes
  [~,zb,xb,bcz,bcx]=histcounts2(beam(5,:),beam(1,:),nbin);
else % use user-specified number of bins in both planes
  n1=linspace(min(beam(5,:)),max(beam(5,:)),nbin(1)+1);
  n2=linspace(min(beam(1,:)),max(beam(1,:)),nbin(2)+1);
  [~,zb,xb,bcz,bcx]=histcounts2(beam(5,:),beam(1,:),n1,n2);
end
zb=zb(1:end-1)+abs(diff(zb(1:2))); xb=xb(1:end-1)+abs(diff(xb(1:2)));
q2 = accumarray([bcz(:) bcx(:)],beamQ)';
% Smooth 2-d current histogram
if smoothVal>0 % user-requested smoothing parameter
  q2=smoothn(q2,smoothVal,'MaxIter',10000);
else % auto smoothing
  q2=smoothn(q2,'robust','MaxIter',10000);
end
if length(xb)>1
  bwx=abs(diff(xb(1:2)));
else
  bwx=max(beam(1,:))-min(beam(1,:));
end
if length(zb)>1
  bwz=abs(diff(zb(1:2)));
else
  bwz=max(beam(5,:))-min(beam(5,:));
end

% Calculate differential with respect to z of normalized 2-d charge profile
q2=q2./sum(q2(:)*bwx*bwz);
q2=-q2;
dqz=gradient(q2,bwz,bwx);
dqz=dqz';

% Dynamically generate 2d mesh for calculation of longitudinal and
% transverse wake potentials broken into a number of coarse grained and
% fine grained regions
[Z,X,zv,xv]=gengrid(R,gamma,beam,gscale,dodiag);

% Calculate scaled longitudinal and transverse wake potentials over
% different region grid locations
for ig=1:length(Z)
  if ~isempty(Z{ig})
    if ig==2
      [Ys{ig},Yx{ig}]=ysfun(zv{ig},xv{ig},R,beta); %#ok<*AGROW>
    else
      [Ys{ig},Yx{ig}]=ysfun(Z{ig},X{ig},R,beta);
    end
    Ys{ig}(isnan(Ys{ig}))=0;
    Yx{ig}(isnan(Yx{ig}))=0;
    Ys{ig}(isinf(Ys{ig}))=0;
    Yx{ig}(isinf(Yx{ig}))=0;
  end
end
% Diagnostic plot of wake calculations in central fine and surrounding
% sparse region
if dodiag
  figure(2)
  subplot(2,2,1);
  scatter3(zv{2},xv{2},Ys{2},'filled'); hold on;
  for ig=3:length(Z)
    if ~isempty(Z{ig})
      scatter3(Z{ig}(:),X{ig}(:),Ys{ig}(:),'filled');
    end
  end
  hold off;
  subplot(2,2,2);
  scatter3(zv{2},xv{2},Yx{2},'filled'); hold on;
  for ig=3:length(Z)
    if ~isempty(Z{ig})
      scatter3(Z{ig}(:),X{ig}(:),Yx{ig}(:),'filled');
    end
  end
  hold off
  subplot(2,2,3);
  mesh(Z{1},X{1},Ys{1});
  subplot(2,2,4);
  mesh(Z{1},X{1},Yx{1});
end
  
% make 3D arrays with z binning encoded for vector processing of
% interpolation routines
for ig=1:length(Z)
  if ig==2
    Ys{ig}=repmat(Ys{ig}(:),1,length(zb));
    Yx{ig}=repmat(Yx{ig}(:),1,length(zb));
  elseif ~isempty(Z{ig})
    Ys{ig}=repmat(Ys{ig}',1,1,length(zb));
    Yx{ig}=repmat(Yx{ig}',1,1,length(zb));
  end
end

% Perform 2D overlap integral of wake potential with normalized 2D current
% histogram. For each observation point in 2D, interpolate 2D current
% histogram at grid locations wake potential calculated at.
dqz1=zeros(length(zb)*3,length(xb)*3);
dqz1(length(zb)+1:2*length(zb),length(xb)+1:2*length(xb))=dqz;
zlen=zb(end)-zb(1); xlen=xb(end)-xb(1);
zvq=[zb-zlen-bwz/2 zb zb+zlen+bwz/2];
xvq=[xb-xlen-bwx/2 xb xb+xlen+bwx/2];
[Zq,Xq]=ndgrid(zvq,xvq);
Ws=zeros(length(xb),length(zb)); Wx=Ws;
for ig=1:length(Z)
  if ig==2
    trarea=repmat(Z{2}(:),1,length(zb));
    Z{2}=zv{2}(:)-reshape(zb,1,length(zb));
    X{2}=repmat(xv{2}(:),1,length(zb));
  elseif ~isempty(Z{ig})
    Z{ig}=Z{ig}'-reshape(zb,1,1,length(zb));
    X{ig}=repmat(X{ig}',1,1,length(zb));
  end
end

% load variables into GPU memory
if usegpu
  ginfo=gpuDevice;
  gpuvar=whos('Zq','Xq','dqz1','trarea','Z','X','Ys','Yx','xv','zv');
  reqmem=sum(arrayfun(@(x) gpuvar(x).bytes,1:length(gpuvar)));
  if reqmem>ginfo.AvailableMemory
    error('Not enough available GPU memory: available = %g (bytes), required = %g\n',gpuinfo.AvailableMemory,reqmem)
  end
  Zq=gpuArray(Zq);
  Xq=gpuArray(Xq);
  dqz1=gpuArray(dqz1);
  trarea=gpuArray(trarea);
  for ig=1:length(Z)
    if ~isempty(Z{ig})
      Z{ig}=gpuArray(Z{ig});
      X{ig}=gpuArray(X{ig});
      Ys{ig}=gpuArray(Ys{ig});
      Yx{ig}=gpuArray(Yx{ig});
      xv{ig}=gpuArray(xv{ig});
      zv{ig}=gpuArray(zv{ig});
    end
  end
end

% Loop over histogam bins and calculate wake potentials
for ix=1:length(xb) % loop over x dimension (z dimension is vectorized)
  for ig=1:length(Z) % loop over integration grids
    if ~isempty(Z{ig})
      dqzI=interpn(Zq,Xq,dqz1,Z{ig},X{ig}-xb(ix)); % interpolate charge distribution at integration grid points
      if ig==2
        Ws(ix,:)=Ws(ix,:)+gather(sum(Ys{ig}.*dqzI.*trarea,1)); % integrate over triangular grid points for fine area
        Wx(ix,:)=Wx(ix,:)+gather(sum(Yx{ig}.*dqzI.*trarea,1));
      else
        Ws(ix,:)=Ws(ix,:)+gather(squeeze(trapz(xv{ig},trapz(zv{ig},Ys{ig}.*dqzI,1),2))'); % quad integration over rectangular grids for other areas
        Wx(ix,:)=Wx(ix,:)+gather(squeeze(trapz(xv{ig},trapz(zv{ig},Yx{ig}.*dqzI,1),2))');
      end
    end
  end
end

% Convert longitudinal and transverse wake potentials into energy loss
% [dE/E] and transverse kicks [rad]
dE=Ws(sub2ind(size(Ws),bcx,bcz)).*cv;
dE(isnan(dE))=0;
dE(isinf(dE))=0;
dXP=Wx(sub2ind(size(Wx),bcx,bcz)).*cv;
dXP(isnan(dXP))=0;
dXP(isinf(dXP))=0;
if theta<0
  dXP=-dXP;
end
if dodiag
  figure(3)
  if length(zb)>1 && length(xb)>1
    [Z1,X1]=meshgrid(zb,xb);
    subplot(2,2,1),surf(Z1,X1,Ws,'LineStyle','none');xlabel('c\Deltat [m]');ylabel('X [m]');zlabel('W_s [m^{-2}]');
    subplot(2,2,3),surf(Z1,X1,Wx,'LineStyle','none');xlabel('c\Deltat [m]');ylabel('X [m]');zlabel('W_x [m^{-2}]');
    subplot(2,2,2)
    ix=ceil(length(xb)/2);
    plot(zb,Ws(ix,:)); xlabel('c\Deltat [m]'); ylabel('W_s [m^{-2}]');
    subplot(2,2,4)
    ix=ceil(length(xb)/2);
    plot(zb,Wx(ix,:)); xlabel('c\Deltat [m]'); ylabel('W_x [m^{-2}]');
  elseif length(zb)==1
    subplot(2,1,1),plot(xb,Ws);xlabel('X [m]');ylabel('W_s [m^{-2}]');
    subplot(2,1,2),plot(xb,Wx);xlabel('X [m]');ylabel('W_x [m^{-2}]');
  else
    subplot(2,1,1),plot(zb,Ws);xlabel('Z [m]');ylabel('W_s [m^{-2}]');
    subplot(2,1,2),plot(zb,Wx);xlabel('Z [m]');ylabel('W_x [m^{-2}]');
  end
end

function [Z,X,zv,xv]=gengrid(R,gamma,beam,gfact,diag)
% Define multiple grids to cover integration area with mesh size adapted to
% specific areas (there is a spiriling region in x-z plane where the wake
% function changes on fine distance scales that needs finer meshing than
% other regions)
warning('off','MATLAB:delaunayTriangulation:DupPtsWarnId');
warning('off','MATLAB:delaunayTriangulation:DupPtsConsUpdatedWarnId');

% Required constants etc
beta=sqrt(1-gamma^-2); % lorentz beta
nsparse=200; % Number of gridding bins for sparse areas (adjusted by user gfact(2) variable)

% First region is a fine mesh around 0
zmin=min(beam(5,:))*2; zmax=max(beam(5,:))*2;
xmin=min(beam(1,:))*2; xmax=max(beam(1,:))*2;
% sx=std(beam(1,:));
zl=5*R;
nz1=ceil(gfact(1)*16);
zv{1}=linspace(-zl/(3*gamma^3),zl/(3*gamma^3),nz1);
if any(zv{1}==0)
  zv{1}=linspace(-zl/(3*gamma^3),zl/(3*gamma^3),nz1+1);
end
xran = (3*sqrt(R)) / gamma^2 ;
if xran>xmax
  xran=abs(xmax-xmin)/10;
end
nx=ceil(24*gfact(1));
xv{1}=linspace(-xran,xran,nx);
if any(xv{1}==0)
  xv{1}=linspace(-xran,xran,nx+1);
end

% Define fine detail area (in the -ve z, +ve x quadrant)
% - Points zp,xm define line centered around area of fine mesh requirement
nz=100;
xm=ones(1,nz*2).*xmax+1;
itry=0; zlim=zmin;
while isempty(find(xm>=xmax,1)) || find(xm>=xmax,1,'last')>nz
  zp=logspace(0,-2,nz*2).*zlim;
  xp=linspace(0,xmax,ceil(0.1*(xmax-xmin)/abs(diff(xv{1}(1:2)))));
  [Z1,X1]=meshgrid(zp,xp);
  Ys=ysfun(Z1,X1,R,beta);
  [~,I]=max(Ys);
  xm=xp(I);
  zlim=zp(find(xm>=xmax,1,'last'));
  if isempty(zlim)
    zlim=zmin;
    break
  end
  itry=itry+1;
  if itry>100
    error('2D CSR gridding error!')
  end
end
zp=logspace(0,-2,nz+1).*zlim;
zp=[zp(1:end-1) linspace(zp(end),min(zv{1}),nz)];
xp=linspace(0,xmax,ceil(0.1*(xmax-xmin)/abs(diff(xv{1}(1:2)))));
[Z1,X1]=meshgrid(zp,xp);
Ys=ysfun(Z1,X1,R,beta);
[~,I]=max(Ys);
xm=xp(I); 

% Diagnostic plot - First-order grid to get fine-mesh areas
if diag
  figure(4)
  subplot(2,1,1),mesh(Z1,X1,Ys);
  subplot(2,1,2),plot(zp,xm)
end

% Define grid regions
% region 2 = area in -ve z and +ve x containing fine mesh marker locations
% Use delaunny triagulation to mesh this area for limiting fine mesh size
% to area around [zp xm] points found above
zv{2}=linspace(min(zp),min(zv{1}),ceil(nsparse*gfact(2)));
nx1=ceil(nsparse*gfact(2));
xv{2}=linspace(max(xv{1}),xmax,ceil(nx1/2)); xv{2}=xv{2}(2:end);
xv{2}=[xv{1}(xv{1}>0) xv{2}];
pz=linspace(zv{2}(1),zv{2}(end),length(zv{2})+1);
P(:,1)=pz;
P(:,2)=interp1(zp,xm,P(:,1));
for ix=1:length(xv{1})
  xnew=interp1(zp,xm+xv{1}(ix),pz,'spline');
  Irej=xnew<min(xv{2}) | xnew>max(xv{2});
  P=[P; [pz(~Irej)' xnew(~Irej)']];
end
[Z2,X2]=ndgrid(zv{2},xv{2});
P=[P; [Z2(:) X2(:)]];
% Constrain edges of triangulation to fit rectangular integration grid area
zplus=find(P(:,1)==max(P(:,1))); zminus=find(P(:,1)==min(P(:,1))); xplus=find(P(:,2)==max(P(:,2))); xminus=find(P(:,2)==min(P(:,2)));
[~,zplusInd]=sort(P(zplus,2)); [~,zminusInd]=sort(P(zminus,2)); [~,xplusInd]=sort(P(xplus,1)); [~,xminusInd]=sort(P(xminus,1));
C=[];
for iz=2:length(zminusInd)
  C(end+1,:)=[zminus(zminusInd(iz-1)) zminus(zminusInd(iz))];
end
for iz=2:length(zplusInd)
  C(end+1,:)=[zplus(zplusInd(iz-1)) zplus(zplusInd(iz))];
end
for ix=2:length(xminusInd)
  C(end+1,:)=[xminus(xminusInd(ix-1)) xminus(xminusInd(ix))];
end
for ix=2:length(xplusInd)
  C(end+1,:)=[xplus(xplusInd(ix-1)) xplus(xplusInd(ix))];
end
DT = delaunayTriangulation(P,C) ;
% Return central coordinates of triangles in zv{2}, xv{2} and area of
% triangles in Z{2}
x1=DT.Points(DT.ConnectivityList(:,1),1);
x2=DT.Points(DT.ConnectivityList(:,2),1);
x3=DT.Points(DT.ConnectivityList(:,3),1);
y1=DT.Points(DT.ConnectivityList(:,1),2);
y2=DT.Points(DT.ConnectivityList(:,2),2);
y3=DT.Points(DT.ConnectivityList(:,3),2);
area = 0.5.*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1));
IC = incenter(DT);
zv{2} = IC(:,1)' ;
xv{2} = IC(:,2)' ;
Z{2} = area;
X{2} = [];
% region 3 = area in -ve z not containing fine mesh marker locations
zv{3}=linspace(min(zp),min(zv{1}),ceil(nsparse*gfact(2)));
xv{3}=linspace(xmin,min(xv{1}),ceil(nx1/2)); xv{3}=xv{3}(1:end-1);
xv{3}=[xv{3} xv{1}(xv{1}<0)];
% region 4 = rest of -ve z space
if min(zv{2})>zmin
  zv{4}=linspace(zmin,min(zv{2}),ceil(nsparse*gfact(2)));
  xv{4}=linspace(xmin,xmax,nx1);
  if any(xv{4}==0)
    xv{4}=linspace(xmin,xmax,nx1+1);
  end
  xv{4}=[xv{1} xv{4}]; xv{4}=unique(sort(xv{4}));
else
  zv{4}=[];
  xv{4}=[];
end
% region 5 = space +ve z of region 1
zv{5}=linspace(max(zv{1}),zmax,ceil(nsparse*gfact(2)));
xv{5}=xv{4};
xv{5}=[xv{1} xv{5}]; xv{5}=unique(sort(xv{5}));
% region 6 = space +ve x from region 1
zv{6}=zv{1};
xv{6}=linspace(max(xv{1}),xmax,ceil(nsparse*gfact(2)));
% region 7 = space -ve x from region 1
zv{7}=zv{1};
xv{7}=linspace(min(xv{1}),xmin,ceil(nsparse*gfact(2)));
% Make 2d mesh grids for rectangular regions
for ig=[1 3 4 5 6 7]
  if ~isempty(zv{ig})
    [Z1,X1]=meshgrid(zv{ig},xv{ig});
    Z{ig}=Z1; X{ig}=X1;
  else
    Z{ig}=[]; X{ig}=[];
  end
end

% - Fuctions below from Y. Cai: Phys. Rev. Accel. Beams 23, 014402 for calculating 2D
% wake potentials (requires incomplete elliptical integral functions
% included in Lucretia/src/utils)
function [psi_s,psi_x]=psifun(xi,chi,R,beta)
qe=1.60217662e-19;
chi=chi+1e-12; % avoid infs
alp=alpfun(xi,chi,beta);
kap=sqrt(chi.^2 + 4.*(1+chi).*sin(alp).^2);
psi_s=qe.*beta^2.* ( cos(2.*alp) - (1./(1+chi)) ) ./ ...
  ( 2.*R.^2.* ( kap - beta.*(1+chi) .* sin(2.*alp) ) );
if nargout==1
  return
end
[F,E]=elliptic123(alp(1:end),-4.*(1+chi(1:end))./chi(1:end).^2);
F=reshape(F,size(alp));
E=reshape(E,size(alp));
psi_x= ((qe^2*beta^2)/(2*R^2)) .* ( ...
  ( ( 1./(abs(chi).*(1+chi)) ) .* ( (2+2.*chi+chi.^2).*F - chi.^2.*E ) ) + ...
  ( kap.^2 - 2.*beta.^2.*(1+chi).^2 + beta^2.*(1+chi).*(2+2.*chi+chi.^2).*cos(2.*alp) ) ./ ...
  ( beta.*(1+chi).*(kap.^2-beta.^2.*(1+chi).^2.*sin(2.*alp).^2) ) - ...
  ( kap.*(1-beta.^2.*(1+chi).*cos(2.*alp)).*sin(2.*alp) ) ./ ...
  ( kap.^2 - beta.^2.*(1+chi).^2.*sin(2.*alp).^2 ) ) ;
psi_phi = ((qe.^2)./(R.^2.*abs(chi))) .* F ;
psi_x = psi_x - psi_phi ;
function alp=alpfun(xi,chi,beta)
% solve alp^4 + v*alp^2 + eta*alp + xsi = 0
v=3.*(1-beta.^2-beta.^2.*chi)./(beta.^2.*(1+chi));
eta=-6.*xi./(beta.^2.*(1+chi));
xsi=3.*(4.*xi.^2-beta.^2.*chi.^2)./(4.*beta.^2.*(1+chi));
ohm= eta.^2./16 - xsi.*v./6 + v.^3./216 + sqrt( (eta.^2./16 - xsi.*v./6 + v.^3./216).^2 - (xsi./3 + v.^2./36).^3 );
m= -v./3 + (xsi./3 + v.^2./36).*ohm.^(-1/3) + ohm.^(1/3) ;
alp=zeros(size(xi),'like',xi);
alp(xi>=0)=0.5.*( sqrt(2.*m(xi>=0)) + sqrt( -2.*(m(xi>=0)+v(xi>=0)) - ...
  2.*eta(xi>=0)./sqrt(2.*m(xi>=0)) ) ) ;
alp(xi<0)=0.5.*( -sqrt(2.*m(xi<0)) + sqrt( -2.*(m(xi<0)+v(xi<0)) + ...
  2.*eta(xi<0)./sqrt(2.*m(xi<0)) ) ) ;
alp=real(alp);
alp(isinf(alp))=0;
function [Ys,Yx]=ysfun(z,x,R,beta)
qe=1.60217662e-19;
xi=z./(2*R);
chi=x./R;
if nargout==2
  [PSIs,PSIx]=psifun(xi,chi,R,beta);
  Ys = real(2.*R.*PSIs./(qe.*beta.^2)) ;
  Yx = real(2.*R.*PSIx./((qe.*beta).^2)) ;
else
  PSIs=psifun(xi,chi,R,beta);
  Ys = real(2.*R.*PSIs./(qe.*beta.^2)) ;
end