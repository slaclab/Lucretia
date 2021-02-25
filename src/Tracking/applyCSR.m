function [beam, W, dE, zOut]=applyCSR(beam,beamQ,stop,nbin,smoothVal,itrack,driftL,driftDL,xmean0)
% [beam W dE zOut]=applyCSR(beam,nbin,smoothVal,itrack,driftL,driftDL)
%  Calculate CSR wake and change provided momentum profile
% - Incorporates 2D CSR wake function (in x, z planes)
%   Implemented using calculations by Y. Cai: Phys. Rev. Accel. Beams 23, 014402
%                                             https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.20.064402
%   (switched on by setting CSR_2D TrackFlag to 1- see documentation)
% - This function is designed to be called from the mex tracking
%   environment and should not be altered
%
% beam: Lucretia macro particle beam definition (Actually pass Beam.Bunch.x)
% beamQ: charge of macro particles
% stop: vector of stop indices (from Lucretia Bunch)
% nbin: number of bins to use for CSR calculation
% smoothVal: level of smoothing to use (int >= 1)
% itrack: BEAMLINE element number
% driftL: distance from d/s edge of previous bend
% driftDL: segment length of this downstream (from bend) element
% xmean0: mean x position of bunch at bend entrance
% (last 2 arguments only for application of CSR in downstream areas from
% bend)
global BEAMLINE
persistent bininds z Z ZSP lastInd iter gev2tm beamR diagData UseDiagInd

W=[]; dE=[]; zOut=[];

% Return diagnostic data on demand
if isequal(beam,'GetDiagData')
  beam=diagData;
  return
elseif isequal(beam,'ClearDiagData')
  diagData=[]; UseDiagInd=[];
  return
end

% Interpolate from previously calculated z vs CSR energy loss curves for this slice
if isfield(BEAMLINE{itrack}.TrackFlag,'UseDiagData') && BEAMLINE{itrack}.TrackFlag.UseDiagData && ~isempty(diagData)
  if isempty(UseDiagInd) || UseDiagInd>=length(diagData.dE)
    UseDiagInd=1;
  else
    UseDiagInd=UseDiagInd+1;
  end
  de = interp1(diagData.z{UseDiagInd},diagData.dE{UseDiagInd},beam(5,~stop)) ;
  de(isnan(de)) = 0 ; % ignore particles outside of interpolated region
  beam(6,~stop)=beam(6,:)+de;
  return
end

% Check there are some non-stopped particles
if all(stop>0)
  error('All particles stopped, element %d',itrack)
end

% Initialize diagnostic data structure
if isempty(diagData)
  diagData.S=[];
  diagData.dE={};
  diagData.z={};
  diagData.W={};
  diagData.Ws={};
  diagData.Wx={};
end

% constants
if isempty(gev2tm)
  gev2tm=3.335640952;
end

%- If zero length element just return
if ~isfield(BEAMLINE{itrack},'L') || BEAMLINE{itrack}.L==0
  return
end

% Cut beam tails if requested
Qcut=0;
if isfield(BEAMLINE{itrack},'TrackFlag') && isfield(BEAMLINE{itrack}.TrackFlag,'CSR') && abs(BEAMLINE{itrack}.TrackFlag.CSR)>0
  if isfield(BEAMLINE{itrack},'TrackFlag') && isfield(BEAMLINE{itrack}.TrackFlag,'Qcut')
    Qcut=BEAMLINE{itrack}.TrackFlag.Qcut;
  end
else
  csrele=[]; % -- find CSR elements upstream
  for iele=itrack-1:-1:1
    if isfield(BEAMLINE{iele},'TrackFlag') && isfield(BEAMLINE{iele}.TrackFlag,'CSR') && abs(BEAMLINE{iele}.TrackFlag.CSR)>0
      csrele(end+1)=iele;
    elseif ~isempty(csrele) && isfield(BEAMLINE{iele},'L') && BEAMLINE{iele}.L>0
      break
    end
  end
  if isempty(csrele)
    error('No BEND found upstream for CSR calculation')
  end
  if isfield(BEAMLINE{csrele(1)},'TrackFlag') && isfield(BEAMLINE{csrele(1)}.TrackFlag,'Qcut')
    Qcut=BEAMLINE{csrele(1)}.TrackFlag.Qcut;
  else
    Qcut=0;
  end
end
if Qcut>0
  [~,iz]=sort(beam(5,:));
  Qtmp = beamQ ; Qtmp(stop>0)=0; Qtmp=Qtmp(iz);
  qsum=cumsum(Qtmp); qsum=qsum./qsum(end);
  i1=find(qsum>Qcut/2,1)-1;i2=find(qsum>(1-Qcut/2),1);
  stop(iz([1:i1 i2:end]))=itrack;
end

% Find out what fraction of the way through this element we are
if isfield(BEAMLINE{itrack},'TrackFlag') && isfield(BEAMLINE{itrack}.TrackFlag,'Split') && BEAMLINE{itrack}.TrackFlag.Split>0
  if isempty(lastInd) || itrack~=lastInd
    iter=1;
    if isfield(BEAMLINE{itrack},'TrackFlag') && isfield(BEAMLINE{itrack}.TrackFlag,'CSR') && abs(BEAMLINE{itrack}.TrackFlag.CSR)>0
      beamR=sqrt(std(beam(1,~stop))^2 + std(beam(3,~stop))^2);
    end
  else
    iter=iter+1;
  end
  splitfrac=iter/BEAMLINE{itrack}.TrackFlag.Split;
  nsplit=BEAMLINE{itrack}.TrackFlag.Split;
else
  splitfrac=1;
  nsplit=1;
end
if splitfrac>1; error('Iterating element %d too many times - Try to fix by issuing command: ''clear applyCSR''',itrack); end
lastInd=itrack;

% Mean particle momentum
beamP=mean(beam(6,~stop));

% Find distance from start of CSR element
if isfield(BEAMLINE{itrack},'TrackFlag') && isfield(BEAMLINE{itrack}.TrackFlag,'CSR') && abs(BEAMLINE{itrack}.TrackFlag.CSR)>0
  if itrack==1
    ind1=itrack;
  else
    for iele=itrack-1:-1:0
      % is this still a CSR element?
      if iele==0 || ~(isfield(BEAMLINE{iele},'TrackFlag') && isfield(BEAMLINE{iele}.TrackFlag,'CSR') && abs(BEAMLINE{iele}.TrackFlag.CSR)>0)
        % is it a real element that isn't a CSR element?
        % - then the start of CSR element is the element after this
        if iele==0 || (isfield(BEAMLINE{iele},'L') && BEAMLINE{iele}.L>0)
          ind1=iele+1;
          break
        end
      end
    end
  end
  % Get parameters for CSR wake calculation
  if strcmp(BEAMLINE{itrack}.Class,'SBEN')
    R = BEAMLINE{itrack}.L / abs(BEAMLINE{itrack}.Angle) ;
  elseif strcmp(BEAMLINE{itrack}.Class,'PLENS')
    quadang = (beamR*abs(BEAMLINE{itrack}.B))/BEAMLINE{itrack}.L/beamP/gev2tm ;
    R = BEAMLINE{itrack}.L / abs(quadang) ;
  else
    R=0;
  end
  PHI=0;
  if ind1~=itrack
    for iele=ind1:itrack-1
      if isfield(BEAMLINE{iele},'Angle')
        PHI = PHI + abs(BEAMLINE{iele}.Angle(1));
      elseif strcmp(BEAMLINE{iele}.Class,'PLENS')
        PHI = PHI + (beamR*abs(BEAMLINE{iele}.B))/BEAMLINE{iele}.L/beamP/gev2tm ;
      end
    end
  end
  if isfield(BEAMLINE{itrack},'Angle')
    PHI=PHI+abs(BEAMLINE{itrack}.Angle(1)*splitfrac);
  elseif strcmp(BEAMLINE{itrack}.Class,'PLENS')
    PHI = PHI + quadang*splitfrac ;
  end
  X=0;
else
  % --- bend angle and radius
  PHI = 0;
  for iele = csrele
    if isfield(BEAMLINE{iele},'Angle')
      PHI = PHI + abs(BEAMLINE{iele}.Angle);
    elseif strcmp(BEAMLINE{iele}.Class,'PLENS')
      PHI = PHI + (beamR*abs(BEAMLINE{iele}.B))/BEAMLINE{iele}.L/beamP/gev2tm ;
    end
  end
  L=sum(arrayfun(@(x) BEAMLINE{x}.L,csrele));
  R=L/PHI;
  % --- distance to CSR element from end of section being considered
  if exist('driftL','var')
    X=driftL/R ;
  else
    X=((BEAMLINE{itrack}.S+BEAMLINE{itrack}.L/2)-(BEAMLINE{csrele(1)}.S+BEAMLINE{csrele(1)}.L))/R;
  end
  lDecay=3*(24*std(beam(5,~stop))*R^2)^(1/3);
  %   fprintf('iele: %d driftL: %g driftDL: %g\n',itrack,driftL-driftDL/2,driftDL)
  if X*R > lDecay; return; end
  % get bins and smoothing parameter from upstream bend element
  if isfield(BEAMLINE{csrele(1)},'TrackFlag') && isfield(BEAMLINE{csrele(1)}.TrackFlag,'CSR_SmoothFactor')
    smoothVal=BEAMLINE{csrele(1)}.TrackFlag.CSR_SmoothFactor;
  end
  if isfield(BEAMLINE{csrele(1)},'TrackFlag') && isfield(BEAMLINE{csrele(1)}.TrackFlag,'CSR')
    nbin=BEAMLINE{csrele(1)}.TrackFlag.CSR;
  end
end

% Generate longitudinal grid
zvals=-beam(5,~stop);
zvals=zvals-mean(zvals); % locally centered z distribution
zmin=min(zvals);
zmax=max(zvals);
if zmin==zmax
  error('Need some spread in z-distribution of bunch to compute CSR!')
end
if nbin<0 % Use automatic binning
  [~,z,bininds]=histcounts(zvals);
  nbin=length(z)-1;
  z=z(1:end-1)+diff(z(1:2))/2;
elseif nbin==1
  error('Must select >1 bin (using CSR field in element TrackFlag');
else % user-defined binning
  [~,z,bininds]=histcounts(zvals,nbin);
  z=z(1:end-1)+diff(z(1:2))/2;
end
[Z, ZSP]=meshgrid(z,z);

% Bin beam particle longitudinal direction
% - zero out charge for stopped particles
beamQ(stop>0)=[];
q = accumarray(bininds',beamQ')';
bw=abs(z(2)-z(1));
Q=sum(q);
q=q./bw;
if smoothVal==0
  q=smoothn(q,'robust','MaxIter',10000);
else
  q=smoothn(q,smoothVal,'MaxIter',10000);
end
dq=gradient(q,bw);
if smoothVal==0
  dq=smoothn(dq,'robust','MaxIter',10000);
else
  dq=smoothn(dq,smoothVal,'MaxIter',10000);
end

%Wake conversion constants
gamma=beamP/0.511e-3; % lorentz gamma
re=2.8179403227e-15; % classical electron radius
qe=1.60217662e-19; % electron charge
cv=((Q/qe).*re)./gamma; % conversion factor for wakes-> relative energy loss

% Diagnostics request flag
if isfield(BEAMLINE{itrack}.TrackFlag,'Diagnostics')
  dodiag=BEAMLINE{itrack}.TrackFlag.Diagnostics;
else
  dodiag=false;
end

% Is this the CSR element itself or downstream?
if isfield(BEAMLINE{itrack},'TrackFlag') && isfield(BEAMLINE{itrack}.TrackFlag,'CSR') && abs(BEAMLINE{itrack}.TrackFlag.CSR)>0
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
  % If request 2D CSR calculation, then just compute transient 1D wake not
  % included in 2D CSR steady-state solution
  if isfield(BEAMLINE{itrack}.TrackFlag,'CSR_2D') && BEAMLINE{itrack}.TrackFlag.CSR_2D>0 && strcmp(BEAMLINE{itrack}.Class,'SBEN')
    W=-(4/(R*PHI)).*q(I1)' ; % Transient 1-d wake potential
    if abs(dodiag)>1 % also store 1d wake calculation if diagnostics mode enabled
      W_1d=-(4/(R*PHI)).*q(I1)' + (4/(R*PHI)).*q(I2)' + (2/((3*R^2)^(1/3))).*ZINT;
    end
    % Calculate 2D CSR wake energy loss per particle (internally binned in 2D)
    if isfield(BEAMLINE{itrack}.TrackFlag,'CSR_2D_GSCALE')
      gscale=BEAMLINE{itrack}.TrackFlag.CSR_2D_GSCALE;
      if length(gscale)~=2 || any(gscale<=0)
        error('CSR flag error with element %d (incorrect CSR_2D_RESSCALE flag format), see documentaion',itrack);
      end
    else
      gscale=[1 1];
    end
    % Specify Z and X bins separately?
    if isfield(BEAMLINE{itrack}.TrackFlag,'CSR_2D_NX') && BEAMLINE{itrack}.TrackFlag.CSR_2D_NX>0
      nbin=[nbin BEAMLINE{itrack}.TrackFlag.CSR_2D_NX];
    end
    % Use GPU for calculation?
    if isfield(BEAMLINE{itrack}.TrackFlag,'CSR_USEGPU') && BEAMLINE{itrack}.TrackFlag.CSR_USEGPU>0
      dogpu=true;
    else
      dogpu=false;
    end
    if abs(dodiag)
      [dE2d,dXP]=csr2dfun(gscale,R,BEAMLINE{itrack}.Angle,gamma,beam(:,~stop),beamQ,nbin,smoothVal,dodiag,xmean0,dogpu);
    else
      [dE2d,dXP]=csr2dfun(gscale,R,BEAMLINE{itrack}.Angle,gamma,beam(:,~stop),beamQ,nbin,smoothVal,dodiag,xmean0,dogpu);
    end
    dE2d = dE2d .* BEAMLINE{itrack}.L./nsplit ; % dE/E
    dXP = dXP .* BEAMLINE{itrack}.L./nsplit ;
  else
    W=-(4/(R*PHI)).*q(I1)' + (4/(R*PHI)).*q(I2)' + (2/((3*R^2)^(1/3))).*ZINT;
  end
  dE=-cv.*(W'.*(BEAMLINE{itrack}.L./nsplit))./Q; % dE/E
else % DRIFT or other element following bend
  % Get parameters for wake calculation
  dsmax=((R*PHI^3)/24)*((PHI+4*X)/(PHI+X));
  
  % Calculate CSR wake and energy loss per bin
  ZINT=zeros(nbin,1,'like',z);
  psi=nan(1,nbin);
  % Ignore case D if dsmax less than a bin width
  if dsmax>diff(z(1:2))
    for is=1:nbin
      isp=z>=(z(is)-dsmax) & (1:length(z))<is ;
      if any(isp)
        ds=z(is)-z(isp);
        a=24*ds(1)/R; b=4*X; C=[-1 -b 0 a a*X];
        % Polynomial roots via a companion matrix
        a = diag(ones(1,3,'like',X),-1);
        d = C(2:end)./C(1);
        a(1,:) = -d;
        rpsi = eig(a);
        psi(1)=max(real(rpsi(imag(rpsi)==0))); % take real root > 0
        psi_eval = psi(~isnan(psi) & isp);
        ZINT(is)=sum((1./(psi_eval+2.*X)).*dq(isp).*bw);
        psi=circshift(psi,1);
      end
    end
  end
  IND1=abs(Z-(ZSP-(R/6)*PHI^2*(PHI+3*X)));
  [~, I]=min(IND1,[],2);
  IND2=abs(Z-(ZSP-dsmax));
  [~, I1]=min(IND2,[],2);
  W = (4/R)*( (q(I1)./(PHI+2*X)) + ZINT' ) - (4/R)*(1/(PHI+2*X)).*q(I) ;
  if exist('driftDL','var')
    dE=-cv.*(W'.*driftDL)./Q; % dE/E
  else
    dE=-cv.*(W'.*BEAMLINE{itrack}.L)./Q; % dE/E
  end
  dE=dE';
end
if abs(dodiag) % store diagnostics data
  if ~strcmp(BEAMLINE{itrack}.Class,'SBEN') && exist('driftDL','var')
    diagData.S(end+1)=BEAMLINE{itrack}.S+driftDL;
  else
    diagData.S(end+1)=BEAMLINE{itrack}.S+BEAMLINE{itrack}.L*splitfrac;
  end
  diagData.dE{end+1}=dE(bininds).*beamP;
  diagData.W{end+1}=(-cv.*W(bininds)'./Q).*beamP.*1e9;
  diagData.z{end+1}=beam(5,~stop);
  % Order by z and decimate
  [~,zord]=sort(diagData.z{end});
  diagData.z{end}=diagData.z{end}(zord); diagData.dE{end}=diagData.dE{end}(zord); diagData.W{end}=diagData.W{end}(zord);
  if length(diagData.z{end})>10000
    inds=1:ceil(length(diagData.z)/10000):length(diagData.z{end});
    if inds(end)~=length(diagData.z{end})
      inds=[inds length(diagData.z{end})];
    end
    diagData.z{end}=diagData.z{end}(inds);
    diagData.dE{end}=diagData.dE{end}(inds);
    diagData.W{end}=diagData.W{end}(inds);
  end
  if isfield(BEAMLINE{itrack}.TrackFlag,'CSR_2D') && BEAMLINE{itrack}.TrackFlag.CSR_2D && strcmp(BEAMLINE{itrack}.Class,'SBEN')
    diagData.Ws{end+1} = (dE2d ./ (BEAMLINE{itrack}.L./nsplit)).*beamP .* 1e9 ;
    diagData.Wx{end+1} = dXP ./ (BEAMLINE{itrack}.L./nsplit) ;
    diagData.dE{end}=diagData.dE{end}+dE2d.*beamP;
  end
end
% Apply energy loss for all particles in each bin (in 2d case this is the
% transient only contributions to energy loss)
if ~dodiag || dodiag>0 % Don't apply wake kicks if Diagnostics flag set to <0
  beam(6,~stop)=beam(6,~stop)+dE(bininds).*beamP;
end
% If calculated 2D CSR wakes, then apply 2D steady-state eloss and transverse kicks to beam
if isfield(BEAMLINE{itrack}.TrackFlag,'CSR_2D') && BEAMLINE{itrack}.TrackFlag.CSR_2D>0
  if ~dodiag || dodiag>0 % Don't apply wake kicks if Diagnostics flag set to <0
    beam(6,~stop)=beam(6,~stop)+dE2d.*beamP;
    beam(2,~stop)=beam(2,~stop)+dXP;
  end
  if abs(dodiag)>1
    figure(1)
    dE1=-cv.*(W_1d'.*(BEAMLINE{itrack}.L./nsplit))./Q;
    subplot(2,1,1), plot(beam(5,~stop),(dE2d).*beamP,'b.'),xlabel('Z [m]');ylabel('dE [GeV]');
    hold on
    plot(beam(5,~stop),dE1(bininds).*beamP,'r.');
    yyaxis right, histogram(beam(5,~stop),'DisplayStyle','stairs','Normalization','probability'); ylabel('1/N dn/dz [m^[-1]');
    subplot(2,1,2), plot(beam(5,~stop),dXP,'.'),xlabel('Z [m]'); ylabel('X'' [rad]');
  end
elseif abs(dodiag)>1
  figure(1)
  plot(beam(5,~stop),dE(bininds).*beamP,'.'),xlabel('Z [m]');ylabel('qW(s) [V/m]');
  hold on
end
zOut=z;

function [dE,dXP,Ws,Wx,zb,xb]=csr2dfun(gscale,R,theta,gamma,beam,beamQ,nbin,smoothVal,dodiag,xmean0,usegpu)
persistent gridData
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
if nbin(1)<0 % use automatic binning
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
% and apply smoothing again
q2=q2./sum(q2(:)*bwx*bwz);
q2=-q2;
dqz=gradient(q2,bwz,bwx);
if smoothVal>0 % user-requested smoothing parameter
  dqz=smoothn(dqz,smoothVal,'MaxIter',10000);
else % auto smoothing
  dqz=smoothn(dqz,'robust','MaxIter',10000);
end
dqz=dqz';

% Dynamically generate 2d mesh for calculation of longitudinal and
% transverse wake potentials broken into a number of coarse grained and
% fine grained regions
% - keep last generated grid values if bunch length and horizontal size
% haven't changed by more than 10% since last calculation
sx=std(beam(1,:)); sz=std(beam(5,:)); e=mean(beam(6,:));
if isempty(gridData) || sx<gridData.sx*0.9 || sx>gridData.sx*1.1 || sz<gridData.sz*0.9 || sz>gridData.sz*1.1 || e<gridData.e*0.9 || e>gridData.e*1.1 || R~=gridData.R || ~isequal(gscale,gridData.gscale)
  [Z,X,zv,xv]=gengrid(R,gamma,beam,gscale,dodiag);
  gridData.sz=sz; gridData.sx=sx; gridData.Z=Z; gridData.X=X; gridData.zv=zv; gridData.xv=xv;
  gridData.e=e; gridData.R=R; gridData.gscale=gscale;
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
  if abs(dodiag)>2
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
  gridData.Ys=Ys; gridData.Yx=Yx;
else
  Z=gridData.Z; X=gridData.X; zv=gridData.zv; xv=gridData.xv;
  Ys=gridData.Ys; Yx=gridData.Yx;
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
if abs(dodiag)>2
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
xl=3*sqrt(R);
nx=ceil(24*gfact(1));
xv{1}=linspace(-xl/gamma^2,xl/gamma^2,nx);
if any(xv{1}==0)
  xv{1}=linspace(-xl/gamma^2,xl/gamma^2,nx+1);
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
if abs(diag)>2
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

% - Fuctions below from Y. Cai: PPhys. Rev. Accel. Beams 23, 014402 for calculating 2D
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