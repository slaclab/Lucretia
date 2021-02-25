

% Compute beam sigma matrix, covariance matrix, and chisquare from measured
% beam sizes and their errors at 3 or more WIREs.

%measnum=0;data=emit2d_data(measnum);
measnum=2;data=emit2d_data_2010dec(measnum);

wname=data.wname;
xy=data.xy;
sigRaw=data.sigRaw;
dsigRaw=data.dsigRaw;
etaRaw=data.etaRaw;
detaRaw=data.detaRaw;
wsize=data.wsize;
dp=data.dp;
sigfix=data.sigfix;

% do statistics
[nwire,nmeas]=size(sigRaw);
if (nmeas>1)
  sig=zeros(size(wname));
  dsig=zeros(size(wname));
  for n=1:length(wname)
    id=find(sigRaw(n,:)~=0);
    sig(n)=mean(sigRaw(n,id));
    dsig(n)=std(sigRaw(n,id));
  end
else
  sig=sigRaw;
  dsig=dsigRaw;
end
[nwire,nmeas]=size(etaRaw);
if (nmeas>1)
  eta=zeros(size(wname));
  deta=zeros(size(wname));
  for n=1:length(wname)
    eta(n)=mean(etaRaw(n,:));
    deta(n)=std(etaRaw(n,:));
  end
else
  eta=etaRaw;
  deta=detaRaw;
end

% correct the measured spot sizes for dispersion and finite wire size
if (sigfix)
  sigt=sig;
  sigd=abs(dp*eta);
  sigw=wsize/4;
  sig2=sigt.^2-sigd.^2-sigw.^2;
  if (any(sig2<0))
    disp('  Negative sig2 values after correction')
  end
  sig=sqrt(sig2);
  disp(' ')
  disp('  sigt    sigd    sigw    sig')
  disp('------- ------- ------- -------')
  for n=1:nwire
    fprintf('%7.2f %7.2f %7.2f %7.2f\n',1e6*[sigt(n),sigd(n),sigw(n),sig(n)])
  end
end

% x or y emittance calculation?
ixy=strcmp(xy,'x');
if (ixy)
  roff=0;
  pname='Horizontal';
else
  roff=2;
  pname='Vertical';
end

% get the model
idw=zeros(nwire,1);
for n=1:nwire
  idw(n)=findcells(BEAMLINE,'Name',wname{n});
end
R=zeros(nwire,2);
for n=1:nwire
  [stat,Rab]=RmatAtoB(idw(1),idw(n));
  if (stat{1}~=1),error(stat{2}),end
  R(n,:)=[Rab(1+roff,1+roff),Rab(1+roff,2+roff)];
end
energy=DR_energy(1,813.72); % FL.SimModel.Initial.Momentum;

% load analysis variables
x=sig.^2; % sig^2
sigx=2*sig.*dsig; % statistical error on sig^2

% get design twiss at first wire scanner
if (ixy)
  b0=FL.SimModel.Design.Twiss.betax(idw(1)); % beta
  a0=FL.SimModel.Design.Twiss.alphax(idw(1)); % alpha
else
  b0=FL.SimModel.Design.Twiss.betay(idw(1)); % beta
  a0=FL.SimModel.Design.Twiss.alphay(idw(1)); % alpha
end

% compute least squares solution
M=zeros(nwire,3);
for n=1:nwire
  M(n,1)=R(n,1)^2;
  M(n,2)=2*R(n,1)*R(n,2);
  M(n,3)=R(n,2)^2;
end
z=x./sigx;
B=zeros(nwire,3);
for n=1:nwire
  B(n,:)=M(n,:)/sigx(n);
end
T=inv(B'*B);
u=T*B'*z; %#ok<*MINV>
du=sqrt(diag(T));
chisq=z'*z-z'*B*T*B'*z;

% convert fitted input sigma matrix elements to emittance, BMAG, ...
[p,dp]=emit_params(u(1),u(2),u(3),T,b0,a0);
emit=p(1);demit=dp(1);
bmag=p(2);dbmag=dp(2);
embm=p(3);dembm=dp(3);
beta=p(4);dbeta=dp(4);
alph=p(5);dalph=dp(5);
bcos=p(6);dbcos=dp(6);
bsin=p(7);dbsin=dp(7);
egamma=energy/0.511e-3;
emitn=egamma*emit;demitn=egamma*demit;
embmn=egamma*embm;dembmn=egamma*dembm;

% display results
disp(' ')
fprintf('%s emittance parameters at %s\n',pname,wname{1})
disp('-------------------------------------------------------')
fprintf('energy     = %10.4f              GeV\n',energy)
fprintf('emit       = %10.4f +- %9.4f pm\n',1e12*emit,1e12*demit)
fprintf('emitn      = %10.4f +- %9.4f nm\n',1e9*emitn,1e9*demitn)
fprintf('emitn*bmag = %10.4f +- %9.4f nm\n',1e9*embmn,1e9*dembmn)
fprintf('bmag       = %10.4f +- %9.4f      (%9.4f)\n',bmag,dbmag,1)
fprintf('bmag_cos   = %10.4f +- %9.4f      (%9.4f)\n',bcos,dbcos,0)
fprintf('bmag_sin   = %10.4f +- %9.4f      (%9.4f)\n',bsin,dbsin,0)
fprintf('beta       = %10.4f +- %9.4f m    (%9.4f)\n',beta,dbeta,b0)
fprintf('alpha      = %10.4f +- %9.4f      (%9.4f)\n',alph,dalph,a0)
fprintf('chisq/N    = %10.4f\n',chisq)
disp(' ')

if (any(imag(p(1:3))~=0)||any(p(1:3)<=0))
  error('Error in emittance computation')
end

% propagate measured beam to wire scanners
xf=sqrt(M*u);
fprintf('Propagated %s spot sizes\n',lower(pname))
disp('-----------------------------------')
for n=1:nwire
  fprintf('%4s = %6.1f um (%6.1f +- %6.1f)\n', ...
    wname{n},1e6*[xf(n),sig(n),dsig(n)])
end
disp(' ')

% back propagate fitted sigma matrix to MDISP
sigw=[u(1),u(2);u(2),u(3)];
id1=findcells(BEAMLINE,'Name','MDISP');
id2=findcells(BEAMLINE,'Name','BEGFF');
[stat,Rab]=RmatAtoB(id1,idw(1));
if (stat{1}~=1),error(stat{2}),end
Rab=Rab(1+roff:2+roff,1+roff:2+roff);
sig0=inv(Rab)*sigw*inv(Rab');

% forward propagate through diagnostic section ...
S=FL.SimModel.Design.Twiss.S';
id=(id1:id2)';
idt=find([1;diff(S(id))]~=0); % unique S values
id=id(idt);
sigf=zeros(size(id));
[iss,temp]=GetRmats(id1,id2);
for n=1:length(id)
  [stat,Rab]=RmatAtoB(id1,id(n));
  if (stat{1}~=1),error(stat{2}),end
  Rab=Rab(1+roff:2+roff,1+roff:2+roff);
  sigm=Rab*sig0*Rab';
  sigf(n)=sqrt(sigm(1,1));
end

% ... and plot
figure(1)
plot(S(id),1e6*sigf,'b--')
hold on
plot_barsc(S(idw),1e6*sig,1e6*dsig,'b','o')
hold off
set(gca,'XLim',[S(id1),S(id2)])
title('EXT Diagnostics Section')
ylabel([pname,' Beam Size (um)'])
xlabel('S (m)')
plot_magnets_Lucretia(BEAMLINE(id1:id2),1,1);

% track Twiss parameters
b0=sig0(1,1)/emit;
a0=-sig0(2,1)/emit;
g0=(1+a0^2)/b0;
t0=[b0;a0;g0];
bf=zeros(size(id));bf(1)=b0; % beta
af=zeros(size(id));af(1)=a0; % alpha
pf=zeros(size(id)); % phase
for n=2:length(id)
  [iss,Rab]=RmatAtoB(id1,id(n));
  Rab=Rab(1+roff:2+roff,1+roff:2+roff);
  R11=Rab(1,1);R12=Rab(1,2);R21=Rab(2,1);R22=Rab(2,2);
  M=[  R11^2    -2*R11*R12      R12^2 ; ...
     -R11*R21 R11*R22+R12*R21 -R12*R22; ...
       R21^2    -2*R21*R22      R22^2 ];
  t=M*t0;
  bf(n)=t(1);
  af(n)=t(2);
  pf(n)=pf(n-1)+atan2d(R12,t0(1)*R11-t0(2)*R12);
end

% plot predicted and deduced beta function
figure(2)
plot(S(id),bf,'b--',S(idw),sig.^2/emit,'bo')
set(gca,'XLim',[S(id1),S(id2)])
title('EXT Diagnostics Section')
ylabel([pname,' Beta (m)'])
xlabel('S (m)')
h0=plot_magnets_Lucretia(BEAMLINE(id1:id2),1,1);

% output predicted wire-to-wire phase advances
jdw=zeros(nwire,1);
for n=1:nwire
  jdw(n)=find(id==idw(n));
end
dpf=[0;diff(pf(jdw))];
fprintf('%s wire-to-wire phase advance\n',pname)
disp('-------------------------------------')
for n=1:nwire
  fprintf('%4s = %6.1f deg\n',wname{n},dpf(n))
end
disp(' ')

