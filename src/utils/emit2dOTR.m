function [stat,emitData]=emit2dOTR(otruse,useEllipse,printData)
%
% [stat,emitData]=emit2dOTR(otruse,[useEllipse,printData])
%
% Compute beam sigma matrix, covariance matrix, and chisquare from measured
% beam sizes and their errors at 3 or more OTRs (assumes data already gathered
% by OTR system saved into EPICS PVs)
%
% INPUTs:
%
%   otruse     = [1,4] logical array of which OTRs to use
%   useEllipse = logical (optional)
%                 true = use fitted ellipse data
%                 false = use projected data [default]
%   printData  = logical (optional)
%                 true = display emittance data to screen and show plots
%                 false = no printing to screen or plotting [default]
% OUTPUTs:
%
%   stat     = Lucretia return status
%   emitData = data formatted for sending over FlECS interface:
%     [energy ...
%      emitx demitx emitxn demitxn embmx dembmx ...
%      bmagx dbmagx bcosx dbcosx bsinx dbsinx ...
%      betax dbetax bx0 alphx dalphx ax0 chi2x ...
%      emity demity emityn demityn embmy dembmy ...
%      bmagy dbmagy bcosy dbcosy bsiny dbsiny ...
%      betay dbetay by0 alphy dalphy ay0 chi2y ...
%      ido length(id) id length(S) S sigxf sigyf ...
%      DX dDX DPX dDPX DY dDY DPY dDPY dp ...
%      sigx dsigx sigy dsigy xf yf ...
%      R{1:notr} ...
%      exip0 bxip0 axip0 eyip0 byip0 ayip0 ...
%      sigxip dsigxip sigpxip dsigpxip betaxip dbetaxip alphxip dalphxip ...
%      sigyip dsigyip sigpyip dsigpyip betayip dbetayip alphyip dalphyip ...
%      Lxw dLxw betaxw dbetaxw sigxw dsigxw ...
%      Lyw dLyw betayw dbetayw sigyw dsigyw ...
%     ];
% ------------------------------------------------------------------------------
% 03-Dec-2012, M. Woodley
%    Add to emitData: distance to waists; beta and size at waists; load
%    emitData (row vector) into data (structure) before saving
% 04-Nov-2012, M. Woodley
%    Use energy=FL.SimModel.Initial.Momentum (BH1R fudge updated 24Oct12);
%    add to emitData: 4x4 OTR-to-OTR Rmats, design IP Twiss, propagated IP beam
%    parameters
% ------------------------------------------------------------------------------

global BEAMLINE INSTR FL PS %#ok<NUSED>
stat{1}=1;
emitData=[];

% use fake data if Woodley wills it
% NOTE: mdwGetFlagVal.m lives in ~/home/mdw
if (exist('mdwGetFlagVal.m','file')==2)
  mdwFlag=mdwGetFlagVal('emit2dOTR');
else
  mdwFlag=0;
end

% parse optional parameters
if (~exist('useEllipse','var')||isempty(useEllipse))
  useEllipse=false;
end
if (useEllipse)
  measType='ellipse';
else
  measType='projected';
end
if (~exist('printData','var')||isempty(printData))
  printData=false;
end

% which OTRs to use
if (sum(otruse)<3)
  stat{1}=-1; stat{2}='Must select 3 or 4 OTRs for use';
  return
end
oname={};
for iotr=find(otruse)
  oname{end+1}=sprintf('OTR%dX',iotr-1);
end
notr=length(oname);
z=zeros(size(oname));
sigx=z;dsigx=z;sigy=z;dsigy=z;theta=z;

% get pointers
ido=z;idoi=z;
for n=1:notr
  ido(n)=findcells(BEAMLINE,'Name',oname{n});
  idoi(n)=findcells(INSTR,'Index',ido(n));
end

% get OTR data
if (mdwFlag)
  data=mdwGetEmit(1,otruse);
  sigx=data.sigx;dsigx=data.dsigx; % m
  sigy=data.sigy;dsigy=data.dsigy; % m
  theta=zeros(1,notr);
  DX=data.DX;dDX=data.dDX; % m
  DPX=data.DPX;dDPX=data.dDPX; % m
  DY=data.DY;dDY=data.dDY; % m
  DPY=data.DPY;dDPY=data.dDPY; % m
  dp=data.dp;
  clear data
  stat{1}=1;
else
  for n=1:notr
    otrNum=str2double(oname{n}(4))+1;
    [stat,data]=getOTRsize(otrNum);
    ictdata=data.ict; icterrdata=data.icterr; %#ok<NASGU>
    if (stat{1}~=1)
      stat{1}=-1;
      stat{2}=sprintf('Failed to get data for OTR%dX',n-1);
      return
    end
    rawotrdata{n}=data; %#ok<NASGU>
    if (useEllipse)
      sigx(n)=1e-6*sqrt(data.sig11);dsigx(n)=1e-6*(data.sig11err/2); % m
      sigy(n)=1e-6*sqrt(data.sig33);dsigy(n)=1e-6*(data.sig33err/2); % m
    else
      sigx(n)=1e-6*data.projx;dsigx(n)=1e-6*data.projxerr; % m
      sigy(n)=1e-6*data.projy;dsigy(n)=1e-6*data.projyerr; % m
    end
    theta(n)=data.theta; % deg
  end
  % get dispersion data
  DX=z;DPX=z;DY=z;DPY=z;dDX=z;dDPX=z;dDY=z;dDPY=z;
  for n=1:notr
    D=INSTR{idoi(n)}.dispref;dD=INSTR{idoi(n)}.dispreferr;
    [DX(n),DPX(n),DY(n),DPY(n)]=deal(D(1),D(2),D(3),D(4)); % m,rad,m,rad
    [dDX(n),dDPX(n),dDY(n),dDPY(n)]=deal(dD(1),dD(2),dD(3),dD(4)); % m,rad,m,rad
  end
  if (isfield(FL,'props')&&isfield(FL.props,'dE'))
    dp=FL.props.dE;
  else
    dp=8e-4; % nominal energy spread
  end
end

% correct measured spot sizes for dispersion
sigxt=sigx;sigxd=abs(dp*DX);sigx2=sigxt.^2-sigxd.^2;
sigyt=sigy;sigyd=abs(dp*DY);sigy2=sigyt.^2-sigyd.^2;
if (any(sigx2<0)||any(sigy2<0))
  stat{1}=-1;
  stat{2}='Negative sigx2 or sigy2 values after dispersion correction';
  return
end
sigx=sqrt(sigx2); % dispersion corrected
sigy=sqrt(sigy2); % dispersion corrected

txt{1}=' ';
txt{end+1}=strcat(upper(measType(1)),measType(2:end),':');
txt{end+1}=' ';
txt{end+1}='  sigxt   sigxd    sigx   sigyt   sigyd    sigy';
txt{end+1}='------- ------- ------- ------- ------- -------';
%       nnnn.nn nnnn.nn nnnn.nn nnnn.nn nnnn.nn nnnn.nn
for n=1:notr
  txt{end+1}=sprintf('%7.2f %7.2f %7.2f %7.2f %7.2f %7.2f', ...
    1e6*[sigxt(n),sigxd(n),sigx(n),sigyt(n),sigyd(n),sigy(n)]);
end

% get the model
R=cell(1,notr);
Rx=zeros(notr,2);Ry=zeros(notr,2);
for n=1:notr
  [stat,Rab]=RmatAtoB(ido(1),ido(n));
  if (stat{1}~=1),error(stat{2}),end
  R{n}=Rab(1:4,1:4);
  Rx(n,:)=[Rab(1,1),Rab(1,2)];
  Ry(n,:)=[Rab(3,3),Rab(3,4)];
end

% get design Twiss at first OTR
energy=FL.SimModel.Initial.Momentum;
egamma=energy/0.51099906e-3;
ex0=FL.SimModel.Initial.x.NEmit/egamma;
bx0=FL.SimModel.Design.Twiss.betax(ido(1));
ax0=FL.SimModel.Design.Twiss.alphax(ido(1));
ey0=FL.SimModel.Initial.y.NEmit/egamma;
by0=FL.SimModel.Design.Twiss.betay(ido(1));
ay0=FL.SimModel.Design.Twiss.alphay(ido(1));

% get design Twiss at IP
ip=findcells(BEAMLINE,'Name','IP');
exip0=ex0;
bxip0=FL.SimModel.Design.Twiss.betax(ip);
axip0=FL.SimModel.Design.Twiss.alphax(ip);
eyip0=ey0;
byip0=FL.SimModel.Design.Twiss.betay(ip);
ayip0=FL.SimModel.Design.Twiss.alphay(ip);

% load analysis variables
dsigxd=dDX.*dp;
dsigyd=dDY.*dp;
x=sigx.^2;% m^2
dx=sqrt(((2.*sigxt).^2.*dsigx.^2)+((2.*sigxd).^2.*dsigxd.^2));
y=sigy.^2; % m^2
dy=sqrt(((2.*sigyt).^2.*dsigy.^2)+((2.*sigyd).^2.*dsigyd.^2));
x=x';dx=dx';y=y';dy=dy'; % columns

% compute least squares solution
Mx=zeros(notr,3);My=zeros(notr,3);
for n=1:notr
  Mx(n,1)=Rx(n,1)^2;
  Mx(n,2)=2*Rx(n,1)*Rx(n,2);
  Mx(n,3)=Rx(n,2)^2;
  My(n,1)=Ry(n,1)^2;
  My(n,2)=2*Ry(n,1)*Ry(n,2);
  My(n,3)=Ry(n,2)^2;
end

for itry=1:2
  zx=x./dx;zy=y./dy;
  Bx=zeros(notr,3);By=zeros(notr,3);
  for n=1:notr
    Bx(n,:)=Mx(n,:)/dx(n);
    By(n,:)=My(n,:)/dy(n);
  end
  Tx=inv(Bx'*Bx);Ty=inv(By'*By);
  u=Tx*Bx'*zx;du=sqrt(diag(Tx));  %#ok<MINV>
  v=Ty*By'*zy;dv=sqrt(diag(Ty));  %#ok<MINV>
  if itry==1
    chi2x=zx'*zx-zx'*Bx*Tx*Bx'*zx; %#ok<MINV>
    chi2y=zy'*zy-zy'*By*Ty*By'*zy; %#ok<MINV>
    dx=dx.*sqrt(chi2x);
    dy=dy.*sqrt(chi2y);
  end
end

% convert fitted input sigma matrix elements to emittance, BMAG, ...
[px,dpx]=emit_params(u(1),u(2),u(3),Tx,bx0,ax0);
px(1:3)=abs(px(1:3));
% if (any(imag(px(1:3))~=0)||any(px(1:3)<=0))
%   stat{1}=-1;
%   stat{2}=sprintf('Error in horizontal %s emittance computation',measType);
%   return
% end
emitx=px(1);demitx=dpx(1);
bmagx=px(2);dbmagx=dpx(2);
embmx=px(3);dembmx=dpx(3);
betax=px(4);dbetax=dpx(4);
alphx=px(5);dalphx=dpx(5);
bcosx=px(6);dbcosx=dpx(6);
bsinx=px(7);dbsinx=dpx(7);
emitxn=egamma*emitx;demitxn=egamma*demitx;

[py,dpy]=emit_params(v(1),v(2),v(3),Ty,by0,ay0);
py(1:3)=abs(py(1:3));
% if (any(imag(py(1:3))~=0)||any(py(1:3)<=0))
%   stat{1}=-1;
%   stat{2}=sprintf('Error in vertical %s emittance computation',measType);
%   return
% end
emity=py(1);demity=dpy(1);
bmagy=py(2);dbmagy=dpy(2);
embmy=py(3);dembmy=dpy(3);
betay=py(4);dbetay=dpy(4);
alphy=py(5);dalphy=dpy(5);
bcosy=py(6);dbcosy=dpy(6);
bsiny=py(7);dbsiny=dpy(7);
emityn=egamma*emity;demityn=egamma*demity;

% back propagate fitted sigma matrices to MDISP
sigx1=[u(1),u(2);u(2),u(3)];sigy1=[v(1),v(2);v(2),v(3)];
id1=findcells(BEAMLINE,'Name','MDISP');
id2=findcells(BEAMLINE,'Name','BEGFF');
[stat,Rab]=RmatAtoB(id1,ido(1));
if (stat{1}~=1),error(stat{2}),end
Rx=Rab(1:2,1:2);Ry=Rab(3:4,3:4);
sigx0=inv(Rx)*sigx1*inv(Rx');sigy0=inv(Ry)*sigy1*inv(Ry'); %#ok<MINV>

% forward propagate through diagnostic section ...
S=FL.SimModel.Design.Twiss.S';
id=(id1:id2)';
idt=find([1;diff(S(id))]~=0); % unique S values
id=id(idt); %#ok<FNDSB>
sigxf=zeros(size(id));sigyf=zeros(size(id));
for n=1:length(id)
  [stat,Rab]=RmatAtoB(id1,id(n));
  if (stat{1}~=1),error(stat{2}),end
  Rx=Rab(1:2,1:2);Ry=Rab(3:4,3:4);
  sigxm=Rx*sigx0*Rx';sigym=Ry*sigy0*Ry';
  sigxf(n)=sqrt(sigxm(1,1));sigyf(n)=sqrt(sigym(1,1));
end

% propagate measured beam from first OTR to IP (ignore coupling)
[stat,Rab]=RmatAtoB(ido(1),ip); % first OTR to IP
if (stat{1}~=1),error(stat{2}),end

sig0=[u(1),u(2);u(2),u(3)];
Rx=Rab(1:2,1:2);
sigip=Rx*sig0*Rx';
RT=[   Rx(1,1)^2            2*Rx(1,1)*Rx(1,2)            Rx(1,2)^2   ; ...
    Rx(1,1)*Rx(2,1)  Rx(1,1)*Rx(2,2)+Rx(1,2)*Rx(2,1)  Rx(1,2)*Rx(2,2); ...
       Rx(2,1)^2            2*Rx(2,1)*Rx(2,2)            Rx(2,2)^2   ];
T=RT*Tx*RT'; % propagate covariance matrix to IP
[px,dpx]=emit_params(sigip(1,1),sigip(1,2),sigip(2,2),T,1,0);
sigxip=sqrt(sigip(1,1));dsigxip=sqrt(T(1,1))/(2*sigxip);
sigpxip=sqrt(sigip(2,2));dsigpxip=sqrt(T(3,3))/(2*sigpxip);
betaxip=px(4);dbetaxip=dpx(4);
alphxip=px(5);dalphxip=dpx(5);

% waist shift, beta, beam size, and beam divergence at the waist
try
  [pxw,dpxw]=waist_params(sigip(1,1),sigip(1,2),sigip(2,2),T);
  Lxw=pxw(1);dLxw=dpxw(1);
  betaxw=pxw(2);dbetaxw=dpxw(2);
  sigxw=pxw(3);dsigxw=dpxw(3);
catch
  Lxw=0;dLxw=0;
  betaxw=0;dbetaxw=0;
  sigxw=0;dsigxw=0;
end

sig0=[v(1),v(2);v(2),v(3)];
Ry=Rab(3:4,3:4);
sigip=Ry*sig0*Ry';
RT=[   Ry(1,1)^2            2*Ry(1,1)*Ry(1,2)            Ry(1,2)^2   ; ...
    Ry(1,1)*Ry(2,1)  Ry(1,1)*Ry(2,2)+Ry(1,2)*Ry(2,1)  Ry(1,2)*Ry(2,2); ...
       Ry(2,1)^2            2*Ry(2,1)*Ry(2,2)            Ry(2,2)^2   ];
T=RT*Ty*RT'; % propagate covariance matrix to IP
[py,dpy]=emit_params(sigip(1,1),sigip(1,2),sigip(2,2),T,1,0);
sigyip=sqrt(sigip(1,1));dsigyip=sqrt(T(1,1))/(2*sigyip);
sigpyip=sqrt(sigip(2,2));dsigpyip=sqrt(T(3,3)/(2*sigpyip));
betayip=py(4);dbetayip=dpy(4);
alphyip=py(5);dalphyip=dpy(5);

% waist shift, beta, beam size, and beam divergence at the waist
try
  [pyw,dpyw]=waist_params(sigip(1,1),sigip(1,2),sigip(2,2),T);
  Lyw=pyw(1);dLyw=dpyw(1);
  betayw=pyw(2);dbetayw=dpyw(2);
  sigyw=pyw(3);dsigyw=dpyw(3);
catch
  Lyw=0; dLyw=0;
  betayw=0; dbetayw=0;
  sigyw=0; dsigyw=0;
end

% results txt
txt{end+1}=' ';

txt{end+1}=sprintf('Horizontal %s emittance parameters at %s',measType,oname{1});
txt{end+1}='-----------------------------------------------------';
txt{end+1}=sprintf('energy     = %10.4f              GeV',energy);
txt{end+1}=sprintf('emit       = %10.4f +- %9.4f pm',1e12*emitx,1e12*demitx);
txt{end+1}=sprintf('emitn      = %10.4f +- %9.4f nm',1e9*emitxn,1e9*demitxn);
txt{end+1}=sprintf('emit*bmag  = %10.4f +- %9.4f nm',1e9*embmx,1e9*dembmx);
txt{end+1}=sprintf('bmag       = %10.4f +- %9.4f      (%9.4f)',bmagx,dbmagx,1);
txt{end+1}=sprintf('bmag_cos   = %10.4f +- %9.4f      (%9.4f)',bcosx,dbcosx,0);
txt{end+1}=sprintf('bmag_sin   = %10.4f +- %9.4f      (%9.4f)',bsinx,dbsinx,0);
txt{end+1}=sprintf('beta       = %10.4f +- %9.4f m    (%9.4f)',betax,dbetax,bx0);
txt{end+1}=sprintf('alpha      = %10.4f +- %9.4f      (%9.4f)',alphx,dalphx,ax0);
txt{end+1}=sprintf('chisq/N    = %10.4f',chi2x);
txt{end+1}=' ';

txt{end+1}=sprintf('Horizontal %s emittance parameters at IP',measType);
txt{end+1}='-----------------------------------------------------';
sigx0=sqrt(emitx*bxip0);
sigpx0=sqrt(emitx*(1+axip0^2)/bxip0);
txt{end+1}=sprintf('sig        = %10.4f +- %9.4f um   (%9.4f)',1e6*[sigxip,dsigxip,sigx0]);
txt{end+1}=sprintf('sigp       = %10.4f +- %9.4f ur   (%9.4f)',1e6*[sigpxip,dsigpxip,sigpx0]);
txt{end+1}=sprintf('beta       = %10.4f +- %9.4f mm   (%9.4f)',1e3*[betaxip,dbetaxip,bxip0]);
txt{end+1}=sprintf('alpha      = %10.4f +- %9.4f      (%9.4f)',alphxip,dalphxip,axip0);
txt{end+1}=' ';

txt{end+1}=sprintf('Horizontal %s emittance parameters at waist',measType);
txt{end+1}='-----------------------------------------------------';
txt{end+1}=sprintf('L          = %10.4f +- %9.4f m',Lxw,dLxw);
txt{end+1}=sprintf('beta       = %10.4f +- %9.4f mm',1e3*[betaxw,dbetaxw]);
txt{end+1}=sprintf('sig        = %10.4f +- %9.4f um',1e6*[sigxw,dsigxw]);
txt{end+1}=' ';

txt{end+1}=sprintf('Vertical %s emittance parameters at %s',measType,oname{1});
txt{end+1}='-----------------------------------------------------';
txt{end+1}=sprintf('energy     = %10.4f              GeV',energy);
txt{end+1}=sprintf('emit       = %10.4f +- %9.4f pm',1e12*emity,1e12*demity);
txt{end+1}=sprintf('emitn      = %10.4f +- %9.4f nm',1e9*emityn,1e9*demityn);
txt{end+1}=sprintf('emit*bmag  = %10.4f +- %9.4f pm',1e12*embmy,1e12*dembmy);
txt{end+1}=sprintf('bmag       = %10.4f +- %9.4f      (%9.4f)',bmagy,dbmagy,1);
txt{end+1}=sprintf('bmag_cos   = %10.4f +- %9.4f      (%9.4f)',bcosy,dbcosy,0);
txt{end+1}=sprintf('bmag_sin   = %10.4f +- %9.4f      (%9.4f)',bsiny,dbsiny,0);
txt{end+1}=sprintf('beta       = %10.4f +- %9.4f m    (%9.4f)',betay,dbetay,by0);
txt{end+1}=sprintf('alpha      = %10.4f +- %9.4f      (%9.4f)',alphy,dalphy,ay0);
txt{end+1}=sprintf('chisq/N    = %10.4f',chi2y);
txt{end+1}=' ';

txt{end+1}=sprintf('Vertical %s emittance parameters at IP',measType);
txt{end+1}='-----------------------------------------------------';
sigy0=sqrt(emity*byip0);
sigpy0=sqrt(emity*(1+ayip0^2)/byip0);
txt{end+1}=sprintf('sig        = %10.4f +- %9.4f um   (%9.4f)',1e6*[sigyip,dsigyip,sigy0]);
txt{end+1}=sprintf('sigp       = %10.4f +- %9.4f ur   (%9.4f)',1e6*[sigpyip,dsigpyip,sigpy0]);
txt{end+1}=sprintf('beta       = %10.4f +- %9.4f mm   (%9.4f)',1e3*[betayip,dbetayip,byip0]);
txt{end+1}=sprintf('alpha      = %10.4f +- %9.4f      (%9.4f)',alphyip,dalphyip,ayip0);
txt{end+1}=' ';

txt{end+1}=sprintf('Vertical %s emittance parameters at waist',measType);
txt{end+1}='-----------------------------------------------------';
txt{end+1}=sprintf('L          = %10.4f +- %9.4f m',Lyw,dLyw);
txt{end+1}=sprintf('beta       = %10.4f +- %9.4f mm',1e3*[betayw,dbetayw]);
txt{end+1}=sprintf('sig        = %10.4f +- %9.4f um',1e6*[sigyw,dsigyw]);
txt{end+1}=' ';

% propagate measured beam to OTRs
xf=sqrt(Mx*u);yf=sqrt(My*v);
txt{end+1}=sprintf('Propagated spot sizes');
txt{end+1}='---------------------------------------------------------------------------';
for n=1:notr
  txt{end+1}=sprintf('%5s(x) = %6.1f um (%6.1f +- %6.1f), (y) = %6.1f um (%6.1f +- %6.1f)', ...
    oname{n},1e6*[xf(n),sigx(n),dsigx(n),yf(n),sigy(n),dsigy(n)]);
end
txt{end+1}=' ';

% Print data to screen and plot if requested
if (printData)
  for n=1:length(txt),disp(txt{n}),end
  figure(1)
  plot(S(id),1e6*sigxf,'b--')
  hold on
  plot_barsc(S(ido)',1e6*sigx',1e6*dsigx','b','o')
  hold off
  set(gca,'XLim',[S(id1),S(id2)])
  title('EXT Diagnostics Section')
  ylabel('Horizontal Beam Size (um)')
  xlabel('S (m)')
  plot_magnets_Lucretia(BEAMLINE(id),1,1);
  figure(2)
  plot(S(id),1e6*sigyf,'b--')
  hold on
  plot_barsc(S(ido)',1e6*sigy',1e6*dsigy','b','o')
  hold off
  set(gca,'XLim',[S(id1),S(id2)])
  title('EXT Diagnostics Section')
  ylabel('Vertical Beam Size (um)')
  xlabel('S (m)')
  plot_magnets_Lucretia(BEAMLINE(id),1,1);
end

% load return data array
emitData=[energy ...
  emitx demitx emitxn demitxn embmx dembmx ...
  bmagx dbmagx bcosx dbcosx bsinx dbsinx ...
  betax dbetax bx0 alphx dalphx ax0 chi2x ...
  emity demity emityn demityn embmy dembmy ...
  bmagy dbmagy bcosy dbcosy bsiny dbsiny ...
  betay dbetay by0 alphy dalphy ay0 chi2y ...
  ido length(id) id' length(S) S' sigxf' sigyf' ...
  DX dDX DPX dDPX DY dDY DPY dDPY dp ...
  sigx dsigx sigy dsigy xf' yf'];
for n=1:notr
  emitData=[emitData reshape(R{n},1,[])]; % R{n}=reshape(...,4,[])
end
emitData=[emitData ...
  exip0 bxip0 axip0 eyip0 byip0 ayip0 ...
  sigxip dsigxip sigpxip dsigpxip betaxip dbetaxip alphxip dalphxip ...
  sigyip dsigyip sigpyip dsigpyip betayip dbetayip alphyip dalphyip ...
  Lxw dLxw betaxw dbetaxw sigxw dsigxw ...
  Lyw dLyw betayw dbetayw sigyw dsigyw];

% copy results to a structure for saving
data=emitDataUnpack(otruse,emitData); %#ok<NASGU>

% save everything
if (useEllipse)
  % ellipse
  fname=sprintf('userData/emit2dOTRe_%s',datestr(now,30));
else
  % projected
  fname=sprintf('userData/emit2dOTRp_%s',datestr(now,30));
end
save(fname,'rawotrdata','otruse','ictdata','icterrdata','data','BEAMLINE','PS')

end