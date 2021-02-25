function IPdat = getIPTwiss(wip,sdata,dsdata,etadata,detadata,psqd0,dim,dE)
% GETIPTWISS
% IPdat = getIPTwiss(wip,sdata,dsdata,etadata,detadata,psqd0,dim,dE)
%   Uses 2 methods to get IP twiss and dispersion given QD0FF scan data and
%   corresponding dispersion measurements
%     IPdat.mfit data uses linear fit based on Model
%     IPdat.pfit uses parabolic fit to data based on thin-lens analytical
%     model
%
% Input:
%    wip: which IP location (1= nominal IP 2= Post-IP)
%    sdata: measured beam size SQUARED at each QD0FF scan point
%    dsdata: error on above for each QD0FF scan point
%    etadata: measured dispersion at scan location for each QD0FF scan
%             point
%    detadata: error on above for each QD0FF scan point
%    psqd0: Lucretia PS setting for each QD0FF scan point
%    dim: 1= horizontal data supplied; 2=vertical data supplied
%    dE: Energy spread of beam (dE/E)
%
% Output:
%    IPdat.mfit
%
%              At IP:
%              ------
%              .emit : emittance (m.rad)
%              .emit_err : error on emittance
%              .beta : beta (m)
%              .beta_err : error on beta
%              .alpha : alpha
%              .alpha_err : error on alpha
%
%              At Waist:
%              ---------
%              .wemit
%              .wemit_err
%              .wbeta
%              .wbeta_err
%              .walpha
%              .walpha_err
%              .wpos : longitudinal position offset of waist from IP (m)
%              .wpos_err : error on above
%    IPdat.pfit
%              .emit
%              .emit_err
%              .beta
%              .beta_err
%
% =====================================
% G. White May 10 2010
% =====================================
global BEAMLINE

% Warnings to ignore for this subroutine
warnstat(1)=warning('query','MATLAB:illConditionedMatrix');
warnstat(2)=warning('query','MATLAB:lscov:RankDefDesignMat');
for istat=1:length(warnstat)
  warning('off',warnstat(istat).identifier);
end

% Get QD0 and IP indicies
if wip==1
  ip=findcells(BEAMLINE,'Name','IP');
else
  ip=findcells(BEAMLINE,'Name','MW1IP');
end
indqd0=findcells(BEAMLINE,'Name','QD0FF');

% Get response matricies QD0-IP for nominal and provided quad settings
Rd = getQRmat(indqd0,ip,psqd0);

% check arguments
ss=size(sdata);
if ss(2)>ss(1); sdata=sdata'; end;
ss=size(dsdata);
if ss(2)>ss(1); dsdata=dsdata'; end;
ss=size(etadata);
if ss(2)>ss(1); etadata=etadata'; end;
ss=size(detadata);
if ss(2)>ss(1); detadata=detadata'; end;
if length(dsdata)==1
  dsdata=(1:length(sdata))'.*dsdata;
end
if length(detadata)==1
  detadata=(1:length(etadata))'.*detadata;
end

% Fit sigmas and etas at QD0 entrance
for imc=1:100
  A=[Rd.A(:,dim) Rd.B(:,dim) Rd.C(:,dim)];
  B=[Rd.D(:,dim) Rd.E(:,dim)];
  w=1./dsdata.^2;
  qsig=lscov(A,sdata+randn(size(dsdata)).*dsdata,w);
  w=1./detadata.^2;
  qeta=lscov(B,etadata+randn(size(detadata)).*detadata,w);
  etaq=qeta(1);
  etapq=qeta(2);
  % Propogate sigmas and etas to IP
  sig1=[qsig(1) qsig(2) etaq*dE^2;
        qsig(2) qsig(3) etapq*dE^2;
          0       0          dE^2];
  ipsig=Rd.Rn{dim}*sig1*Rd.Rn{dim}';

  % Pull off eta and eta' data at IP
  IPdat(imc).eta=ipsig(1,3)/ipsig(3,3);
  IPdat(imc).etap=ipsig(2,3)/ipsig(3,3);

  % Subtract dispersive contribution to sigma matrix
  ipsig(1,1)=ipsig(1,1)-IPdat(imc).eta^2*ipsig(3,3);
  ipsig(1,2)=ipsig(1,2)-IPdat(imc).eta*IPdat(imc).etap*ipsig(3,3);
  ipsig(2,1)=ipsig(1,2);
  ipsig(2,2)=ipsig(2,2)-IPdat(imc).etap^2*ipsig(3,3);

  % Get IP twiss parameters
  IPdat(imc).emit=sqrt(ipsig(1,1)*ipsig(2,2)-ipsig(1,2)^2);
  IPdat(imc).beta=ipsig(1,1)/IPdat(imc).emit;
  IPdat(imc).alpha=-ipsig(1,2)/IPdat(imc).emit;

  % Get beta waist dispersion and twiss parameters
  R=diag(ones(1,3));L=zeros(3,3);L(1,2)=1;
  IPdat(imc).wpos=fminsearch(@(x) minWaist(x,R,L,ipsig,1),0,optimset('Tolx',1e-8,'TolFun',0.1e-9^2));
  wsig=(R+L.*IPdat(imc).wpos)*ipsig*(R+L.*IPdat(imc).wpos)';
  R=R+L.*IPdat(imc).wpos;
  IPdat(imc).weta=IPdat(imc).eta*R(1,1)+IPdat(imc).etap*R(1,2);
  IPdat(imc).wetap=IPdat(imc).etap*R(2,1)+IPdat(imc).etap*R(2,2);
  IPdat(imc).weta=wsig(1,3)/wsig(3,3);
  IPdat(imc).wetap=wsig(2,3)/wsig(3,3);
  IPdat(imc).wemit=sqrt(wsig(1,1)*wsig(2,2)-wsig(1,2)^2);
  IPdat(imc).wbeta=wsig(1,1)/IPdat(imc).wemit;
  IPdat(imc).walpha=-wsig(1,2)/IPdat(imc).wemit;
end
fn=fieldnames(IPdat);
for ifn=1:length(fn)
  ipd.mfit.(fn{ifn})=mean([IPdat.(fn{ifn})]);
  ipd.mfit.([fn{ifn} '_err'])=std([IPdat.(fn{ifn})]);
end
IPdat=ipd;

% Twiss from parabolic fit method
brho = BEAMLINE{indqd0(1)}.P / 0.299792458;
bqd0=psqd0.*BEAMLINE{indqd0(1)}.B.*2;
kqd0=bqd0./brho;
[stat R_q2ip]=RmatAtoB(indqd0(end)+1,ip);
nmc=100;
pemit=zeros(1,nmc); pbeta=zeros(1,nmc);
for imc=1:nmc
  etadataerr=etadata+randn(size(etadata)).*detadata;
  sdataerr=sdata+randn(size(sdata)).*dsdata;
  % Subtract fitted dispersion from beam size
  scor=sdataerr - etadataerr.^2.*dE^2;
  % Fit parabola to data
  pfit=noplot_parab(kqd0,scor,dsdata.^2+detadata.^4.*dE^4);
  % Get Twiss parameters from fit (at waist)
  if dim==1
    pemit(imc)=sqrt(pfit(1)*pfit(3))/R_q2ip(1,2);
    pbeta(imc)=sqrt(pfit(3)/pfit(1))*R_q2ip(1,2);
  else
    pemit(imc)=sqrt(pfit(1)*pfit(3))/R_q2ip(3,4);
    pbeta(imc)=sqrt(pfit(3)/pfit(1))*R_q2ip(3,4);
  end
end
% figure
% plot_parab(kqd0,sdata-etadata.^2.*dE^2,dsdata.^2+detadata.^4.*dE^4);
IPdat.pfit.emit=mean(pemit);
IPdat.pfit.emit_err=std(pemit);
IPdat.pfit.beta=mean(pbeta);
IPdat.pfit.beta_err=std(pbeta);

% Return warnings to their initial state
for istat=1:length(warnstat)
  warning(warnstat(istat).state,warnstat(istat).identifier);
end


function chi2 = minWaist(x,R,L,sig,dir)
newsig=(R+L.*x(1))*sig*(R+L.*x(1))';
chi2=newsig(dir,dir);

function d = getQRmat(indqd0,ip,psqd0)
global PS BEAMLINE
[stat R]=RmatAtoB(indqd0(1),ip); if stat{1}~=1; error(stat{2}); end;
d.Rn{1}=R([1:2 6],[1:2 6]);
d.Rn{2}=R([3:4 6],[3:4 6]);
psinit=PS(BEAMLINE{indqd0(1)}.PS).Ampl;
for iqd0=1:length(psqd0)
  PS(BEAMLINE{indqd0(1)}.PS).Ampl=psqd0(iqd0);
  [stat R]=RmatAtoB(indqd0(1),ip); if stat{1}~=1; error(stat{2}); end;
  d.R{iqd0,1}=R([1:2 6],[1:2 6]);
  d.R{iqd0,2}=R([3:4 6],[3:4 6]);
  d.A(iqd0,1)=R(1,1)^2;
  d.B(iqd0,1)=2*R(1,1)*R(1,2);
  d.C(iqd0,1)=R(1,2)^2;
  d.D(iqd0,1)=R(1,1)*R(6,6);
  d.E(iqd0,1)=R(1,2)*R(6,6);
  d.A(iqd0,2)=R(3,3)^2;
  d.B(iqd0,2)=2*R(3,3)*R(3,4);
  d.C(iqd0,2)=R(3,4)^2;
  d.D(iqd0,2)=R(3,3)*R(6,6);
  d.E(iqd0,2)=R(3,4)*R(6,6);
end
PS(BEAMLINE{indqd0(1)}.PS).Ampl=psinit;

