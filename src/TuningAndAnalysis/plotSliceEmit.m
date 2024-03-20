function [dat,fh]=plotSliceEmit(beam,ibunch,dl,doplot)
% plotSliceEmit(beam,ibunch,dl [,doplot])

fh=NaN(1,4);
if ~exist('doplot','var')
  doplot=false;
elseif ~islogical(doplot)
  fh=doplot;
  doplot=true;
end

ns=~beam.Bunch(ibunch).stop;
z1=min(beam.Bunch(ibunch).x(5,ns)); z2=max(beam.Bunch(ibunch).x(5,ns));
zslice=linspace(z1,z2,ceil((z2-z1)/dl));

nx=zeros(1,length(zslice)-1); ny=nx; Q=nx; de=nx; xpos=nx; ypos=nx; betax=nx; alphax=nx; betay=nx; alphay=nx; E=nx; bmx=nx; bmy=nx;
for islice=1:length(zslice)-1
  B=beam;
  sel=ns&beam.Bunch(ibunch).x(5,:)>=zslice(islice)&beam.Bunch(ibunch).x(5,:)<zslice(islice+1);
  B.Bunch(ibunch).x=beam.Bunch(ibunch).x(:,sel); B.Bunch(ibunch).Q=beam.Bunch(ibunch).Q(sel); B.Bunch(ibunch).stop=beam.Bunch(ibunch).stop(sel);
  [ex,ey]=GetNEmitFromBeam(B,(ibunch));
  nx(islice)=ex; ny(islice)=ey;
  Q(islice)=sum(beam.Bunch(ibunch).Q(sel));
  de(islice)=std(beam.Bunch(ibunch).x(6,sel))./mean(beam.Bunch(ibunch).x(6,sel));
  E(islice)=mean(beam.Bunch(ibunch).x(6,sel));
  xpos(islice)=mean(beam.Bunch(ibunch).x(1,sel)); ypos(islice)=mean(beam.Bunch(ibunch).x(3,sel));
  [Tx,Ty]=GetUncoupledTwissFromBeamPars(B,1);
  betax(islice)=real(Tx.beta); alphax(islice)=real(Tx.alpha);
  betay(islice)=real(Ty.beta); alphay(islice)=real(Ty.alpha);
end
maxQ=max(Q);
maxE=max([nx ny]); minE=min([nx ny]);
qsel=Q>maxQ*1e-2;
dz=diff(zslice(1:2));
zp=zslice(1:end-1)+dz/2;
clight=2.99792458e8; % speed of light (m/sec)
I=Q./(dz/clight);
[~,ind]=max(I);
for islice=1:length(zslice)-1
  bmx(islice)=bmag(betax(ind),alphax(ind),betax(islice),alphax(islice));
  bmy(islice)=bmag(betay(ind),alphay(ind),betay(islice),alphay(islice));
end
dat.nx=nx; dat.ny=ny; dat.Q=Q; dat.de=de; dat.zp=zp; dat.betax=betax; dat.betay=betay; dat.alphax=alphax; dat.alphay=alphay; dat.I=I; dat.E=E;
dat.bmag_x=bmx; dat.bmag_y=bmy;
if doplot
  if isnan(fh(1))
    fh(1)=figure;
  else
    figure(fh(1))
  end
  yyaxis left
  if (maxE/minE)>100
    semilogy(zp(qsel).*1e6,nx(qsel).*1e6,'*',zp(qsel).*1e6,ny(qsel).*1e6,'o')
  else
    plot(zp(qsel).*1e6,nx(qsel).*1e6,'*',zp(qsel).*1e6,ny(qsel).*1e6,'o')
  end
  xlabel('z [\mum]'); ylabel('\gamma\epsilon [\mum-rad]'); grid on;
 
  yyaxis right
  plot(zp(qsel).*1e6,I(qsel).*1e-3,'+');
  ylabel('I [kA]');
   legend({'\gamma\epsilon_x' '\gamma\epsilon_y' 'I_{pk}'});
  %
  if isnan(fh(2))
    fh(2)=figure;
  else
    figure(fh(2));
  end
  yyaxis left, hold on
  plot(zp(qsel).*1e6,de(qsel),'*'); hold off;
  xlabel('z [\mum]'); ylabel('\delta_E/E'); grid on;
  yyaxis right, hold on
  clight=2.99792458e8; % speed of light (m/sec)
  I=Q./(dz/clight);
  plot(zp(qsel).*1e6,I(qsel).*1e-3,'+'); hold off;
  ylabel('I [kA]');
  legend({'\delta_E/E' 'I_{pk}'});
  %
  if isnan(fh(3))
    fh(3)=figure;
  else
    figure(fh(3));
  end
  yyaxis left, hold on
  plot(zp(qsel).*1e6,xpos(qsel).*1e6,'*',zp(qsel).*1e6,ypos(qsel).*1e6,'o'); hold off;
  xlabel('z [\mum]'); ylabel('Mean Slice Position [\mum]'); grid on;
  yyaxis right, hold on
  clight=2.99792458e8; % speed of light (m/sec)
  I=Q./(dz/clight);
  plot(zp(qsel).*1e6,I(qsel).*1e-3,'+'); hold off;
  ylabel('I [kA]');
  legend({'<X>' '<Y>' 'I_{pk}'});
   %
  if isnan(fh(4))
    fh(4)=figure;
  else
    figure(fh(4));
  end
  yyaxis left, hold on
  plot(zp(qsel).*1e6,bmx(qsel),'*',zp(qsel).*1e6,bmy(qsel),'o'); hold off;
  xlabel('z [\mum]'); ylabel('BMAG_{x,y} [m]'); grid on;
  yyaxis right, hold on
  clight=2.99792458e8; % speed of light (m/sec)
  I=Q./(dz/clight);
  plot(zp(qsel).*1e6,I(qsel).*1e-3,'+'); hold off;
  ylabel('I [kA]');
  legend({'<X>' '<Y>' 'I_{pk}'});
end