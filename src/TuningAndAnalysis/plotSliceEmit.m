function [nx,ny,Q,I,de,zp,fh]=plotSliceEmit(beam,dl,doplot)
% plotSliceEmit(beam,dl [,doplot])

fh=NaN(1,3);
if ~exist('doplot','var')
  doplot=true;
elseif ~islogical(doplot)
  fh=doplot;
  doplot=true;
end

ns=~beam.Bunch.stop;
z1=min(beam.Bunch.x(5,ns)); z2=max(beam.Bunch.x(5,ns));
zslice=linspace(z1,z2,ceil((z2-z1)/dl));

nx=zeros(1,length(zslice)-1); ny=nx; Q=nx; de=nx; xpos=nx; ypos=nx;
for islice=1:length(zslice)-1
  B=beam;
  sel=ns&beam.Bunch.x(5,:)>=zslice(islice)&beam.Bunch.x(5,:)<zslice(islice+1);
  B.Bunch.x=beam.Bunch.x(:,sel); B.Bunch.Q=beam.Bunch.Q(sel); B.Bunch.stop=beam.Bunch.stop(sel);
  [ex,ey]=GetNEmitFromBeam(B,1);
  nx(islice)=ex; ny(islice)=ey;
  Q(islice)=sum(beam.Bunch.Q(sel));
  de(islice)=std(beam.Bunch.x(6,sel))./mean(beam.Bunch.x(6,sel));
  xpos(islice)=mean(beam.Bunch.x(1,sel)); ypos(islice)=mean(beam.Bunch.x(3,sel));
end
maxQ=max(Q);
maxE=max([nx ny]); minE=min([nx ny]);
qsel=Q>maxQ*1e-2;
dz=diff(zslice(1:2));
zp=zslice(1:end-1)+dz/2;
clight=2.99792458e8; % speed of light (m/sec)
I=Q./(dz/clight);
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
  xlabel('z [\mum]'); ylabel('\gamma\epsilon [\mum-rad]');
 
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
  xlabel('z [\mum]'); ylabel('\delta_E/E');
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
  xlabel('z [\mum]'); ylabel('Mean Slice Position [\mum]');
  yyaxis right, hold on
  clight=2.99792458e8; % speed of light (m/sec)
  I=Q./(dz/clight);
  plot(zp(qsel).*1e6,I(qsel).*1e-3,'+'); hold off;
  ylabel('I [kA]');
  legend({'<X>' '<Y>' 'I_{pk}'});
end