%
% execute the NLC-FF Lucretia example
%
  disp('...parsing the deck...')
  clear global BEAMLINE ;
  randn('state',0) ;
  global BEAMLINE ;
  [statall,Initial] = XSIFToLucretia(...
    'elec_bdsh_250GeV_master.xsif','BDSH250') ;
  disp('...generating Twiss parameters...')
  [stat,T] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
  statall{1} = [statall{1} stat{1}] ;
  statall = AddStackToStack(statall,stat) ;
  figure 
  plot(T.S,T.betay,'r') ;
  ylabel('\beta_{x,y} [m]') ;
  xlabel('S, m')
  hold on
  [ax,h1,h2] = plotyy(T.S,T.betax,T.S,T.etax) ;
  set(get(ax(2),'Ylabel'),'String','\eta_x [m]') ;
  figure 
  plot(T.S,sqrt(T.betay),'r') ;
  ylabel('sqrt(\beta_{x,y}) [sqrt(m)]') ;
  xlabel('S, m')
  hold on
  [ax,h1,h2] = plotyy(T.S,sqrt(T.betax),T.S,T.etax) ;
  set(get(ax(2),'Ylabel'),'String','\eta_x [m]') ;
  disp(['::: Beta* values:  x = ',...
        num2str(T.betax(length(BEAMLINE)+1)*1000),...
        ' mm, y = ',...
        num2str(T.betay(length(BEAMLINE)+1)*1000000),...
        ' um :::']) ;
  disp('...generating beam with nominal momentum...') ;
  beam0 = MakeBeam6DGauss(Initial,10000,5,0) ;
  [x0,sig0] = GetBeamPars(beam0,1) ;
  [nx,ny,nt] = GetNEmitFromSigmaMatrix(x0(6),sig0) ;
  disp('...tracking thru lattice...')
  [stat,beamout] = TrackThru(1,length(BEAMLINE),beam0,1,1,0) ;
  statall{1} = [statall{1} stat{1}] ;
  statall = AddStackToStack(statall,stat) ;
  [x,sig] = GetBeamPars(beamout,1) ;
  figure
  subplot(2,2,3) ;
  plot(beamout.Bunch(1).x(1,:)*1e9,beamout.Bunch(1).x(3,:)*1e9,'.') ;
  axis([-4000 4000 -20 20]) ;
  xlabel('x [nm]') ;
  ylabel('y [nm]') ;
  subplot(2,2,1) ;
  [center,height] = BeamHistogram(beamout,1,1,0.25) ;
  bar(center*1e9,height*1e9) ;
  axis([-4000 4000 0 0.15]) ;
  title(['RMS = ',num2str(sqrt(sig(1,1))*1e9),' nm']) ;
  subplot(2,2,4) ;
  [center,height] = BeamHistogram(beamout,1,3,0.25) ;
  barh(center*1e9,height*1e9) ;
  axis([0 0.15 -20 20]) ;
  title(['RMS = ',num2str(sqrt(sig(3,3))*1e9),' nm']) ;
  [nx,ny,nt] = GetNEmitFromSigmaMatrix(x(6),sig,'normalmode') ;
  disp('::: Normal-mode normalized emittances:  ') ;
  disp(['   gepsx = ',num2str(nx*1e6),' um, ']) ;
  disp(['   gepsy = ',num2str(ny*1e9),' nm  ']) ;
  disp('...performing energy scan...') 
  mscan = linspace(250*0.99,250*1.01,11) ; 
  dp = mscan - 250 ; delta = dp / 250 ;
  sigx = zeros(1,length(mscan)) ;
  sigy = sigx ; 
  sigpx = sigx ; sigpy = sigx ;
  for count = 1:length(mscan)
    beam = beam0 ;
    beam.Bunch(1).x(6,:) = beam.Bunch(1).x(6,:) + dp(count) ;
    [stat,beamout] = TrackThru(1,length(BEAMLINE),beam,1,1,0) ;
    [x,sig] = GetBeamPars(beamout,1) ;
    sigx(count) = sqrt(sig(1,1)) ;
    sigpx(count) = sqrt(sig(2,2)) ;
    sigy(count) = sqrt(sig(3,3)) ;
    sigpy(count) = sqrt(sig(4,4)) ;
  end
  figure ;
  plot(delta*100, 100*(sigx / sigx(6) - 1)) ;
  hold on
  plot(delta*100, 100*(sigy / sigy(6) - 1),'r') ;
  xlabel('\delta [%]') ;
  ylabel('\Delta\sigma_{x,y}^* [%]') ;
  title('Fractional Variation in Beam Size vs Momentum') ;
  figure ;
  plot(delta*100, 100*(sigpx / sigpx(6) - 1)) ;
  hold on
  plot(delta*100, 100*(sigpy / sigpy(6) - 1),'r') ;
  xlabel('\delta [%]') ;
  ylabel('\Delta\sigma_{px,py}^* [%]') ;
  title('Fractional Variation in Angular Divergence vs Momentum') ;
    
