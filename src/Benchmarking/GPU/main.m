% Benchmark lucretia tracking through example lattice on GPU and compare
% with CPU
%==========================================================================
clear
clear global BEAMLINE WF PS
global BEAMLINE PS WF KLYSTRON
close all

% parameters
docpu=true;
dogpu=false;
% npart=linspace(1e4,1e6,10);
npart=1e6; % 1e7 = 281s CPU, 13s GPU (~2 GB memory useage)

% Choose GPU device
if dogpu
  g=gpuDevice(1);
  reset(g);
  display(g)
end

% step 1:  load the lattice
load FACET2e.mat BEAMLINE KLYSTRON PS WF istart Initial

% set desired tracking flags
SetTrackFlags('ZMotion',1,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',0,1,length(BEAMLINE)) ;
SetTrackFlags('LRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('LRWF_ERR',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetBPMData',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetBPMBeamPars',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetInstData',0,1,length(BEAMLINE)) ;
SetTrackFlags('SynRad',2,1,length(BEAMLINE)) ;
SetTrackFlags('Aper',0,1,length(BEAMLINE)) ;
SetTrackFlags('LorentzDelay',1,1,length(BEAMLINE)) ;
SetTrackFlags('GetSBPMData',0,1,length(BEAMLINE)) ;
CSRon=0; % 1= Coherent Sync. Rad. on, 0= CSR off
LSCon=0; % 1= Longitudinal space charge on, 0= LSC off

SetTrackFlags('CSR',0,1,length(BEAMLINE)) ;
SetTrackFlags('LSC',0,1,length(BEAMLINE)) ;
% -- Sync. Radiation settings
if CSRon
  eleSR=findcells(BEAMLINE,'Class','SBEN');
  for iele=eleSR
    BEAMLINE{iele}.TrackFlag.CSR=-1; % automatic binning
    BEAMLINE{iele}.TrackFlag.CSR_SmoothFactor=0; % automated smoothing of longitudinal profile
    BEAMLINE{iele}.TrackFlag.CSR_DriftSplit=25;
    BEAMLINE{iele}.TrackFlag.Split=25;
  end
end
% -- Space charge settings
if LSCon
  for iele=findcells(BEAMLINE,'TrackFlag')
    BEAMLINE{iele}.TrackFlag.LSC=1;
    BEAMLINE{iele}.TrackFlag.LSC_storeData=0;
    % Set NBPM on LCAV elements to ensure 0.1m drift sections for
    % application of LSC
    if strcmp(BEAMLINE{iele}.Class,'LCAV')
      BEAMLINE{iele}.NBPM=LSCon*BEAMLINE{iele}.L/0.1;
      BEAMLINE{iele}.GetSBPMData=1;
      BEAMLINE{iele}.GetInstData=1;
    end
  end
end

% Define indices for tracking
iL1=findcells(BEAMLINE,'Name','BEGL1F'); % start of L1
iL2=findcells(BEAMLINE,'Name','BEGL2F'); % start of L2
iL3=findcells(BEAMLINE,'Name','BEGL3F'); % start of L3
iFF=findcells(BEAMLINE,'Name','MFFF'); % start of FFS
ip=findcells(BEAMLINE,'Name','MIP'); % IP
trackind=[istart iL1 iL2 iL3 iFF ip];
for ipart=1:length(npart)
  % generate a beam to track through the lattice
  beam = MakeBeam6DGauss(Initial,floor(npart(ipart)),4,1) ;
  % track and time beam tracking through CPU and display tracked results
  if docpu
    disp('...tracking beam through lattice (CPU)...')
    tic
    beamout=beam;
    for iti=2:length(trackind)
      [~,beamout,idat] = TrackThru(trackind(iti-1),trackind(iti),beamout,1,1,0) ;
      for ind=1:5
        beamout.Bunch.x(ind,~beamout.Bunch.stop)=beamout.Bunch.x(ind,~beamout.Bunch.stop)-mean(beamout.Bunch.x(ind,~beamout.Bunch.stop));
      end
    end
    cput(ipart)=toc;
    fprintf('CPU tracking, time taken = %.2f s\n',cput(ipart))
    [nx,ny] = GetNEmitFromBeam(beamout,1) ;
    fprintf('Final beam energy: %g GeV\n',mean(beamout.Bunch.x(6,~beamout.Bunch.stop)))
    disp(['Horizontal norm emittance = ',num2str(nx*1e6),' um']) ;
    disp(['Vertical norm emittance = ',num2str(ny*1e6),' um']) ;
    fprintf('Beam size at final element (sx, sy, sz): %g %g %g\n',std(beamout.Bunch.x(1,~beamout.Bunch.stop)),...
      std(beamout.Bunch.x(3,~beamout.Bunch.stop)),std(beamout.Bunch.x(5,~beamout.Bunch.stop)))
    fprintf('Energy spread at final element dE/E: %g (%%)\n',100*(std(beamout.Bunch.x(6,~beamout.Bunch.stop))/mean(beamout.Bunch.x(6,~beamout.Bunch.stop))))
    fprintf('Beam position at final element (x,x'',y,y'',z): %g %g %g %g %g\n',mean(beamout.Bunch.x(1,~beamout.Bunch.stop)),...
      mean(beamout.Bunch.x(2,~beamout.Bunch.stop)),mean(beamout.Bunch.x(3,~beamout.Bunch.stop)),...
      mean(beamout.Bunch.x(4,~beamout.Bunch.stop)),mean(beamout.Bunch.x(5,~beamout.Bunch.stop)))
    beamoutCPU=beamout;
    beamImage(beamout);
    if isempty(idat{1})
      disp('BPM data empty...')
    else
      BPMZplot(idat{1});
    end
    if ~isempty(idat{1}) && ~isempty(idat{1}(1).sigma) && ipart==1
      figure
      subplot(2,1,1), plot([idat{1}.S],arrayfun(@(x) sqrt(idat{1}(x).sigma(1,1))*1e6,1:length(idat{1})))
      xlabel('S / m'); ylabel('\sigma_x / um');
      subplot(2,1,2), plot([idat{1}.S],arrayfun(@(x) sqrt(idat{1}(x).sigma(3,3))*1e6,1:length(idat{1})))
      xlabel('S / m'); ylabel('\sigma_y / um');
    end
    idatc=idat;
    disp('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
  end
  
  % track and time beam tracking through CPU and display tracked results
  if dogpu
    disp('...tracking beam through lattice (GPU)...')
    beamout=beam;
    beamout.Bunch.x=gpuArray(beam.Bunch.x);
    beamout.Bunch.Q=gpuArray(beam.Bunch.Q);
    beamout.Bunch.stop=gpuArray(beam.Bunch.stop);
    tic
    for iti=2:length(trackind)
      [~,beamout,idat] = TrackThru_gpu(trackind(iti-1),trackind(iti),beamout,1,1,0) ;
      for ind=1:5
        beamout.Bunch.x(ind,~beamout.Bunch.stop)=beamout.Bunch.x(ind,~beamout.Bunch.stop)-mean(beamout.Bunch.x(ind,~beamout.Bunch.stop));
      end
    end
    gput(ipart)=toc;
    fprintf('GPU tracking, time taken = %.2f s\n',gput(ipart))
    beamout.Bunch.x=gather(beamout.Bunch.x) ;
    beamout.Bunch.Q=gather(beamout.Bunch.Q) ;
    beamout.Bunch.stop=gather(beamout.Bunch.stop) ;
    [nx,ny] = GetNEmitFromBeam(beamout,1) ;
    fprintf('Final beam energy: %g GeV\n',mean(beamout.Bunch.x(6,:)))
    disp(['Horizontal norm emittance = ',num2str(nx*1e6),' um']) ;
    disp(['Vertical norm emittance = ',num2str(ny*1e6),' um']) ;
    fprintf('Beam size at final element (sx, sy, sz): %g %g %g\n',std(beamout.Bunch.x(1,~beamout.Bunch.stop)),...
      std(beamout.Bunch.x(3,~beamout.Bunch.stop)),std(beamout.Bunch.x(5,~beamout.Bunch.stop)))
    fprintf('Energy spread at final element dE/E: %g (%%)\n',100*(std(beamout.Bunch.x(6,~beamout.Bunch.stop))/mean(beamout.Bunch.x(6,~beamout.Bunch.stop))))
    fprintf('Beam position at final element (x,x'',y,y'',z): %g %g %g %g %g\n',mean(beamout.Bunch.x(1,~beamout.Bunch.stop)),...
      mean(beamout.Bunch.x(2,~beamout.Bunch.stop)),...
      mean(beamout.Bunch.x(3,~beamout.Bunch.stop)),mean(beamout.Bunch.x(4,~beamout.Bunch.stop)),mean(beamout.Bunch.x(5,~beamout.Bunch.stop)))
    if isempty(idat{1})
      disp('BPM data empty...')
    else
      BPMZplot(idat{1});
    end
    if ~isempty(idat{1}) && ~isempty(idat{1}(1).sigma) && ipart==1
      figure(1)
      subplot(2,1,1), plot([idat{1}.S],arrayfun(@(x) sqrt(idat{1}(x).sigma(1,1))*1e6,1:length(idat{1})))
      xlabel('S / m'); ylabel('\sigma_x / um');
      subplot(2,1,2), plot([idat{1}.S],arrayfun(@(x) sqrt(idat{1}(x).sigma(3,3))*1e6,1:length(idat{1})))
      xlabel('S / m'); ylabel('\sigma_y / um');
      drawnow
    end
    disp('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    if docpu
      for idim=1:6
        fprintf('Beam size Diff (%%) dim=%d: %g\n',idim,100*(std(beamoutCPU.Bunch.x(idim,~beamoutCPU.Bunch.stop))-...
          std(beamout.Bunch.x(idim,~beamout.Bunch.stop)))/std(beamoutCPU.Bunch.x(idim,~beamoutCPU.Bunch.stop)))
      end
      fprintf('GPU:CPU speedup = %.1fX\n',cput(ipart)/gput(ipart))
      disp('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    end
  end
end

if length(npart)>1
  figure
  subplot(2,1,1)
  plot(npart,cput,npart,gput)
  grid on
  xlabel('# Macro-Particles')
  ylabel('Tracking Time / s')
  legend({'CPU',sprintf('%s GPU',g.Name)},'Location','NorthWest')
  subplot(2,1,2)
  plot(npart,cput./gput)
  grid on
  xlabel('# Macro-Particles')
  ylabel('Speedup Factor')
end

