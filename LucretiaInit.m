function LucretiaInit(InstallDir,whichmex)
if ~isdeployed
  % Add Lucretia directories
  if ~exist(InstallDir,'dir') || ~exist(fullfile(InstallDir,'src'),'dir')
    error('Lucretia directory not found')
  end
  ldir=fullfile(InstallDir,'src');
  D=dir(ldir);
  for ind=3:length(D)
    if D(ind).isdir && isempty(strfind(D(ind).name,'svn')) && ...
        isempty(strfind(D(ind).name,'mexsrc'))
      addpath(fullfile(ldir,D(ind).name),'-END');
    end
  end
  % Setup required mex files
  if exist('whichmex','var')
    switch lower(whichmex)
      case 'cpu'
        copyfile(fullfile(InstallDir,'src','Tracking',sprintf('TrackThru_cpu.%s',mexext)),fullfile(InstallDir,'src','Tracking',sprintf('TrackThru.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','RMatrix',sprintf('GetRmats_cpu.%s',mexext)),fullfile(InstallDir,'src','RMatrix',sprintf('GetRmats.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','RMatrix',sprintf('RmatAtoB_cpu.%s',mexext)),fullfile(InstallDir,'src','RMatrix',sprintf('RmatAtoB.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','LatticeVerification',sprintf('VerifyLattice_cpu.%s',mexext)),fullfile(InstallDir,'src','LatticeVerification',sprintf('VerifyLattice.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','Twiss',sprintf('GetTwiss_cpu.%s',mexext)),fullfile(InstallDir,'src','Twiss',sprintf('GetTwiss.%s',mexext)));
      case 'cpu-g4'
        copyfile(fullfile(InstallDir,'src','Tracking',sprintf('TrackThru_cg4.%s',mexext)),fullfile(InstallDir,'src','Tracking',sprintf('TrackThru.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','RMatrix',sprintf('GetRmats_g4.%s',mexext)),fullfile(InstallDir,'src','RMatrix',sprintf('GetRmats.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','RMatrix',sprintf('RmatAtoB_g4.%s',mexext)),fullfile(InstallDir,'src','RMatrix',sprintf('RmatAtoB.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','LatticeVerification',sprintf('VerifyLattice_g4.%s',mexext)),fullfile(InstallDir,'src','LatticeVerification',sprintf('VerifyLattice.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','Twiss',sprintf('GetTwiss_g4.%s',mexext)),fullfile(InstallDir,'src','Twiss',sprintf('GetTwiss.%s',mexext)));
      case 'gpu'
        copyfile(fullfile(InstallDir,'src','Tracking',sprintf('TrackThru_gpu.%s',mexext)),fullfile(InstallDir,'src','Tracking',sprintf('TrackThru.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','RMatrix',sprintf('GetRmats_cpu.%s',mexext)),fullfile(InstallDir,'src','RMatrix',sprintf('GetRmats.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','RMatrix',sprintf('RmatAtoB_cpu.%s',mexext)),fullfile(InstallDir,'src','RMatrix',sprintf('RmatAtoB.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','LatticeVerification',sprintf('VerifyLattice_cpu.%s',mexext)),fullfile(InstallDir,'src','LatticeVerification',sprintf('VerifyLattice.%s',mexext)));
        copyfile(fullfile(InstallDir,'src','Twiss',sprintf('GetTwiss_cpu.%s',mexext)),fullfile(InstallDir,'src','Twiss',sprintf('GetTwiss.%s',mexext)));
      otherwise
        error('Unknown mex target')
    end
  end
end