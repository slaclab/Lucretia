if ~isdeployed
  LUCRETIADIR='~/Lucretia';
  
  % Add Lucretia directories
  if ~exist(LUCRETIADIR,'dir') || ~exist(fullfile(LUCRETIADIR,'src'),'dir')
    error('Lucretia directory not found, edit LUCRETIADIR variable in startup.m and ensure current version of Lucretia is installed there')
  end
  ldir=fullfile(LUCRETIADIR,'src');
  D=dir(ldir);
  for ind=3:length(D)
    if D(ind).isdir && isempty(strfind(D(ind).name,'svn')) && isempty(strfind(D(ind).name,'Floodland')) && ...
        isempty(strfind(D(ind).name,'mexsrc'))
      addpath(fullfile(ldir,D(ind).name),'-END');
    end
  end
end
