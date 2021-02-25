% Make list of all shared object libs for compiled mex binaries and copy
% into Lucretia Libs directory
exclude={'libstdc++','libc'};
if ismac
  [~,solibs1]=system(sprintf('find ../* -name "*.%s" | xargs otool -L | grep "/Users"',mexext));
  [~,solibs2]=system(sprintf('find ../* -name "*.%s" | xargs otool -L | grep "/usr/local"',mexext));
  solibs=[solibs1 solibs2];
else
  [~,solibs]=system(sprintf('find ../* -name "*.%s" | xargs ldd | grep "=> /"',mexext));
end
solibs=unique(regexp(strsplit(solibs,'\n'),'/(\S+)','match','once'));
solibs=solibs(cellfun(@(x) ~isempty(x),solibs));
for ilib=1:length(solibs)
  % Ignore Matlab so's
  if ismac
    if ~isempty(regexp(solibs{ilib},'maci64', 'once'))
      continue
    end
  else
    if ~isempty(regexp(solibs{ilib},'glnxa64', 'once'))
      continue
    end
  end
  fprintf('Copying %s to ../Libs/%s ...\n',solibs{ilib},mexext)
  try
    system(sprintf('cp -b %s ../Libs/%s',solibs{ilib},mexext));
  catch
  end
end
