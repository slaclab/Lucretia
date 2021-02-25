warning('off','MATLAB:mex:GccVersion_link');
cd g4track
try
  build
catch ME
  cd ..
  rethrow(ME)
end
cd ..
build cpu-g4 install
