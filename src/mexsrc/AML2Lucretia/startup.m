comp = evalc('!hostname');
comp = comp(1:end-1);

switch lower(comp)
    case {'iris01','iris02','iris03'}
        
    case 'ilc-molloylnx.slac.stanford.edu'
      addpath /home/smolloy/mfiles
      lpaths = dir('/home/smolloy/Lucretia/src/');
      for dirnum=3:length(lpaths)
        if lpaths(dirnum).isdir
          addpath(['/home/smolloy/Lucretia/src/' lpaths(dirnum).name]);
        end
      end
      addpath /home/smolloy/Lucretia/src/
        
    otherwise
        disp('Computer not recognised.')
        
end
