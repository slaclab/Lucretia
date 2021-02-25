function [stat,W] = ParseSRWF( filename, BinWidth )

% PARSESRWF Read a short-range wakefield file from disk
%
%   [stat,W] = ParseSRWF( filename, BinWidth ) loads the z and kick data
%       from a short range wakefield file in the LIAR format (3 column =
%       indx, z, wake, with comment lines indicated by parentheses).  A
%       data structure with z positions and wake values as its fields is
%       returned.  The BinWidth tells the width of bins (in sigmas) for
%       histogramming the beam during convolution, and is returned as a
%       field of the wakefield data structure.  The wakefield is returned
%       in W; stat is a cell array with text messages in fields 2 and
%       after, and a status integer (1 == success, 0 == failure) for
%       stat{1}.

%=========================================================================

  stat{1} = 1 ; W = [] ;
  fp = fopen(filename,'r') ;
  if (fp == -1)
      stat{1} = 0 ;
      stat{2} = ['Unable to open wakefield data file ',filename] ;
      return ;
  end
  
  W.z = [] ; W.K = [] ; W.BinWidth = BinWidth ;
  
  reading = 1 ;
  
  while (reading == 1)
      
    tline = fgets(fp) ;
    if (tline == -1)
        fclose(fp) ;
        return
    end
    
%   if this is a comment line, skip it

    if ( (strcmp(tline(1),'(')) | ...
         (strcmp(tline(1),'#')) | ...
         (strcmp(tline(1),'!')) | ...
         (strcmp(tline(1),'%'))       )
      continue ;
    end

% otherwise attempt to read

    A = sscanf(tline,'%f') ;
    if (length(A) ~=3)
        stat{1} = 0 ;
        stat{2} = ['Can''t parse line in file ',filename,': ',...
            num2str(tline)] ;
        return ;
    end
    
    W.z = [W.z A(2)] ;
    W.K = [W.K A(3)] ;
    
  end