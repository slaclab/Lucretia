function [stat,W] = ParseFrequencyDomainTLR( filename, BinWidth )
%
% PARSEFREQUENCYDOMAINTLR Parse a transverse, long range (multibunch)
% wakefield in the frequency domain.
%
% [stat,W] = ParseFrequencyDomainTLR( filename, BinWidth ) parses a file
%    which contains the frequency-domain description of a multi-bunch long
%    range transverse wakefield.  The file is assumed to be in Jones
%    format:
%
% -> Comments are indicated by #
% -> Data is arranged in 3-5 columns:
%    Central Frequency [GHz], Q1, Kick [V/C/m^2], [dFreq [GHz] Q2]
%
% The structure W is in the correct format for WF.TLR, with frequencies
% moved to MHz and 1 kick factor per mode (WF.TLRErr requires 2).
%
% Return argument stat is a Lucretia status and message cell array (type
% help LucretiaStatus for more information), with stat{1} == 1 for success
% or stat{1} == 0 for failure.
%

%==========================================================================

  stat = InitializeMessageStack( ) ;
  W.Class = 'Frequency' ;
  W.Freq = [] ; W.Q = [] ; W.K = [] ; W.Tilt = [] ;
  W.BinWidth = BinWidth ;
  fp = fopen(filename,'r') ;
  if (fp==0)
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,['Unable to open file:  ',filename]) ;
    return 
  end
  
% loop over comments and modes

  reading = 1 ;
  
  while (reading == 1)
      
    tline = fgets(fp) ;
    if (tline == -1)
        fclose(fp) ;
        W.Freq = W.Freq * 1e3 ;
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
    if ( (length(A) > 5) & (length(A) < 3) )
        stat{1} = 0 ;
        stat = AddMessageToStack(stat,...
            ['Can''t parse line in file ',filename,': ',...
            num2str(tline)]) ;
        return ;
    end
    if (length(A) <= 3)
      A(4) = 0 ; 
    end
    if (length(A) <= 4)
      A(5) = A(2) ;
    end
    
    W.Freq = [W.Freq [A(1)+A(4) ; A(1)-A(4)]] ;
    W.Q = [W.Q [A(2) ; A(5)]] ;
    W.K = [W.K 2*A(3)] ;
    W.Tilt = [W.Tilt 0] ;
    
  end
% 
  