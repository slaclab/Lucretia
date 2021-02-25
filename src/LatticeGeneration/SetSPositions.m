function SetSPositions(i,j,S0)
%
% SETSPOSITIONS Initialize element S positions in BEAMLINE.
%
% SetSPositions( i, j, S0 ) will initialize the S positions of BEAMLINE{i}
%    through BEAMLINE{j} using the element lengths and beginning with
%    BEAMLINE{i}.S = S0.
%

global BEAMLINE;
%
% do the loop
%
  S = S0 ;
  for count = i:j

      BEAMLINE{count}.S = S ;
      if (isfield(BEAMLINE{count},'L'))
        S = S + BEAMLINE{count}.L ;
      end
      
  end
%