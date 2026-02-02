function T = sddstwiss(tfile,doplot)
% SDDSTWISS Process (and plot) sdds Twiss Parameter data

if ~exist('doplot','var')
  doplot = false ;
end

if ~exist('tfile','var') || ~exist(tfile,'file')
  error('Must provide valid Twiss SDDS input file');
end

s = sddsload(char(tfile)) ;

T.S = s.column.s.page1 ;
T.P = s.column.pCentral0.page1 .* 0.511e-3 ; % GeV/c
T.Name = s.column.ElementName.page1 ;
T.betax = s.column.betax.page1 ;
T.alphax = s.column.alphax.page1 ;
T.etax = s.column.etax.page1 ;
T.etapx = s.column.etaxp.page1 ;
T.nux = s.column.psix.page1 ;
T.betay = s.column.betay.page1 ;
T.alphay = s.column.alphay.page1 ;
T.etay = s.column.etay.page1 ;
T.etapy = s.column.etayp.page1 ;
T.nuy = s.column.psiy.page1 ;

if doplot
  figure
  subplot(2,1,1), plot(T.S,T.betax,T.S,T.betay); xlabel('S [m]'); ylabel('\beta_{x,y} [m]');
  subplot(2,1,2), plot(T.S,T.etax,T.S,T.etay); xlabel('S [m]'); ylabel('\eta_{x,y} [m]');
end