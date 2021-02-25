function [Tx,Ty] = GetUncoupledTwissFromBeamPars(beam,bunchno)
%
% GETUNCOUPLEDTWISSFROMBEAMPARS Compute the uncoupled Twiss parameters of a
% beam.
%
% [Tx,Ty] = GetUncoupledTwissFromBeamPars(beam,bunchno) uses the beam
%    matrix of a beam to compute the uncoupled Twiss parameters of that
%    beam.  Argument bunchno is the bunch number to be used in the
%    computation.  If the bunch number is zero, all bunches are considered
%    as an ensemble.  If the bunch number is -1, each bunch's parameters
%    are computed and returned.
%

%==========================================================================

  Tx = struct('beta',[],'alpha',[],'eta',[],'etap',[],'nu',[]) ;
  Ty = Tx ;
  
  [x,sig] = GetBeamPars(beam,bunchno) ;
  xdim = size(x) ; nbunch = xdim(2) ;
  
  for count = 1:nbunch
      
% compute dispersions in m/GeV      
      
      sigx = sig(1:2,1:2,count) ; sigy = sig(3:4,3:4,count) ;
      P = x(6,count) ;
      sig66 = sig(6,6,count) ;
      etax  = sig(1,6,count) / sig66  ;
      etapx = sig(2,6,count) / sig66  ;
      etay  = sig(3,6,count) / sig66  ;
      etapy = sig(4,6,count) / sig66  ;
      
% suppress the dispersions out of the sigma matrix
      
      sigx(1,1) = sigx(1,1) - etax  * etax  * sig66 ;
      sigx(1,2) = sigx(1,2) - etax  * etapx * sig66 ;
      sigx(2,1) = sigx(1,2) ;
      sigx(2,2) = sigx(2,2) - etapx * etapx * sig66 ;
      
      sigy(1,1) = sigy(1,1) - etay  * etay  * sig66 ;
      sigy(1,2) = sigy(1,2) - etay  * etapy * sig66 ;
      sigy(2,1) = sigy(1,2) ;
      sigy(2,2) = sigy(2,2) - etapy * etapy * sig66 ;
      
% compute emittance, beta, and alpha

      ex = sqrt(det(sigx)) ; ey = sqrt(det(sigy)) ;
      bx = sigx(1,1) / ex ; ax = -sigx(1,2) / ex ;
      by = sigy(1,1) / ey ; ay = -sigy(1,2) / ey ;
      
% put data into data structures

      Tx(count).beta = bx ;      Tx(count).alpha = ax ; 
      Tx(count).eta = etax ; Tx(count).etap = etapx ;
      Tx(count).nu = 0 ;
      
      Ty(count).beta = by ;      Ty(count).alpha = ay ; 
      Ty(count).eta = etay ; Ty(count).etap = etapy ;
      Ty(count).nu = 0 ;
      
  end
  
% and that's it.  