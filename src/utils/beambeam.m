function [L0, D, H, sigma] = beambeam( N, sigma, beta, sigz )

%
% compute the geometric luminosity (actually integrated lumi
% per bunch crossing, in cm^{-2})
%
  L0 = N^2 / 4 / pi / sigma(1) / sigma(2) * 1e-4 ;
%
% compute the two disruptions
%
  D(1) = 2 * N * sigz * 2.8e-15 / relgamma / sigma(1) / (sum(sigma)) ;
  D(2) = D(1) * sigma(1)/sigma(2) ;
%
% compute the lumi enhancement factors
%
  H = LumiEnhance(D,beta,sigz) ;
  
  R = sigma(1) / sigma(2) ;
  f = (1+2*R^3) / (6*R^3) ;
  H0 = sqrt(H(1)) * H(2)^f ;
  H = [H0 H] ;
%
% package the upsilon, ngamma, and deltabs in the H vector
%
  upsilon = 5/6 * N * 2.8e-15 * relgamma * 3.86e-13 / sigz / sum(sigma) ;
  ngam = 2.54 * sigz / 3.86e-13 / relgamma / 137 * ...
         upsilon / sqrt(1+upsilon^(2/3)) ;
  deltabs = 1.24 * sigz / 3.86e-13 / relgamma / 137 * ...
         upsilon^2 / (1+(1.5*upsilon)^(2/3))^2 ;
     
  H = [H upsilon ngam deltabs] ; 