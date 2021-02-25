function PlotPZVariation( xvec, beamsuperstruc, beam0 )

% Produce a standard plot of variation in longitudinal phase space,
% given a structure of beam data and a reference beam

% start by getting the reference beam parameters 

  [x,sig] = GetBeamPars(beam0, 1) ;
  z0 = x(5) ; P0 = x(6) ; sigz0 = sqrt(sig(5,5)) ; 
  sigdelta0 = sqrt(sig(6,6)) / P0 ;
  betystar = 400e-6 ;

% define our vectors 

  dzovbety = zeros(1,length(beamsuperstruc)) ;
  dSigzovSigz = dzovbety ;
  dPovSigP = dzovbety ;
  dSigPovSigP = dzovbety ;
  
% loop over beams

  for count = 1:length(beamsuperstruc) 
    [x,sig] = GetBeamPars(beamsuperstruc(count), 1) ;
    dzovbety(count) = (x(5)-z0)/betystar ;
    dSigzovSigz(count) = (sqrt(sig(5,5))-sigz0)/sigz0 ;
    dPovSigP(count) = (x(6)-P0)/P0/sigdelta0 ;
    dSigPovSigP(count) = sqrt(sig(6,6)) / x(6) / sigdelta0 ;
  end
  figure 
  subplot(1,2,1) 
  plot(xvec,dPovSigP*100,'bo-') ;
  hold on
  plot(xvec,(dSigPovSigP-1)*100,'rs-') ;
  ylabel('Momentum Error [%]') ;
%  legend('\Delta{P}/\sigma_P','\Delta\sigma_P/\sigma_P') ;
  subplot(1,2,2) 
  plot(xvec,dzovbety*100,'bo-') ;
  hold on
  plot(xvec,(dSigzovSigz)*100,'rs-') ;
  ylabel('Time Error [%]') ;
%  legend('\Delta{z}/\beta_y^*','\Delta\sigma_z/\sigma_z') ;
  
