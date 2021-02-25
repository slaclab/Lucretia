function [x,sigma] = GetBeamPars( beam, bunch )

% GETBEAMPARS Compute the charge-weighted parameters of a Lucretia beam.
%
%    [x,sigma] = GetBeamPars( beam, bunch ) takes as arguments a Lucretia
%    beam data structure and a bunch number, and returns the centroid and
%    second moment matrix of that bunch number.  If the bunch number is
%    zero, all bunches are considered as an ensemble.  If the bunch number
%    is -1, each bunch's parameters are computed and returned in x and
%    sigma (x becomes a 6 x nbunch matrix, sigma becomes a 6 x 6 x nbunch
%    matrix).

%========================================================================
%
% set the bunchrange
%
  nbunch = length(beam.Bunch) ;
  if (bunch <= 0)
      firstbunch = 1 ;
      lastbunch = nbunch ;
  else
      firstbunch = bunch;
      lastbunch = bunch ;
  end
  if (bunch >=0)
    nbunch = 1 ;
  else
    nbunch = length(beam.Bunch) ;
  end
%  
% get a blank position vector and a blank sigma matrix 
%
  x = zeros(6,nbunch) ; sigma = zeros(6,6,nbunch) ; Q = 0 ;
%
% loop over bunches
%
  for bcount = firstbunch:lastbunch
      
% for multibunch calculation it's necessary to zero the charge accumulator
% on each pass thru the loop      
      
      if (nbunch > 1)
        Q = 0 ;
        bunchno = bcount ;
      else
        bunchno = 1 ;
      end
      
% limit consideration to rays which did not stop 

      Qvec = beam.Bunch(bcount).Q ;
      nstop = beam.Bunch(bcount).stop == 0 ;
      
      Q = Q + sum(Qvec) ;
      for ccount = 1:6
          
          xq = beam.Bunch(bcount).x(ccount,nstop).* Qvec(nstop) ;
          x(ccount,bunchno) = x(ccount,bunchno) + sum(xq) ;
          
          for ccount2 = ccount:6
              
              sigma(ccount,ccount2,bunchno) = ...
                sigma(ccount,ccount2,bunchno) + ...
                dot(xq,beam.Bunch(bcount).x(ccount2,nstop)) ;
            
          end
          
      end
      
% if this is the lastbunch, OR if we are doing multibunch, then do some
% stuff:      

      if ( (nbunch > 1) || (bcount==lastbunch) )
        
% normalize the vector and the matrix to the charge 

        x(:,bunchno)       = x(:,bunchno)       / Q ;
        sigma(:,:,bunchno) = sigma(:,:,bunchno) / Q ;
      
% subtract the matrix of <x_i><x_j> from sigma 

        xx = x(:,bunchno) * (x(:,bunchno))' ;
        sigma(:,:,bunchno) = sigma(:,:,bunchno) - xx ;
      
% copy upper-diagonal terms to lower-diagonal slots

        for ccount = 1:6
          for ccount2 = ccount:6
            sigma(ccount2,ccount,bunchno) = sigma(ccount,ccount2,bunchno) ;
          end
        end
      
% end of lastbunch / multibunch stuff      
      
      end
      
% end of for-loop      
      
  end
%