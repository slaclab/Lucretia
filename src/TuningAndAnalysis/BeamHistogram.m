function [bincenter,binheight] = BeamHistogram(Beam,bunchno,coord,precision)
%
% BEAMHISTOGRAM Divide a set of rays into a histogram, taking into account
%    non-gaussian distribution and unequal charge weightings.
%
%   [bincenter,binheight] = BeamHistogram(Beam,bunchno,coord,precision)
%      bins the rays in Beam.Bunch(bunchno).x(coord,:), returning the
%      centers of the bins in bincenter and the height in binheight.
%      If bunchno == 0, all bunches are binned together.
%      The height of the bin is the sum of the charges of the rays in
%      the bin.  In order to reasonably distribute the bins for a non-
%      Gaussian bunch distribution, the binning algorithm first finds 
%      the median position and the effective sigma (ie, region within 
%      which is 68% of the charge, approximately).  The precision 
%      argument indicates the desired bunch spacing in fractions of this
%      effective sigma.

%==========================================================================

% select first and last bunch

  if (bunchno == 0)
      bfirst = 1 ;
      blast = length(Beam.Bunch) ;
  else
      bfirst = bunchno ;
      blast = bunchno ;
  end
  
% put all the rays into a single vector; eliminate stopped rays by 
% setting their charges to zero

  x = [] ; Q = [] ;
  for bcount = bfirst:blast
      keep = (Beam.Bunch(bcount).stop == 0) ;
      x = [x Beam.Bunch(bcount).x(coord,:)] ;
      Q = [Q Beam.Bunch(bcount).Q .* keep] ;
  end

  nray = sum(Q ~= 0) ;
  nrayall = length(Q) ;
  
% if zero or 1 rays, handle that now:

  if (nray == 0)
      bincenter = [] ;
      binheight = [] ;
      return ;
  end

  if (nray == 1)
      binheight = sum(Q) ;
      bincenter = sum(Q.*x) / sum(Q) ;
      return ;
  end

% compute half the total charge

  Q0p5 = 0.5 * sum(Q) ;
  
% generate a sortkey in the dimension of interest

  [dmy,sortkey] = sort(x) ;

% count up along sortkey and stop when 50% of charge is found

  qsum = 0 ; xcent = 0 ; icent = 0 ;
        
  for rcount = 1:nrayall
            
    qsum = qsum + Q(sortkey(rcount)) ;
    if (qsum >= Q0p5)
        xcent = x(sortkey(rcount)) ; ;
         icent = rcount ;
         break
    end
            
  end
        
% loop forward and backwards, finding boundaries about the center which
% contain 68% of the total charge

  Q0p34 = 0.34 * sum(Q) ; Qplus = 0 ; Qminus = 0 ; 
  iplus = icent ; iminus = icent ;
        
  for rcount = icent+1:nrayall
            
    Qplus = Qplus + Q(sortkey(rcount)) ;
    iplus = rcount ;
    if (Qplus >= Q0p34)
        break ;
    end
            
  end
        
  for rcount = icent-1:-1:1
            
    Qminus = Qminus + Q(sortkey(rcount)) ;
    iminus = rcount ;
    if (Qminus >= Q0p34)
        break
    end
            
  end

% compute an effective sigz by scaling the interval bounded by iminus and
% iplus by the total charge contained therein

  sigx = ( x(sortkey(iplus))    - ...
           x(sortkey(iminus)) ) * ...
           Q0p34/(Qplus+Qminus) ;
             
% The slice spacing is a fraction of sigz given by the WF.BinWidth 
% parameter

  binx = sigx * precision ;
        
% compute the maximum number of bins needed

  nbin = ceil( ( x(sortkey(nrayall)) - x(sortkey(1)) ) / binx ) ;

  nbin = nbin + 2 ;
  
  Qbin = zeros(1,nbin) ;
  xbin = zeros(1,nbin) ;
                      
% loop over rays, assign to bins.  Whilst doing so, compute the electrical
% center position of each bin and determine whether the bins are empty.

  binno = 1 ; binstart = x(sortkey(1)) - binx ;
  binstop = binstart + binx ;
  binctr = binstart + binx/2 ;
  qsum = 0 ; qxsum = 0 ;

  for rcount = 1:nrayall
            
      while (x(sortkey(rcount)) > binstop)                
          binstart = binstop ;
          binstop = binstop + binx ;
          binctr = binstart + binx / 2 ;
%          if (qsum > 0)
             Qbin(binno) = qsum ;
%             xbin(binno) = qxsum / qsum ;
             xbin(binno) = binctr-binx ;
             qsum = 0 ;
             qxsum = 0 ;
             binno = binno + 1 ;
%          end 
      end
            
      qsum = qsum + Q(sortkey(rcount)) ;
      qxsum = qxsum + Q(sortkey(rcount)) * x(sortkey(rcount)) ;
            
  end
        
% assign the charge and position of the last bin

  if (qsum > 0)
    Qbin(binno) = qsum ; %xbin(binno) = qxsum / qsum ;
    xbin(binno) = binctr ;
  else
    binno = binno - 1 ;
  end
        
% assign return variables

  bincenter = xbin(1:binno) ;
  binheight = Qbin(1:binno) ;
  
end
  
%  