function beamout = MakeBeamHalo( Initial, nrays, nsigma1, nsigma2, allsame )
% beamout = MakeBeamHalo( Initial, nrays, nsigma1, nsigma2, allsame )
%    returns a beam in which the bunch charges have been populated with rays which are
%    weighted according to a "1/r" distribution in transverse phase space
%    Momentum & z charges are distributed according to a gaussian distribution.
%    Particle co-ordinates themselves are uniformly randomly distributed.
%    Argument nrays is either a scalar (number of rays per bunch) or a
%    vector (number of rays in each bunch);
%    nsigma2 is the truncation factor for the distribution. Argument is
%    vector [1 x 4] for (x,x'), (y,y'), z & E dimensions.
%    nsigma1 is the lower cut on (x,x'),(y,y') for the 1/r distribution.
%    Argument should be corresponding [1 x 2] vector
%    If argument allsame == 0, a unique set of rays is generated for each
%    bunch, whereas if allsame == 1 the rays generated for bunch 1 are
%    reused for each subsequent bunch.  If allsame == 1, then all bunches
%    will have nrays(1) rays regardless of the value of nrays(2:end).
%
% See also:  MakeBeam6DSparse, MakeBeamPZGrid, MakeBeamPZGauss, MakeBeam6DGauss,
%            MakeBeam6DWeighted, CreateBlankBeam, InitCondStruc.
%
% Version date:  21-Nov-2014, GRW


%==========================================================================

% get a blank beam with the correct # of bunches etc

beamout = CreateBlankBeam(Initial.NBunch,1,1,Initial.BunchInterval) ;

% get useful stuff out of the arguments
nsigma1=nsigma1(:)'; nsigma2=nsigma2(:)';
if length(nsigma2)~=4 || length(nsigma1)~=2
  error('Incorrect format for ''nsigma1'' and/or ''nsigma2'' argument, see help description');
end
nrays = nrays(:) ;
if (length(nrays) == Initial.NBunch)
  nraygen = nrays ;
else
  nraygen = nrays(1) * ones(Initial.NBunch,1) ;
end
centroid = [Initial.x.pos   ; ...
  Initial.x.ang  ; ...
  Initial.y.pos   ; ...
  Initial.y.ang  ; ...
  Initial.zpos  ; Initial.Momentum    ] ;
relgam0 = sqrt(Initial.Momentum^2 + (0.000510998918)^2) / 0.000510998918 ;
emitx = Initial.x.NEmit / relgam0 ;
emity = Initial.y.NEmit / relgam0 ;

% compute uncorrelated sigmas

sig(1)  = sqrt(emitx*Initial.x.Twiss.beta) ;
sig(2)  = sqrt(emitx/Initial.x.Twiss.beta) ;
sig(3)  = sqrt(emity*Initial.y.Twiss.beta) ;
sig(4)  = sqrt(emity/Initial.y.Twiss.beta) ;
sig(5)  = Initial.sigz ;
sig(6)  = Initial.SigPUncorrel ;

% construct a correlation matrix

R = eye(6) ;
R(1,6) =   Initial.x.Twiss.eta / Initial.Momentum ;
R(2,1) = - Initial.x.Twiss.alpha / Initial.x.Twiss.beta ;
R(2,6) =   Initial.x.Twiss.etap / Initial.Momentum ;
R(3,6) =   Initial.y.Twiss.eta / Initial.Momentum ;
R(4,3) = - Initial.y.Twiss.alpha / Initial.y.Twiss.beta ;
R(4,6) =   Initial.y.Twiss.etap / Initial.Momentum ;
R(6,5) =   Initial.PZCorrel ;

% Ellipse emittances (radii) for chosen sigma levels
emitx1=emitx*nsigma1(1)^2;
emitx2=emitx*nsigma2(1)^2;
emity1=emity*nsigma1(2)^2;
emity2=emity*nsigma2(2)^2;

% loop over bunches

for bunchno = 1:Initial.NBunch
  
  % if this is not the first bunch, AND we want allsame, just glue the old
  % bunch into the new bunch slots
  
  if ( (bunchno>1) && (allsame==1) )
    
    beamout.Bunch(bunchno) = beamout.Bunch(1) ;
    
  else
    
    beamout.Bunch(bunchno).stop = zeros(1,nraygen(bunchno)) ;
    
    % construct the position vectors and charge weights
    x = zeros(6,nraygen(bunchno)) ;
    offset = x ;
    ov = ones(1,nraygen(bunchno)) ;
    pdf=ones(6,nraygen(bunchno));
    for count = 1:6
      if count>=5 % do x & E distributions
        v = nsigma2(count-2).*(2.*rand(1,nraygen(bunchno))-1) ;
        x(count,:) = sig(count) * v ;
        pdf(count,:)=(1/sqrt(2*pi)).*exp(-v.^2./2);
      elseif count==1 % do x & x' distribution
        % Choose emittance randomly emit1:emit2
        e=emitx1+rand(1,nraygen(bunchno)).*(emitx2-emitx1);
        % Get "radius" of ellipse for random theta
        th=rand(1,nraygen(bunchno)).*(pi*2);
        a=sqrt(e.*Initial.x.Twiss.beta ); b=sqrt(e./Initial.x.Twiss.beta); 
%         r=(a.*b)./sqrt((b.*cos(th)).^2+(ar.*sin(th)).^2);
        % Populate distribution and get charge weighting factor
        x(1,:) = a.*cos(th) ;
        x(2,:) = b.*sin(th) ;
      elseif count==3 % do y & y' distribution
        % Choose emittance randomly emit1:emit2
        e=emity1+rand(1,nraygen(bunchno)).*(emity2-emity1);
        % Get "radius" of ellipse for random theta
        th=rand(1,nraygen(bunchno)).*2.*pi;
        a=sqrt(e.*Initial.y.Twiss.beta ); b=sqrt(e./Initial.y.Twiss.beta);
%         r=(a.*b)./sqrt((b.*cos(th)).^2+(a.*sin(th)).^2);
        % Populate distribution and get charge weighting factor
        x(3,:) = a.*cos(th) ;
        x(4,:) = b.*sin(th) ;
      end
      offset(count,:) = centroid(count)*ov ;
    end
    r1=sqrt((x(1,:)./std(x(1,:))).^2+(x(2,:)./std(x(2,:))).^2);
    r2=sqrt((x(3,:)./std(x(3,:))).^2+(x(4,:)./std(x(4,:))).^2);
    pdf(1,:)=1./r1;
    pdf(2,:)=1./r2;
    pdf=prod(pdf);
    pdf=pdf./sum(pdf);
    beamout.Bunch(bunchno).Q = Initial.Q .* pdf ;
        
    % put in the correlation and take out necessary offsets
    % if just asking for single ray, just make this the centroid
    if length(v)>1
      x = R*x + offset ;
      beamout.Bunch(bunchno).x = x ;
    else
      beamout.Bunch(bunchno).x = centroid ;
    end
    
  end
  
end

beamout = CheckBeamMomenta(beamout) ;
%