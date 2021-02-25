function Tol=MagnetTolerances(Initial,demit,r0,i1,i2,verbose)
% MAGNETTOLERANCES - Calculate offset, roll, field error tolerances
%
% MagnetTolerances(Initial,demit,r0,i1, i2 [,verbose])
%
% Inputs:
% -------
% Initial: Lucretia Initial data structure corresponding to BEAMLINE i1
% element
%
% demit: Calculate tolerances which give an estimated emittance growth of
% demit/emit_x,y
%
% r0: Provide field error tolerances for a measurement radius of r0 (m)
%
% i1, i2: First and last BEAMLINE element to calculate tolerances over
%
% if verbose>0 then print out all tolerance data to screen (default: off)
%
% Outputs:
% --------
% Tol: Stucture containing tolerances. Structure primary field names are
% magnet names, secondary field names are:
%  .b1 = b1/b0 (quad field tolerances divided by dipole field at r0)
%  .b2 = b2/b0 or b2/b1 (sextupole field tolerances for bends or quads
%  evaluated at r0)
%  .xoff = horizontal offset tolerance for Quads and Sextupoles
%  .yoff = vertical offset tolerance for Quads and Sextupoles
%  .roll = quad roll tolerance
%
%  .ind = BEAMLINE index
%
% -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
% Calculate tolerances for all BEAMLINE componenents [i1:i2] of Class:
%  SBEN, QUAD, SEXT
% Calculations based on analytical equations as per "Emittance Dilution in
% Linac-Based Free-Electron Lasers" by P. Emma and M. Venturini
%
% Assume no significant change in energy spread in given
% region [i1:i2]
%
% NB (1): SLICED ELEMENTS SHOULD HAVE SLICES FIELDS SET TO TREAT SPLIT MAGNETS
% AS WHOLE
%
% NB (2): USER CAN SET .BMAX BEAMLINE FIELD: IF FOUND USE THIS FOR
% TOLERANCE CALCS INSTEAD OF .B FIELD (FOR QUADS & SEXTS)
%
% ==============================================
% Initial version: G. White, 05/01/2017
%
global BEAMLINE PS

% parse inputs
if nargin<3
  error('Incorrect input format')
end
if ~exist('verbose','var')
  verbose=0;
end

% Get Twiss parameters over region of interest
[stat,Twiss]=GetTwiss(i1,i2,Initial.x.Twiss,Initial.y.Twiss);
if stat{1}~=1
  error('Error evaluation Twiss parameters: %s',stat{2})
end

% global constants
brho = (1e9/2.99792458e8) * Initial.Momentum ;

% - Bend b1/b0 & b2/b0
for iele=findcells(BEAMLINE,'Class','SBEN',i1,i2)
  if ~isfield(BEAMLINE{iele},'Slices'); BEAMLINE{iele}.Slices=iele; end
  if iele~=BEAMLINE{iele}.Slices(1)
    continue
  end
  emitx=Initial.x.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  emity=Initial.y.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  deltaE=Initial.SigPUncorrel/BEAMLINE{iele}.P;
  theta=sum(arrayfun(@(x) abs(BEAMLINE{x}.Angle(1)),BEAMLINE{iele}.Slices));
  betax=Twiss.betax(iele-i1+2);
  betay=Twiss.betay(iele-i1+2);
  ksix2=(Twiss.etax(iele-i1+2)^2*deltaE^2)/(betax*emitx);
  ksiy2=(Twiss.etay(iele-i1+2)^2*deltaE^2)/(betay*emity);
  b1x=(1/theta)*(r0/betax)*sqrt((2*demit)/(1+ksix2));
  b1y=(1/theta)*(r0/betay)*sqrt((2*demit)/(1+ksiy2));
  b1=min([b1x b1y]);
  b2x=(1/theta)*(r0^2/(betax*emitx)^(1/3))*sqrt(demit/(1+ksix2));
  b2y=(1/theta)*(r0^2/(betay*emity)^(1/3))*sqrt(demit/(1+ksiy2));
  b2=min([b2x b2y]);
  Tol.(BEAMLINE{iele}.Name).b1=b1;
  Tol.(BEAMLINE{iele}.Name).b2=b2;
  Tol.(BEAMLINE{iele}.Name).ind=iele;
  if verbose
    fprintf('%s: b1/b0 < %g b2/b0 < %g\n',BEAMLINE{iele}.Name,b1,b2);
  end
end

% - Quad b2/b1
for iele=findcells(BEAMLINE,'Class','QUAD',i1,i2)
  if ~isfield(BEAMLINE{iele},'Slices'); BEAMLINE{iele}.Slices=iele; end
  if iele~=BEAMLINE{iele}.Slices(1)
    continue
  end
  emitx=Initial.x.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  emity=Initial.y.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  deltaE=Initial.SigPUncorrel/BEAMLINE{iele}.P;
  betax=Twiss.betax(iele-i1+2);
  betay=Twiss.betay(iele-i1+2);
  ksix2=(Twiss.etax(iele-i1+2)^2*deltaE^2)/(betax*emitx);
  ksiy2=(Twiss.etay(iele-i1+2)^2*deltaE^2)/(betay*emity);
  l=sum(arrayfun(@(x) BEAMLINE{x}.L,BEAMLINE{iele}.Slices));
  if ksix2>ksiy2
    ksi2=(Twiss.etax(iele-i1+2)^2*deltaE^2)/(betax*emitx);
    chi2 = (betay * emity) / (betax * emitx) ;
    k2_ex = sqrt( demit/(0.25*l^2*betax^3*emitx*((1+ksi2)^2+chi2^2)) ) ;
    k2_ey = sqrt( demit/(0.5*l^2*betax*betay^2*emitx*(1+ksi2)) ) ;
  else
    ksi2=(Twiss.etay(iele-i1+2)^2*deltaE^2)/(betay*emity);
    chi2 = (betax * emitx) / (betay * emity) ;
    k2_ey = sqrt( demit/(0.25*l^2*betay^3*emity*((1+ksi2)^2+chi2^2)) ) ;
    k2_ex = sqrt( demit/(0.5*l^2*betay*betax^2*emity*(1+ksi2)) ) ;
  end
  k2=min([k2_ex k2_ey]);
  b2=0.5*k2*brho*r0^2;
  b1=0;
  for isl=BEAMLINE{iele}.Slices
    if isfield(BEAMLINE{isl},'BMAX')
      b1=b1+BEAMLINE{isl}.BMAX*r0;
    else
      b1=b1+BEAMLINE{isl}.B*r0;
      if isfield(BEAMLINE{iele},'PS') && BEAMLINE{iele}.PS>0
        b1=b1*PS(BEAMLINE{iele}.PS).Ampl;
      end
    end
  end
  Tol.(BEAMLINE{iele}.Name).b2=b2/b1;
  Tol.(BEAMLINE{iele}.Name).ind=iele;
  if verbose
    fprintf('%s: b2/b1 < %g\n',BEAMLINE{iele}.Name,b2/b1);
  end
end

% - Quad offset + roll
for iele=findcells(BEAMLINE,'Class','QUAD',i1,i2)
  if ~isfield(BEAMLINE{iele},'Slices'); BEAMLINE{iele}.Slices=iele; end
  if iele~=BEAMLINE{iele}.Slices(1)
    continue
  end
  BL1=BEAMLINE; PS1=PS;
  for isl=BEAMLINE{iele}.Slices
    if isfield(BEAMLINE{isl},'BMAX')
      BEAMLINE{isl}.B=BEAMLINE{isl}.BMAX;
      if isfield(BEAMLINE{isl},'PS') && BEAMLINE{isl}.PS>0
        PS(BEAMLINE{isl}.PS).Ampl=1;
      end
    end
  end
  emitx=Initial.x.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  emity=Initial.y.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  deltaE=Initial.SigPUncorrel/BEAMLINE{iele}.P;
  [~,R]=RmatAtoB(iele,BEAMLINE{iele}.Slices(end));
  BEAMLINE=BL1; PS=PS1;
  fx=abs(1/R(2,1));
  fy=abs(1/R(4,3));
  betax=Twiss.betax(iele-i1+2);
  betay=Twiss.betay(iele-i1+2);
  ksix2=(Twiss.etax(iele-i1+2)^2*deltaE^2)/(betax*emitx);
  ksiy2=(Twiss.etay(iele-i1+2)^2*deltaE^2)/(betay*emity);
  offx=(fx/deltaE)*sqrt(((2*emitx)/betax)*demit);
  offy=(fy/deltaE)*sqrt(((2*emity)/betay)*demit);
  roll=asin( fx * sqrt( (2*emitx/emity) * (1/(betax*betay)) * demit * (1+ksix2) ) ) * 0.5 ;
  Tol.(BEAMLINE{iele}.Name).xoff=offx;
  Tol.(BEAMLINE{iele}.Name).yoff=offy;
  Tol.(BEAMLINE{iele}.Name).roll=roll;
  Tol.(BEAMLINE{iele}.Name).ind=iele;
  if verbose
    fprintf('%s: xoff < %g yoff < %g roll < %g\n',BEAMLINE{iele}.Name,offx,offy,roll)
  end
end

% - Sextupole offset
for iele=findcells(BEAMLINE,'Class','SEXT',i1,i2)
  if ~isfield(BEAMLINE{iele},'Slices'); BEAMLINE{iele}.Slices=iele; end
  if iele~=BEAMLINE{iele}.Slices(1)
    continue
  end
  betax=Twiss.betax(iele-i1+2);
  betay=Twiss.betay(iele-i1+2);
  k2=0;
  for isl=BEAMLINE{iele}.Slices
    if isfield(BEAMLINE{isl},'BMAX')
      k2=k2+BEAMLINE{isl}.BMAX/brho;
    else
      k2=k2+BEAMLINE{isl}.B/brho;
    end
  end
  if isfield(BEAMLINE{iele},'PS') && BEAMLINE{iele}.PS>0 && ~isfield(BEAMLINE{isl},'BMAX')
    k2=k2*PS(BEAMLINE{iele}.PS).Ampl;
  end
  l=sum(arrayfun(@(x) BEAMLINE{x}.L,BEAMLINE{iele}.Slices));
  emitx=Initial.x.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  emity=Initial.y.NEmit/(BEAMLINE{iele}.P/0.511e-3);
  deltaE=Initial.SigPUncorrel/BEAMLINE{iele}.P;
  if ksix2>=ksiy2
    ksi2=(Twiss.etax(iele-i1+2)^2*deltaE^2)/(betax*emitx);
    chi2 = (betay * emity) / (betax * emitx) ;
    xoff_ex=sqrt(demit/(0.5 * k2^2 * l^2 * betax^2 * (1+ksi2)));
    yoff_ex=sqrt(demit/(0.5 * k2^2 * l^2 * betax^2 * chi2));
    xoff_ey=sqrt(demit/(0.5 * k2^2 * l^2 * betax*betay * (emitx/emity) * chi2));
    yoff_ey=sqrt(demit/(0.5 * k2^2 * l^2 * betax*betay * (emitx/emity) * (1+ksi2)));
  else
    ksi2=(Twiss.etay(iele-i1+2)^2*deltaE^2)/(betay*emity);
    chi2 = (betax * emitx) / (betay * emity) ;
    xoff_ey=sqrt(demit/(0.5 * k2^2 * l^2 * betay^2 * (1+ksi2)));
    yoff_ey=sqrt(demit/(0.5 * k2^2 * l^2 * betay^2 * chi2));
    xoff_ex=sqrt(demit/(0.5 * k2^2 * l^2 * betax*betay * (emity/emitx) * chi2));
    yoff_ex=sqrt(demit/(0.5 * k2^2 * l^2 * betax*betay * (emity/emitx) * (1+ksi2)));
  end
  xoff=min([xoff_ex xoff_ey]); yoff=min([yoff_ex yoff_ey]);
  Tol.(BEAMLINE{iele}.Name).xoff=xoff;
  Tol.(BEAMLINE{iele}.Name).yoff=yoff;
  Tol.(BEAMLINE{iele}.Name).ind=iele;
  if verbose
    fprintf('%s: xoff < %g yoff < %g\n',BEAMLINE{iele}.Name,xoff,yoff);
  end
end
