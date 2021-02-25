function [Twiss,h0,h1]=TwissPlot(istart,iend,Initial,functions,dL)
%
% TWISSPLOT Generate a standard Twiss plot with a magnet display overhead
%
% TwissPlot(istart,iend,Initial,functions [,dL]) produces a single- or
%   dual-y-axis plot of the betatron and dispersion functions, and appends
%   a magnet display over the plot.  Scalar arguments start and end
%   determine the boundaries of the plot in the BEAMLINE cell array.
%   Initial is a Lucretia initial structure corresponding to element istart.
%   Argument select is a 3-vector which indicates which functions are to be
%   plotted:
%   if select(1) > 0, betatron functions are plotted;
%   if select(1) < 0, sqrt beta functions are plotted;
%   if select(2) ~= 0, horizontal dispersion is plotted;
%   if select(3) ~= 0, vertical dispersion is plotted.
%
%   if dL supplied, this is the max size of length element used in plotting.
%   Supply a smaller number for finer-grained plots, BEAMLINE elements
%   larger than this are split
%
% See also:  GetTwiss, AddMagnetPlot.
%
global BEAMLINE PS

% Split up BEAMLINE as required
if ~exist('dL','var')
  dL=[];
end
istart1=istart; iend1=iend;
if ~isempty(dL)
  BL1=BEAMLINE; % Store Original BEAMLINE to restore later
  PS1=PS;
  for ips=1:length(PS)
    if PS(ips).Element(1)<iend
      RenormalizePS(ips);
      for iele=PS(ips).Element
        if PS(ips).Ampl==0
          BEAMLINE{iele}.B=BEAMLINE{iele}.B*0;
        end
        BEAMLINE{iele}.PS=0;
      end
    end
  end
  BEAMLINE=BEAMLINE(istart:iend);
  BL_new={};
  for iele=1:length(BEAMLINE)
    if isfield(BEAMLINE{iele},'Girder')
      BEAMLINE{iele}.Girder=0;
    end
    if isfield(BEAMLINE{iele},'Block')
      BEAMLINE{iele}=rmfield(BEAMLINE{iele},'Block');
    end
    if isfield(BEAMLINE{iele},'Slices')
      BEAMLINE{iele}=rmfield(BEAMLINE{iele},'Slices');
    end
    if isfield(BEAMLINE{iele},'L') && BEAMLINE{iele}.L>dL && ~strcmp(BEAMLINE{iele}.Class,'TMAP')
      nsplit=ceil(BEAMLINE{iele}.L/dL);
      BL=BEAMLINE{iele};
      BL.L=BL.L/nsplit;
      if isfield(BL,'B')
        BL.B=BL.B./nsplit;
      end
      if isfield(BL,'Volt')
        BL.Volt=BL.Volt./nsplit;
        BL.Egain=BL.Egain./nsplit;
      end
      if strcmp(BL.Class,'SBEN')
        BL.Angle=BL.Angle./nsplit;
        if length(BL.EdgeAngle)==1
          BL.EdgeAngle=ones(1,2).*BL.EdgeAngle;
        end
        if length(BL.HGAP)==1
          BL.HGAP=ones(1,2).*BL.HGAP;
        end
        if length(BL.FINT)==1
          BL.FINT=ones(1,2).*BL.FINT;
        end
        if length(BL.EdgeCurvature)==1
          BL.EdgeCurvature=ones(1,2).*BL.EdgeCurvature;
        end
      end
      for isplit=1:nsplit
        if isfield(BL,'EdgeAngle') && isplit==1
          BL.EdgeAngle=[BEAMLINE{iele}.EdgeAngle(1) 0];
          BL.HGAP=[BEAMLINE{iele}.HGAP(1) 0];
          BL.FINT=[BEAMLINE{iele}.FINT(1) 0];
          BL.EdgeCurvature=[BEAMLINE{iele}.EdgeCurvature(1) 0];
        elseif isfield(BL,'EdgeAngle') && isplit==nsplit
          BL.EdgeAngle=[0 BEAMLINE{iele}.EdgeAngle(2)];
          BL.HGAP=[0 BEAMLINE{iele}.HGAP(2)];
          BL.FINT=[0 BEAMLINE{iele}.FINT(2)];
          BL.EdgeCurvature=[0 BEAMLINE{iele}.EdgeCurvature(2)];
        elseif isfield(BL,'EdgeAngle')
          BL.EdgeAngle=[0 0];
          BL.HGAP=[0 0];
          BL.FINT=[0 0];
          BL.EdgeCurvature=[0 0];
        end
        BL_new{end+1}=BL; %#ok<AGROW>
      end
    else
      BL_new{end+1}=BEAMLINE{iele}; %#ok<AGROW>
    end
  end
  BEAMLINE=BL_new(:);
  SetSPositions(1,length(BEAMLINE),BL1{istart}.S);
  istart=1; iend=length(BEAMLINE);
end
% Calc Twiss parameters
[stat,Twiss]=GetTwiss(istart,iend,Initial.x.Twiss,Initial.y.Twiss);
if stat{1}~=1
  error(stat{2});
end
% Do plots
figure
S = Twiss.S ;
ltxt={};
if abs(functions(1))
  if any(functions(2:3))
    yyaxis left
  end
  if functions(1)<0
    b1 = sqrt(Twiss.betax) ;
    b2 = sqrt(Twiss.betay) ;
  else
    b1 = Twiss.betax ;
    b2 = Twiss.betay ;
  end
  plot(S,b1,S,b2);
  if functions(1)<0
    ylabel('\surd\beta_{x,y} [m^{1/2}]');
    ltxt{end+1}='\surd\beta_x';
    ltxt{end+1}='\surd\beta_y';
  else
    ylabel('\beta_{x,y} [m]');
    ltxt{end+1}='\beta_x';
    ltxt{end+1}='\beta_y';
  end
end
if abs(functions(2))
  if abs(functions(1))
    yyaxis right
  end
  plot(S,Twiss.etax)
  ltxt{end+1}='\eta_x';
  ylabel('Dispersion [m]');
  hold on
end
if abs(functions(3))
  if abs(functions(1))
    yyaxis right
  end
  plot(S,Twiss.etay)
  ltxt{end+1}='\eta_y';
  ylabel('Dispersion [m]');
end
legend(ltxt)
hold off
ax=axis; ax(1:2)=[S(1) S(end)]; axis(ax);
xlabel('S [m]');
% Put original beamline back
if ~isempty(dL)
  BEAMLINE=BL1;
  PS=PS1;
end
% Add Magnet plot
[h0,h1]=AddMagnetPlot(istart1,iend1); %#ok<ASGLU>
