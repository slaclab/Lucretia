function [stat X Xerr etaFitPoint etaFitPoint_err] = bpmfit(bpmind,xdata,ydata,fitpoint,xerr,yerr,dispdata,dispdata_err)
% BPMFIT
% [stat X Xerr] = bpmfit(bpmind,xdata,ydata,fitpoint,xerr,yerr,dispdata_fitpoint,dispdata_bpms)
%   Fit initial [x,x',y,y',dE/E] based on bpm readings and dispersion data
% bpmind, BEAMLINE indicies of xdata and/or ydata given
% xdata, bpm horizontal readings (leave blank if just want y fit)
% ydata, bpm vertical readings (leave blank if just want x fit)
% fitpoint, if provided X is for this BEAMLINE element, else the first bpm
%           location is used instead.
%           Must be upstream of first bpm location
% xerr / yerr, if provided and non-empty, these should be same length as
%           bpmind and data arrays and provide error estimates on bpm
%           readings
% dispdata = [length(bpmind),2] , eta_x, eta_y, for each BPM,
%                                 omit or set empty if no dispersion data
%            If no dispdata given, if FL.SimModel.Twiss exists, use the
%            Model dispersion data from here, if not then zero dispersion
%            assumed at all the bpm locations
% dispdata_err = error on above if available
%
global BEAMLINE FL
persistent quadps
stat{1}=1;

% fill in missing parameters
if ~exist('xdata','var') || isempty(xdata)
  xdata=zeros(size(ydata));
end
if ~exist('ydata','var') || isempty(ydata)
  ydata=zeros(size(xdata));
end
if ~exist('fitpoint','var')
  fitpoint=bpmind(1);
elseif fitpoint>bpmind(1)
  error('Must provide fitpoint upstream of first bpm location')
end
if ~exist('xerr','var') || isempty(xerr)
  xerr=zeros(size(xdata));
end
if ~exist('yerr','var') || isempty(yerr)
  yerr=zeros(size(ydata));
end
% If no dispersion data provided, use design dispersion if available, else
% assume design dispersion everywhere
if ~exist('dispdata','var') || isempty(dispdata)
  if isfield(FL,'SimModel')
    dispdata(1,1)=FL.SimModel.Twiss.etax(fitpoint);
    dispdata(1,2)=FL.SimModel.Twiss.etay(fitpoint);
    for iele=1:length(bpmind)
      dispdata(1+iele,1)=FL.SimModel.Twiss.etax(bpmind(iele));
      dispdata(1+iele,2)=FL.SimModel.Twiss.etay(bpmind(iele));
    end
  else
    dispdata=zeros(length(bpmind)+1,2);
  end
end
if ~exist('dispdata_err','var')
  dispdata_err=1e-30.*ones(size(dispdata));
end

% Get the matrix which transforms dispersions into dPos/dDelta slopes
EtaMatrix=zeros(2*length(bpmind),4);
for ibpm=1:length(bpmind)
  if (bpmind(ibpm)>=fitpoint)
    [stat,R]=RmatAtoB(fitpoint,bpmind(ibpm));
  else
    [stat,R]=RmatAtoB(bpmind(ibpm),fitpoint);
    R=inv(R);
  end
  EtaMatrix(ibpm,1:4)=R(1,1:4);
  EtaMatrix(ibpm+length(bpmind),1:4)=R(3,1:4);
end

% Get dispersion at fitpoint
[etaFitPoint, etaFitPoint_err, mse] = lscov(EtaMatrix , [dispdata(:,1) ; dispdata(:,2)],...
    1./[dispdata_err(:,1).^2 ; dispdata_err(:,2).^2]) ;
etaFitPoint_err=etaFitPoint_err*sqrt(1/mse);

% Get Quad indicies
if isempty(quadps)
  quadind=findcells(BEAMLINE,'Class','QUAD');
  isps=arrayfun(@(x) isfield(BEAMLINE{x},'PS'),quadind);
  quadps=arrayfun(@(x) BEAMLINE{x}.PS,quadind(isps));
end

% Form response matrix
Rx=zeros(length(bpmind),5);
Ry=zeros(length(bpmind),5);
for ibpm=1:length(bpmind)
  [stat,R]=RmatAtoB(fitpoint,bpmind(ibpm)); if stat{1}~=1; return; end;
  Rx(ibpm,:)=[R(1,1) R(1,2) R(1,3) R(1,4) dispdata(1+ibpm,1)];
  Ry(ibpm,:)=[R(3,1) R(3,2) R(3,3) R(3,4) dispdata(1+ibpm,2)];
end

% Force data arrays to be column-vectors
d=size(xdata); if d(2)>d(1); xdata=xdata'; end;
d=size(ydata); if d(2)>d(1); ydata=ydata'; end;
d=size(xerr); if d(2)>d(1); xerr=xerr'; end;
d=size(yerr); if d(2)>d(1); yerr=yerr'; end;

% Do fitting and subtract fitted dispersion contribution from beam position
[X,Xerr,mse]=lscov([Rx;Ry],[xdata;ydata],1./[xerr;yerr].^2);
% X(1:4)=X(1:4)-X(5)*etaFitPoint;
Xerr=Xerr*sqrt(1/mse);
% Xerr(1:4)=sqrt(Xerr(1:4).^2 + Xerr(5).^2.*etaFitPoint.^2 + X(5)^2.*etaFitPoint_err.^2);



