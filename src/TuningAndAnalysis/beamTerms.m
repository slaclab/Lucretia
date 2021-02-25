function [fitTerm,fitCoef,bsize_corrected,bsize,p,bsize_accum] = beamTerms(dim,beam)
% [fitTerm,fitCoef,bsize_corrected,bsize] = beamTerms(dim,beam)
% Show relative contributions of up to 3rd-order beam correlations to the
% beam size in the requested dimension
% 3rd order polynomial fit to 4 independent variables vs specified (dim)
% dependent variable
% REQUIRES: PolyfitnTools directory in search path
% INPUTS:
% -------
% dim = required lucretia beam dimension to correlate
% beam = lucretia beam
%
% OUTPUTS:
% --------
% fitTerm(56,6) = Beam correlation (e.g. if dim=3 [1 0 0 0 0 0] = SIGMA(3,1)
%                                                 [1 0 0 1 0 0] = T314
%                                                 [2 1 0 0 0 0] = U3112 etc... )
% fitCoef(1,56) = coefficient related to corresponding beam correlation
% bsize_corrected(1,3) = RMS beam size in dimension dim when all 1st-3rd order
%                        correlations are subtracted
% bsize(1,56) = RMS beam size improvement due to corresponding correlation term
%               being subtracted from beam
% p = results structure from polyfitn

% Warnings not caring about here
warnstate=warning('query','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:nearlySingularMatrix');

% remove stopped particles
beam.Bunch.x(:,beam.Bunch.stop>0)=[];
if isempty(beam.Bunch.x)
  error('All particles in provided beam stopped!')
end

% remove constant offsets
for idim=1:6
  beam.Bunch.x(idim,:)=beam.Bunch.x(idim,:)-mean(beam.Bunch.x(idim,:));
end

% Perform 1st-3rd order fit
allpar=[1 2 3 4 5 6];
xfit=beam.Bunch.x(allpar(~ismember(allpar,dim)),:)';
yfit=beam.Bunch.x(dim,:)';
bsize_corrected=zeros(1,3);
for iorder=1:3
  p=polyfitn(xfit,yfit,iorder);
  bsize_corrected(iorder)=std(yfit-polyvaln(p,xfit));
end
p=polyfitn(xfit,yfit,iorder);
% Get contribution of beamsize to each fit term
bsize=zeros(length(p.Coefficients)-1,1);
fitTerm=zeros(length(p.Coefficients)-1,6);
fitCoef=zeros(1,length(p.Coefficients)-1);
allTerms=1:length(p.ModelTerms);
for iterm=1:length(p.Coefficients)-1
  p2=polyfitn(xfit,yfit,p.ModelTerms(allTerms~=iterm,:));
  bsize(iterm)=std(yfit-polyvaln(p2,xfit))-bsize_corrected(end);
  fitTerm(iterm,allpar~=dim)=p.ModelTerms(iterm,:);
  fitCoef(iterm)=p.Coefficients(iterm);
end
[VAL, I]=sort(bsize,'descend');
bsize=VAL;
fitTerm=fitTerm(I,:);
fitCoef=fitCoef(I);
if nargout>=6
  for iterm=1:length(I)
    p2=polyfitn(xfit,yfit,p.ModelTerms([I(1:iterm); end],:));
    bsize_accum(iterm)=std(yfit-polyvaln(p2,xfit));
  end
end

% Put warnings back to original state
warning(warnstate.state,'MATLAB:nearlySingularMatrix');