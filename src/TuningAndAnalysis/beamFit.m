function [sigFit,sigRms]=beamFit(beam,nsig,E0)
% [sigFit,sigRms]=beamFit(beam,nsig [,E0])
%
% Asymmetric gaussian fits of provided Lucretia beam with optional cuts
% beam = Lucretia beam structure
% nsig = cut on transverse dimension
% E0 (optional) = centroid beam energy
%
% distributions: [x xp y yp z dE/E]
% units: um/urad/%
% sigFit (asymmetric gaussian widths)
% sigRms (RMS of distributions after cuts)
% ===========================================
% GW, March 14, 2013

rays=beam.Bunch.x(:,beam.Bunch.stop==0);
for idim=1:6; R{idim}=rays(idim,:); end;
conv=[1e6,1e6,1e6,1e6,1e6,1e2];
nbin=100;
if (~exist('E0','var')),E0=mean(R{6});end
R{6}=(R{6}-E0)./E0;
if (nsig>0)
  for n=1:6
    N=length(R{n});
    while 1
      R{n}=R{n}(abs(R{n}-mean(R{n}))<nsig*std(R{n}));
      if (length(R{n})==N),break,end
      N=length(R{n});
    end
  end
end
w1=warning('query','MATLAB:rankDeficientMatrix');
w2=warning('query','MATLAB:nearlySingularMatrix');
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
for n=1:6
  [v,u]=hist(R{n},nbin);
  [~,q]=agauss_fit(u,v,[],0);
  sigFit(n)=q(4)*conv(n);
  sigRms(n)=std(R{n})*conv(n);
end
warning(w1.state,'MATLAB:rankDeficientMatrix');
warning(w2.state,'MATLAB:nearlySingularMatrix');