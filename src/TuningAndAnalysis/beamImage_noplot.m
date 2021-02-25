function data=beamImage_noplot(beam,nsig,E0,asym,nbins,dpk)
% data=beamImage_noplot(beam,nsig [,E0,asym,nbins,dpk])
%
% Data that is calculated in beamImage function but without actually
% plotting the results
% nsig = cut on transverse dimension
% E0 (optional) = centroid beam energy
% asym (optional) = true (default) means asymmetric fit
% nbins (optional) = number of histogram bins to use (default 150)
% dpk (optional) = perform double-peak fit for [trans long]
% ===========================================

data=[];

% Turn off frequent annoying warnings
w1=warning('query','MATLAB:rankDeficientMatrix');
w2=warning('query','MATLAB:nearlySingularMatrix');
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix

if ~exist('dpk','var') || isempty(dpk); dpk=[false false]; end;

id=find(beam.Bunch.stop==0);
rays=beam.Bunch.x(:,id)'; %#ok<*FNDSB>
Q=beam.Bunch.Q(id);

conv=[1e6,1e6,1e6,1e6,1e6,1e2];
if ~exist('nbins','var') || isempty(nbins); nbins=150; end;
if ~exist('asym','var') || isempty(asym); asym=false; end;

x=conv(1)*rays(:,1);
px=conv(2)*rays(:,2); %#ok<NASGU>
y=conv(3)*rays(:,3);
py=conv(4)*rays(:,4); %#ok<NASGU>
z=conv(5)*rays(:,5);
E=rays(:,6);
if (~exist('E0','var')) || isempty(E0); E0=mean(E);end
dp=conv(6)*(E-E0)/E0;

if (nsig>0)
  v={'x','px','y','py','z','dp'};
  for n=1:length(v)
    eval(['u=',v{n},';'])
    N=length(u);
    nit=0;
    Qtmp=Q;
    while 1
      u0=mean(u);
      sig=sqrt(var(u,Qtmp));
      id=find(abs(u-u0)<nsig*sig);
      u=u(id);
      Qtmp=Q(id);
      if (length(u)==N),break,end
      N=length(u);
      nit=nit+1;
    end
    eval(sprintf('id%s=find(ismember(%s,u));',v{n},v{n}));
  end
else
  idx=(1:length(x))';
  idy=(1:length(x))';
  idz=(1:length(x))';
  iddp=(1:length(x))';
  idpx=(1:length(x))'; %#ok<NASGU>
  idpy=(1:length(x))'; %#ok<NASGU>
end


u=linspace(min(y(idy)),max(y(idy)),nbins);
[~,BIN]=histc(y(idy),u); v=accumarray(BIN,Q(idy)).*1e9;
if (asym)
  if dpk(1)
    [~,q]=agauss_fit2(u,v,[],0);
  else
    [~,q]=agauss_fit(u,v,[],0);
  end
else
  if dpk(1)
    [~,q]=gauss_fit(u,v,[],0);
  else
    [~,q]=gauss_fit(u,v,[],0);
  end
end
data.sigy=q(4);
u=linspace(min(x(idx)),max(x(idx)),nbins);
[~,BIN]=histc(x(idx),u); v=accumarray(BIN,Q(idx)).*1e9;
if (asym)
  if dpk(1)
    [~,q]=agauss_fit2(u,v,[],0);
  else
    [~,q]=agauss_fit(u,v,[],0);
  end
else
  if dpk(1)
    [~,q]=gauss_fit2(u,v,[],0);
  else
    [~,q]=gauss_fit(u,v,[],0);
  end
end
data.sigx=q(4);
data.rmsx=std(x(idx));
data.rmsy=std(y(idy));

u=linspace(min(dp(iddp)),max(dp(iddp)),nbins);
[~,pbins]=histc(dp(iddp),u); v=accumarray(pbins,Q(iddp)).*1e9;
if (asym)
  [~,q]=agauss_fit(u,v,[],0);
else
  if dpk(2)
    q=peakfit([u' v'],0,0,2,1,0,10,0,0,0,0);
  else
    [~,q]=gauss_fit(u,v,[],0);
  end
end
data.sigE=q(4);
u=linspace(min(z(idz)),max(z(idz)),nbins);
[~,zbins]=histc(z(idz),u); v=accumarray(zbins,Q(idz));
v=1e-3.*v.*(1/(((u(2)-u(1))*1e-6)/299792458)); % y-axis Q->I (kA)
if (asym)
  [~,q]=agauss_fit(u,v,[],0);
else
  if dpk(2)
    q=peakfit([u' v'],0,0,2,1,0,10,0,0,0,0);
  else
    [~,q]=gauss_fit(u,v,[],0);
  end
end
data.sigz=q(4);
data.pkI=max(v);
data.rmsz=std(z(idz));
data.rmsE=std(dp(iddp));

warning(w1.state,'MATLAB:rankDeficientMatrix');
warning(w2.state,'MATLAB:nearlySingularMatrix');
