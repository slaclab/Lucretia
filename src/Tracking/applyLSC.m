function [bunchOut, dl] = applyLSC(varargin)
% APPLYLSC - apply longitudinal space charge effects to bunch
% Calculate free-space LSC impedance and apply resulting voltage
% change to particles in provided Bunch
%
% [bunchOut, dL] = applyLSC(bunchIn,Q,stop,elemno,dL [,storeBunchInd])
% Apply LSC to provided Lucretia bunch (x) "bunchIn", "Q", "stop" vector associated with BEAMLINE{elemno}
%   dL (output) = recommended minimum drift length for this bunch (0.1c/w_p)
%   storeBunchInd (optional) = keep calculation data tagged with provided bunch ID
%   other internal parameters obtained from BEAMLINE{elemno}.TrackFlag.LSC_*
%   parameters
%
% applyLSC('plot',dataLoc [,figureHandle])
% Analysis plots from stored data in previous function call(s).
%   dataLoc=[elemno bunchno]
%   figureHandle (Optional) = plot to give figure handle, else make a new one
%
% data = applyLSC('getdata');
%   Return any stored data:
%   data.tbins : time bins
%   data.I : current histogram enries / kA
%   data.I_ns : as above without smoothing applied
%   data.Y : fft of current histogram
%   data.f : frequency data points
%   data.V : computed energy modulation
%   data.Z : LSC impedance
%   data.L : drift length over which LSC applied
global BEAMLINE
persistent datastore
c=299792458;

% Plot previously stored data
if isequal(varargin{1},'plot')
  ind=varargin{2}(1);
  bind=varargin{2}(2);
  if nargin>2
    figure(varargin{3})
    cla
  else
    figure
  end
  set(gcf,'Name',sprintf('LSC Data for element # %d',ind))
  tbins=datastore(ind,bind).tbins;
  if isempty(tbins); error('No stored data to plot'); end
  I=datastore(ind,bind).I.*1e-3;
  I_ns=datastore(ind,bind).I_ns.*1e-3;
%   f=datastore(ind,bind).f;
  V=datastore(ind,bind).V;
%   Z=datastore(ind,bind).Z;
  L=datastore(ind,bind).L;
  subplot(2,1,1), plot(tbins.*1e12,I_ns,'r'); 
  hold on
  plot(tbins.*1e12,I); hold off; grid
  legend({'Unsmoothed Current histogram' 'Smoothed data'});
  xlabel('t / ps'); ylabel('I / kA')
  subplot(2,1,3), plot(abs(diff(tbins(1:2))+tbins(1:end-1).*1e12,(real(V)/L).*1e-3)), grid on
  xlabel('\t / ps')
  ylabel('V / keV.m^{-1}')
  return
end
if isequal(varargin{1},'getdata')
  bunchOut=datastore;
  return
end

% If this far then first argument needs to be a Lucretia Bunch, second a
% BEAMLINE element #, third the tracking length
bunchIn=varargin{1};
beamQ=varargin{2};
stop=varargin{3};
elemno=varargin{4};
L=varargin{5};
if elemno<1 || elemno>length(BEAMLINE) || ~isfield(BEAMLINE{elemno},'TrackFlag')
  error('Invalid BEAMLINE element index');
end

% get data from provided bunch
z=bunchIn(5,~stop);
if isempty(z)
  error('LSC: all particles stopped in element %d',elemno)
end
x=bunchIn(1,~stop);
y=bunchIn(3,~stop);
E=bunchIn(6,~stop);
beamQ(logical(stop))=[];
sx=std(x); sy=std(y);
rb=1.7*(sx+sy)/2;
gamma=mean(E)./0.511e-3;

% Parse TrackFlag arguments
tf=BEAMLINE{elemno}.TrackFlag;
cutoff=[0 0.9];
if isfield(tf,'LSC_cutoffFreq') && length(tf.LSC_cutoffFreq)==2
  cutoff(2)=tf.LSC_cutoffFreq;
end
npowbins=12;
if isfield(tf,'LSC_npow2Bins')
  npowbins=tf.LSC_npow2Bins;
end
nbins=2^npowbins; % number of bins for current histogram
smoothFactor=0;
if isfield(tf,'LSC_smoothFactor')
  smoothFactor=tf.LSC_smoothFactor;
end
storeDataBunchNo=0;
if nargin>5
  storeDataBunchNo=varargin{6};
end

% Form current histogram
zmin=min(z);
zmax=max(z);
if zmin==zmax
  error('Need some spread in z-distribution of bunch to compute LSC')
end
[~,zbins,bininds]=histcounts(z,nbins);
q = accumarray(bininds',beamQ')';
tbins=zbins./c;
dt=abs(tbins(2)-tbins(1));
I=q./dt;

% min step length
dl= (0.1*c) / ( ((2*c)/rb)*sqrt(max(I)/(17e3*gamma^3)) ); 

% smooth I distribution
I_ns=I;
if smoothFactor>1
  I=smoothn(I,smoothFactor);
elseif smoothFactor==1
  I=smoothn(I,'robust');
end

% Calculate fourier transform of current histogram
nfft=2^nextpow2(nbins);
Y=fft(I,nfft);
Fs=1/dt;
f=Fs/2*linspace(0,1,nfft/2+1);
Nq=f(nfft/2+1); % Nyquist freq.

% Calculate LSC impedance
Z0=377; % Ohm (free space impedance)
f2=Fs*linspace(0,1,nfft);
k=(2.*pi.*f2)./c;
% - apply frequency cuts
k_cutoff=([cutoff(1)*Nq cutoff(2)*Nq].*2.*pi)./c; 
Z=((1i*Z0)./(pi.*k.*rb^2)) .* (1-((k.*rb)./gamma).*besselk(1,(k.*rb)./gamma)); Zs=Z;
Z(k<k_cutoff(1))=0; Z(k>k_cutoff(2))=0;
Z(1)=0; Y(1)=0; % remove DC component

% Calculate energy modulation and apply to bunch
V=real(ifft(Z.*Y,nbins)).*L;
bunchOut=bunchIn;
bunchOut(6,~stop)=bunchOut(6,~stop)+V(bininds).*1e-9 ;

% Keep data if requested
if storeDataBunchNo>0
  datastore(elemno,storeDataBunchNo).tbins=tbins;
  datastore(elemno,storeDataBunchNo).I=I;
  datastore(elemno,storeDataBunchNo).I_ns=I_ns;
  datastore(elemno,storeDataBunchNo).nbins=nbins;
  datastore(elemno,storeDataBunchNo).Y=Y;
  datastore(elemno,storeDataBunchNo).f=f;
  datastore(elemno,storeDataBunchNo).V=V;
  datastore(elemno,storeDataBunchNo).L=L;
  datastore(elemno,storeDataBunchNo).Z=Zs;
  datastore(elemno,storeDataBunchNo).foffs=[cutoff(1)*Nq cutoff(2)*Nq];
end
