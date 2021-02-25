function [stat dataOut1 dataOut2] = svdCorrelate(data,corData,jitpos)
% corOutput = svdCorrelate(data,corData)
% Reform input data keeping just SVN mode that is most correlated to data
% passed in dataOut (corData must have same number of rows as data)

stat{1}=1; dataOut2=[];

% find 5 jitter modes
[U,S,Vt]=svd(data(:,1:jitpos));
V=Vt';
for imode=1:5
  jit(imode,:)=data(:,1:jitpos)*V(imode,:)';
end

% Do singular value decomposition
[U,S,Vt]=svd(data);
V=Vt';

% Find modes most correlated to 5 jitter modes
S_jitremove=S;
for ijit=1:5
  for imode=1:15
    [r,p] = corrcoef(jit(ijit,:),data*V(imode,:)');
    if ~isempty(find(p<0.1, 1))
      rcor(imode)=r(1,2);
    else
      rcor(imode)=0;
    end
  end
  [val ind]=max(rcor);
  if val~=0
    S_jitremove(ind,ind)=0;
  end
end
dataOut1=U*S_jitremove*V;

% Correlate modes with corData
for imode=1:15
  [r,p] = corrcoef(abs(corData),abs(data*V(imode,:)'));
  if ~isempty(find(p<0.1, 1))
    rcor(imode)=r(1,2);
  else
    rcor(imode)=0;
  end
end % for imode
if ~any(rcor)
  stat{1}=0; stat{2}='No significant correlations found';
  return
end % if no correlations found, return

% Return reformed data matrix using most correlated mode
Snew=zeros(size(S));
[val ind]=max(rcor);
Snew(ind,ind)=S(ind,ind);
dataOut2=U*Snew*V;