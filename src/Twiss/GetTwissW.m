function [stat,T] = GetTwissW(i1,i2,Tx,Ty)
% [stat,T] = GetTwissW(i1,i2,Tx,Ty)
% Return the chromatic W function in addition to the standard GetTwiss
% output
global BEAMLINE
de=1e-2;
[stat,T]=GetTwiss(i1,i2,Tx,Ty);
if stat{1}~=1; return; end
P0=zeros(1,1+(i2-i1));
ind=0;
for ibl=i1:i2
  if isfield(BEAMLINE{ibl},'P')
    ind=ind+1;
    P0(ind)=BEAMLINE{ibl}.P;
    BEAMLINE{ibl}.P=P0(ind)*(1+de);
  end
end
[stat,Tp]=GetTwiss(i1,i2,Tx,Ty);
if stat{1}~=1; return; end
ind=0;
for ibl=i1:i2
  if isfield(BEAMLINE{ibl},'P')
    ind=ind+1;
    BEAMLINE{ibl}.P=P0(ind)*(1-de);
  end
end
[stat,Tm]=GetTwiss(i1,i2,Tx,Ty);
if stat{1}~=1; return; end
ind=0;
for ibl=i1:i2
  if isfield(BEAMLINE{ibl},'P')
    ind=ind+1;
    BEAMLINE{ibl}.P=P0(ind);
  end
end
T.Wx=zeros(1,length(T)); T.Wy=T.Wx;
for itw=1:length(T.betax)
  p=polyfit([-de 0 de],[Tm.betax(itw) T.betax(itw) Tp.betax(itw)],1);
  dbx=p(1);
  p=polyfit([-de 0 de],[Tm.betay(itw) T.betay(itw) Tp.betay(itw)],1);
  dby=p(1);
  p=polyfit([-de 0 de],[Tm.alphax(itw) T.alphax(itw) Tp.alphax(itw)],1);
  dax=p(1);
  p=polyfit([-de 0 de],[Tm.alphay(itw) T.alphay(itw) Tp.alphay(itw)],1);
  day=p(1);
  T.Wx(itw)=sqrt( (dax-(T.alphax(itw)/T.betax(itw))*dbx)^2 + (dbx/T.betax(itw))^2 ) ;
  T.Wy(itw)=sqrt( (day-(T.alphay(itw)/T.betay(itw))*dby)^2 + (dby/T.betay(itw))^2 ) ;
end