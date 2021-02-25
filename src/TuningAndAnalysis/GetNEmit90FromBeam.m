function [nx,ny,nz,Tx,Ty]=GetNEmit90FromBeam(B,ibunch)

if ~exist('ibunch','var')
  ibunch=1;
end
rejfrac=0.1;
goodray=~B.Bunch(ibunch).stop;
if ~any(goodray)
  error('All particles stopped')
end
nrej=floor(sum(goodray)*rejfrac);
for idim=[1 3 5]
  Bx=B;
  Bx.Bunch.x=Bx.Bunch(ibunch).x(:,goodray); Bx.Bunch(ibunch).Q=Bx.Bunch(ibunch).Q(goodray); Bx.Bunch(ibunch).stop=Bx.Bunch(ibunch).stop(goodray);
  [~,Ix]=sort(abs(Bx.Bunch(ibunch).x(idim,:)-mean(Bx.Bunch(ibunch).x(idim,:))),'descend');
  [~,Ixp]=sort(abs(Bx.Bunch(ibunch).x(idim+1,:)-mean(Bx.Bunch(ibunch).x(idim+1,:))),'descend');
  I=unique([Ix(nrej+1:end) Ixp(nrej+1:end)]);
  Bx.Bunch(ibunch).x=Bx.Bunch(ibunch).x(:,I); Bx.Bunch(ibunch).Q=Bx.Bunch(ibunch).Q(I); Bx.Bunch(ibunch).stop=Bx.Bunch(ibunch).stop(I);
  if idim==1
    Tx = GetUncoupledTwissFromBeamPars(Bx,ibunch);
    nx=GetNEmitFromBeam(Bx,ibunch);
  elseif idim==3
    [~,Ty] = GetUncoupledTwissFromBeamPars(Bx,ibunch);
    [~,ny]=GetNEmitFromBeam(Bx,ibunch);
  else
    [~,~,nz]=GetNEmitFromBeam(Bx,ibunch);
  end
end