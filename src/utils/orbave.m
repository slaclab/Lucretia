function [ua,dua]=orbave(u,du)
%
% [ua,dua]=orbave(u,du);
%
% Form the weighted average of nominally identical orbits.
%
% INPUTs
%
%   u  = orbit values (Norbit,Nbpm)
%   du = errors on orbit values (Norbit,Nbpm)
%
% OUTPUTs
%
%   ua  = weighted average orbit values (1,Nbpm)
%   dua = combined errors on averaged orbit values (1,Nbpm)

[Norbit,Nbpm]=size(u);
if (Norbit>1)
  ua=zeros(1,Nbpm);
  dua=zeros(1,Nbpm);
  for n=1:Nbpm
    [ua(n),dua(n)]=noplot_polyfit([1:Norbit]',u(:,n),du(:,n),0);
  end
else
  ua=u;
  dua=du;
end
