function mtols=umagtols(K,L,P,S,E,ex,ey,bx,by,hx,hy,sigx,sigy,sigdp)
%
% mtols=umagtols(K,L,P,S,E,ex,ey,bx,by,hx,hy,sigx,sigy,sigdp);
%
% Compute various magnet tolerances (see The Cheat Sheet for formalism)
%
% Inputs:
%
%   K         = element keyword
%   L         = element length (m)
%   P         = element parameters (XTFF conventions)
%   S         = suml (m)
%   E         = energy (GeV)
%   ex,ey     = unnormalized emittance x,y (m) [emit(:,3),emit(:,7)]
%   bx,by     = beta x,y (m)                   [twss(:,2),twss(:,7)]
%   hx,hy     = eta x,y (m)                    [twss(:,4),twss(:,9)]
%   sigx,sigy = sigma x,y (m)                  [sig(:,1),sig(:,3)]
%   sigdp     = sigma delta (fraction)         [sig(:,6)]
%
%  Output:
%
%   mtols     = table of computed tolerances and other data:
%
%                 mtols(n, 1) = magnet type (0=bend,1=quad,2=sext,3=oct)
%                 mtols(n, 2) = array pointer to magnet
%                 mtols(n, 3) = x-position tolerance
%                 mtols(n, 4) = y-position tolerance
%                 mtols(n, 5) = x-jitter tolerance
%                 mtols(n, 6) = y-jitter tolerance
%                 mtols(n, 7) = roll tolerance
%                 mtols(n, 8) = strength (dB/B) tolerance
%                 mtols(n, 9) = harmonic tolerance (bend: b1/b0; quad: b2/b1)
%                 mtols(n,10) = harmonic tolerance (bend: b2/b0; quad: b5/b1)
%                 mtols(n,11) = harmonic tolerance (bend: b4/b0)
%                 mtols(n,12) = harmonic radius

disp(' ')
ans=input('  Select magnet types (d=dipole,q=quad,s=sext,o=oct,[*=all]): ','s');
if (isempty(ans)),ans='*';end
bend=(~isempty(findstr(ans,'d'))|~isempty(findstr(ans,'*')));
quad=(~isempty(findstr(ans,'q'))|~isempty(findstr(ans,'*')));
sext=(~isempty(findstr(ans,'s'))|~isempty(findstr(ans,'*')));
octu=(~isempty(findstr(ans,'o'))|~isempty(findstr(ans,'*')));
if (bend|quad)
  ans=input('  Select normalization radius for multipoles (m) [0.01]: ');
  if (isempty(ans))
    r0=0.01;
  else
    r0=ans;
  end
end
ans=input('  Select maximum luminosity reduction (1) [0.02]: ');
if (isempty(ans))
  dL=0.02;
else
  dL=ans;
end
disp(' ')

c=2.99792458e8;
Cb=1e10/c;
tol=1e-6;  % for unsplitting magnets
mtols=[];

if (bend)

% find the dipoles ... unsplit them and take the location which is
% closest to the center

  idb=find_name('SBEN',K);        % MAD or ELEGANT
  idb=[idb;find_name('RBEN',K)];  % MAD or ELEGANT
  idb=[idb;find_name('NIBE',K)];  % ELEGANT
  idb=[idb;find_name('CSBE',K)];  % ELEGANT
  idb=[idb;find_name('KSBE',K)];  % ELEGANT
  idb=[idb;find_name('CSRC',K)];  % ELEGANT
  idb=sort(idb);
  if (~isempty(idb))
    if (idb(1)==1),idb(1)=[];end
  end
  if (isempty(idb))
    Nb=0;
  else
    Sb=zeros(size(idb));
    Lb=zeros(size(idb));
    Ab=zeros(size(idb));
    n=idb(1);
    Nb=1;
    Sb(Nb)=S(n);
    Lb(Nb)=L(n);
    Ab(Nb)=P(n,1);
    for m=2:length(idb)
      n=idb(m);
      if (abs(S(n-1)-Sb(Nb))<tol)
        Sb(Nb)=S(n);
        Lb(Nb)=Lb(Nb)+L(n);
        Ab(Nb)=Ab(Nb)+P(n,1);
      else
        Nb=Nb+1;
        Sb(Nb)=S(n);
        Lb(Nb)=L(n);
        Ab(Nb)=P(n,1);
      end
    end
    Sb(Nb+1:end)=[];
    Lb(Nb+1:end)=[];
    Ab(Nb+1:end)=[];
    Sb=Sb-Lb/2;  % approximate magnet center

%   OK, now compute tolerances

    for n=1:Nb
      [dummy,id]=min(abs(S-Sb(n)));  % location nearest to center
      mtols=[mtols;0,id,-1,-1,-1,-1,-1,-1,-1,-1,-1,r0];  % initialize
      theta=abs(Ab(n));
      if (theta~=0)

%       beam properties

        ex0=ex(id);
        ey0=ey(id);
        bx0=bx(id);
        by0=by(id);
        hx0=hx(id);
        hy0=hy(id);
        sx0=sigx(id);
        sy0=sigy(id);
        sd0=sigdp(id);

%       intermediate quantities (see The Cheat Sheet ... )

        rb=by0/bx0;
        re=ey0/ex0;
        lambda=(sy0/sx0)^2;
        zeta2=(hx0*sd0)^2/(bx0*ex0);
        xi2=1+zeta2;
        chi2=by0*ey0/(bx0*ex0);
        D1=sqrt(xi2+rb^2);
        D2=sqrt(xi2^2+chi2^2+2*xi2*rb^2);
        phi=4-2*(3-5*rb/re)*lambda+3*(5-4*rb/re)*lambda^2 ...
          -2*(3-5*rb/re)*lambda^3+4*lambda^4;
        D4=sqrt(3*bx0*phi);

%       the tolerances (see The Cheat Sheet ... )

        roll=2*sqrt(2*ey0*dL/by0)/(theta*sd0);
        dB_B=2*sqrt(2*ex0*dL/bx0)/theta;
        b1_b0=2*r0*sqrt(2*dL)/(theta*bx0*D1);
        b2_b0=2*r0^2*sqrt(dL)/(theta*bx0*sqrt(bx0*ex0)*D2);
        b4_b0=r0^4*sqrt(ex0*dL)/(theta*sx0^4*D4);

%       save 'em

        mtols(end,7)=roll;
        mtols(end,8)=dB_B;
        mtols(end,9)=b1_b0;
        mtols(end,10)=b2_b0;
        mtols(end,11)=b4_b0;
      end
    end
  end
end

if (quad)

% find the quadrupoles ... unsplit them and take the location which is
% closest to the center

  idq=find_name('QUAD',K);        % MAD or ELEGANT
  idq=[idq;find_name('KQUA',K)];  % ELEGANT
  idq=sort(idq);
  if (~isempty(idq))
    if (idq(1)==1),idq(1)=[];end
  end
  if (isempty(idq))
    Nq=0;
  else
    Sq=zeros(size(idq));
    Lq=zeros(size(idq));
    KLq=zeros(size(idq));
    n=idq(1);
    Nq=1;
    Sq(Nq)=S(n);
    Lq(Nq)=L(n);
    KLq(Nq)=P(n,2)*L(n);
    for m=2:length(idq)
      n=idq(m);
      if (abs(S(n-1)-Sq(Nq))<tol)
        Sq(Nq)=S(n);
        Lq(Nq)=Lq(Nq)+L(n);
        KLq(Nq)=KLq(Nq)+P(n,2)*L(n);
      else
        Nq=Nq+1;
        Sq(Nq)=S(n);
        Lq(Nq)=L(n);
        KLq(Nq)=P(n,2)*L(n);
      end
    end
    Sq(Nq+1:end)=[];
    Lq(Nq+1:end)=[];
    KLq(Nq+1:end)=[];
    Sq=Sq-Lq/2;  % magnet center

%   OK, now compute tolerances

    for n=1:Nq
      [dummy,id]=min(abs(S-Sq(n)));  % location nearest to center
      mtols=[mtols;1,id,-1,-1,-1,-1,-1,-1,-1,-1,-1,r0];  % initialize
      kl=abs(KLq(n));
      if (kl~=0)

%       beam properties

        ex0=ex(id);
        ey0=ey(id);
        bx0=bx(id);
        by0=by(id);
        hx0=hx(id);
        hy0=hy(id);
        sx0=sigx(id);
        sy0=sigy(id);
        sd0=sigdp(id);

%       intermediate quantities (see The Cheat Sheet ... )

        rb=by0/bx0;
        re=ey0/ex0;
        lambda=(sy0/sx0)^2;
        zeta2=(hx0*sd0)^2/(bx0*ex0);
        xi2=1+zeta2;
        chi2=by0*ey0/(bx0*ex0);
        D2=sqrt(xi2^2+chi2^2+2*xi2*rb^2);
        phi=63-35*(4-5*rb/re)*lambda+30*(11-10*rb/re)*lambda^2 ...
          -30*(10-11*rb/re)*lambda^3+35*(5-4*rb/re)*lambda^4 ...
          +63*(rb/re)*lambda^5;
        D5=sqrt(15*bx0*phi);

%       the tolerances (see The Cheat Sheet ... )

        dx=2*sqrt(2*ex0*dL/bx0)/(kl*sd0);
        dy=2*sqrt(2*ey0*dL/by0)/(kl*sd0);
        sdx=2*sqrt(2*ex0*dL/bx0)/kl;
        sdy=2*sqrt(2*ey0*dL/by0)/kl;
        roll=sqrt(2*dL)/(kl*sqrt(bx0*by0/re)*sqrt(xi2+re));
        dB_B=2*sqrt(2*dL)/(kl*bx0*sqrt(xi2+rb^2));
        b2_b1=2*r0*sqrt(dL)/(kl*bx0*sqrt(bx0*ex0)*D2);
        b5_b1=r0^4*sqrt(ex0*dL)/(kl*sx0^5*D5);

%       save 'em

        mtols(end,3)=dx;
        mtols(end,4)=dy;
        mtols(end,5)=sdx;
        mtols(end,6)=sdy;
        mtols(end,7)=roll;
        mtols(end,8)=dB_B;
        mtols(end,9)=b2_b1;
        mtols(end,10)=b5_b1;
      end
    end
  end
end

if (sext)

% find the sextupoles ... unsplit them and take the location which is
% closest to the center

  ids=find_name('SEXT',K);        % MAD or ELEGANT
  ids=[ids;find_name('KSEX',K)];  % ELEGANT
  ids=sort(ids);
  if (~isempty(ids))
    if (ids(1)==1),ids(1)=[];end
  end
  if (isempty(ids))
    Ns=0;
  else
    Ss=zeros(size(ids));
    Ls=zeros(size(ids));
    KLs=zeros(size(ids));
    n=ids(1);
    Ns=1;
    Ss(Ns)=S(n);
    Ls(Ns)=L(n);
    KLs(Ns)=P(n,3)*L(n);
    for m=2:length(ids)
      n=ids(m);
      if (abs(S(n-1)-Ss(Ns))<tol)
        Ss(Ns)=S(n);
        Ls(Ns)=Ls(Ns)+L(n);
        KLs(Ns)=KLs(Ns)+P(n,3)*L(n);
      else
        Ns=Ns+1;
        Ss(Ns)=S(n);
        Ls(Ns)=L(n);
        KLs(Ns)=P(n,3)*L(n);
      end
    end
    Ss(Ns+1:end)=[];
    Ls(Ns+1:end)=[];
    KLs(Ns+1:end)=[];
    Ss=Ss-Ls/2;  % magnet center

%   OK, now compute tolerances

    for n=1:Ns
      [dummy,id]=min(abs(S-Ss(n)));  % location nearest to center
      mtols=[mtols;2,id,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];  % initialize
      kl=abs(KLs(n))/2;  % The Cheat Sheet uses field expansion without factorials
      if (kl~=0)

%       beam properties

        ex0=ex(id);
        ey0=ey(id);
        bx0=bx(id);
        by0=by(id);
        hx0=hx(id);
        hy0=hy(id);
        sx0=sigx(id);
        sy0=sigy(id);
        sd0=sigdp(id);

%       intermediate quantities (see The Cheat Sheet ... )

        rb=by0/bx0;
        re=ey0/ex0;
        zeta2=(hx0*sd0)^2/(bx0*ex0);
        xi2=1+zeta2;
        chi2=by0*ey0/(bx0*ex0);

%       the tolerances (see The Cheat Sheet ... )

        dx=sqrt(2*dL)/(kl*bx0*sqrt(xi2+rb^2));
        dy=sqrt(2*dL)/(kl*sqrt(bx0*by0/re)*sqrt(xi2+re^2));
        sdx=dx;
        sdy=dy;
        roll=2*sqrt(dL)/(3*kl*bx0*sqrt(by0*ey0)*sqrt(2*xi2+(xi2/re)^2+rb^2));
        dB_B=2*sqrt(dL)/(kl*bx0*sqrt(bx0*ex0)*sqrt(xi2^2+chi2^2+2*xi2*rb^2));

%       save 'em

        mtols(end,3)=dx;
        mtols(end,4)=dy;
        mtols(end,5)=sdx;
        mtols(end,6)=sdy;
        mtols(end,7)=roll;
        mtols(end,8)=dB_B;
      end
    end
  end
end

if (octu)

% find the octupoles ... unsplit them and take the location which is
% closest to the center

  ido=find_name('OCTU',K);  % MAD
  if (~isempty(ido))
    if (ido(1)==1),ido(1)=[];end
  end
  if (isempty(ido))
    No=0;
  else
    So=zeros(size(ido));
    Lo=zeros(size(ido));
    KLo=zeros(size(ido));
    n=ido(1);
    No=1;
    So(No)=S(n);
    Lo(No)=L(n);
    KLo(No)=P(n,5)*L(n);
    for m=2:length(ido)
      n=ido(m);
      if (abs(S(n-1)-So(No))<tol)
        So(No)=S(n);
        Lo(No)=Lo(No)+L(n);
        KLo(No)=KLo(No)+P(n,5)*L(n);
      else
        No=No+1;
        So(No)=S(n);
        Lo(No)=L(n);
        KLo(No)=P(n,5)*L(n);
      end
    end
    So(No+1:end)=[];
    Lo(No+1:end)=[];
    KLo(No+1:end)=[];
    So=So-Lo/2;  % magnet center

%   OK, now compute tolerances

    for n=1:No
      [dummy,id]=min(abs(S-So(n)));  % location nearest to center
      mtols=[mtols;3,id,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];  % initialize
      kl=abs(KLo(n))/6;  % The Cheat Sheet uses field expansion without factorials
      if (kl~=0)

%       beam properties

        ex0=ex(id);
        ey0=ey(id);
        bx0=bx(id);
        by0=by(id);
        hx0=hx(id);
        hy0=hy(id);
        sx0=sigx(id);
        sy0=sigy(id);
        sd0=sigdp(id);

%       intermediate quantities (see The Cheat Sheet ... )

        rb=by0/bx0;
        re=ey0/ex0;
        zeta2=(hx0*sd0)^2/(bx0*ex0);
        xi2=1+zeta2;
        chi2=by0*ey0/(bx0*ex0);
        Dr=sqrt(5*xi2^3-6*xi2^2*chi2+9*xi2*chi2^2 ...
          +(9*xi2^2-6*xi2*chi2+5*chi2^2)/re^2);
        Db=sqrt(xi2*(5*xi2^2-6*xi2*chi2+9*chi2^2) ...
          +(9*xi2^2-6*xi2*chi2+5*chi2^2)*rb^2);

%       the tolerances (see The Cheat Sheet ... )

        dx=2*sqrt(dL)/(3*kl*bx0*sqrt(bx0*ex0)*sqrt(xi2^2+chi2^2+2*xi2*rb^2));
        dy=2*sqrt(dL)/(3*kl*bx0*sqrt(by0*ey0)*sqrt(2*xi2+(xi2^2+chi2^2)/re^2));
        roll=sqrt(dL)/(sqrt(6)*kl*bx0*ex0*sqrt(bx0*by0/re)*Dr);
        dB_B=2*sqrt(2*dL)/(sqrt(3)*kl*bx0^2*ex0*Db);

%       save 'em

        mtols(end,3)=dx;
        mtols(end,4)=dy;
        mtols(end,7)=roll;
        mtols(end,8)=dB_B;
      end
    end
  end
end
