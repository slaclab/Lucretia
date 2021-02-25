function [rays,xv,sigm,sigv,emit,emitn,twss,eta1,eta2]=AnalyzeRays(rays0,P,dflag,pflag)
%
% [rays,xv,sigm,sigv,emit,emitn,twss,eta1,eta2]=AnalyzeRays(rays0,P,dflag,pflag);
%
% INPUTs:
%
%   rays0 = particle coordinates [x,px,y,py,z,dp] (6xn)
%   P     = beam energy (GeV/c)
%   dflag = 0 -> no dispersion correction
%           1 -> remove 1st order dp correlations (x,px)
%           2 -> remove 1st and 2nd order dp correlations (x,px)
%           3 -> remove 1st order dp correlations (y,py)
%           4 -> remove 1st and 2nd order dp correlations (y,py)
%           5 -> remove 1st order dp correlations (x,px,y,py)
%           6 -> remove 1st and 2nd order dp correlations (x,px,y,py)
%   pflag = 0 -> no print
%           1 -> print
%
% OUTPUTs:
%
%   rays  = corrected particle coordinates [x,px,y,py,z,dp] (6xn)
%   xv    = mean values [x,px,y,py,z,dp]
%   sigm  = rms beam matrix (6x6)
%   sigv  = Gaussian fits [x,px,y,py,z,dp]
%   emit  = geometric emittances [ex,ey,e1,e2]
%   emitn = normalized emittances [exn,eyn,e1n,e2n]
%   twss  = twiss [bx,ax,gx,by,ay,gy]
%   eta1  = 1st order dispersion [dx,dpx,dy,dpy]
%   eta2  = 2nd order dispersion [ddx,ddpx,ddy,ddpy]

if (~exist('pflag','var'))
  pflag=1;
end

% extract coordinate distributions

x=rays0(:,1);  % m
px=rays0(:,2); % r
y=rays0(:,3);  % m
py=rays0(:,4); % r
z=rays0(:,5);  % m
dp=rays0(:,6); % 1

% compute and correct dp correlations

dx=0;ddx=0;
dpx=0;ddpx=0;
dy=0;ddy=0;
dpy=0;ddpy=0;

if ((dflag==1)||(dflag==5))
  coef=polyfit(dp,x,1);
  dx=coef(1);
  coef(2)=0;
  x=x-polyval(coef,dp);
  coef=polyfit(dp,px,1);
  dpx=coef(1);
  coef(2)=0;
  px=px-polyval(coef,dp);
end
if ((dflag==2)||(dflag==6))
  coef=polyfit(dp,x,2);
  ddx=coef(1);
  dx=coef(2);
  coef(3)=0;
  x=x-polyval(coef,dp);
  coef=polyfit(dp,px,2);
  ddpx=coef(1);
  dpx=coef(2);
  coef(3)=0;
  px=px-polyval(coef,dp);
end
if ((dflag==3)||(dflag==5))
  coef=polyfit(dp,y,1);
  dy=coef(1);
  coef(2)=0;
  y=y-polyval(coef,dp);
  coef=polyfit(dp,py,1);
  dpy=coef(1);
  coef(2)=0;
  py=py-polyval(coef,dp);
end
if ((dflag==4)||(dflag==6))
  coef=polyfit(dp,y,2);
  ddy=coef(1);
  dy=coef(2);
  coef(3)=0;
  y=y-polyval(coef,dp);
  coef=polyfit(dp,py,2);
  ddpy=coef(1);
  dpy=coef(2);
  coef(3)=0;
  py=py-polyval(coef,dp);
end

% subtract off the mean

xv=zeros(1,6);
xv(1)=mean(x);x=x-xv(1);
xv(2)=mean(px);px=px-xv(2);
xv(3)=mean(y);y=y-xv(3);
xv(4)=mean(py);py=py-xv(4);
xv(5)=mean(z);z=z-xv(5);
xv(6)=mean(dp);dp=dp-xv(6);

% do Gaussian fits

nbin=100;
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
[v,u]=hist(x,nbin);
[xfit,q,dq,chi2]=gauss_fit(u,v,[],0);
xg=q(4);
[v,u]=hist(px,nbin);
[pxfit,q,dq,chi2]=gauss_fit(u,v,[],0);
pxg=q(4);
[v,u]=hist(y,nbin);
[yfit,q,dq,chi2]=gauss_fit(u,v,[],0);
yg=q(4);
[v,u]=hist(py,nbin);
[pyfit,q,dq,chi2]=gauss_fit(u,v,[],0);
pyg=q(4);
[v,u]=hist(z,nbin);
[zfit,q,dq,chi2]=gauss_fit(u,v,[],0);
zg=q(4);
[v,u]=hist(dp,nbin);
[dpfit,q,dq,chi2]=gauss_fit(u,v,[],0);
dpg=q(4);
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
sigv=[xg;pxg;yg;pyg;zg;dpg];sigv=abs(sigv);

% extract geometric (projected) emittances and Twiss

rays=[x,px,y,py,z,dp];
sigm=cov(rays);
g=P/0.51099906e-3;
ex=sqrt(det(sigm(1:2,1:2)));
bx=sigm(1,1)/ex;
ax=-sigm(2,1)/ex;
gx=sigm(2,2)/ex;
ey=sqrt(det(sigm(3:4,3:4)));
by=sigm(3,3)/ey;
ay=-sigm(4,3)/ey;
gy=sigm(4,4)/ey;

% extract normal-mode emittances

S=[ 0  1  0  0  0  0 ;
   -1  0  0  0  0  0 ;
    0  0  0  1  0  0 ;
    0  0 -1  0  0  0 ;
    0  0  0  0  0  1 ;
    0  0  0  0 -1  0 ];
[d,v]=eig(sigm*S);
dvec=max(d(1:2,:));[dmax,n]=max(dvec);e1=imag(v(n,n));
dvec=max(d(3:4,:));[dmax,n]=max(dvec);e2=imag(v(n,n));

% set up output variables

emit=[ex,ey,e1,e2];
emitn=g*emit;
twss=[bx,ax,gx,by,ay,gy];
eta1=[dx,dpx,dy,dpy];
eta2=[ddx,ddpx,ddy,ddpy];

% print

if (pflag)
  r21=sigm(2,1)/sqrt(sigm(1,1)*sigm(2,2));
  r43=sigm(4,3)/sqrt(sigm(3,3)*sigm(4,4));
  r65=sigm(6,5)/sqrt(sigm(5,5)*sigm(6,6));
  dLwx=-sigm(2,1)/sigm(2,2);
  dLwy=-sigm(4,3)/sigm(4,4);
  disp(' ')
  disp(sprintf('   sigx   = %11.6f      um (rms = %11.6f)',1e6*[sigv(1),sqrt(sigm(1,1))]))
  disp(sprintf('   sigxp  = %11.6f      ur (rms = %11.6f)',1e6*[sigv(2),sqrt(sigm(2,2))]))
  disp(sprintf('                                (r21 = %11.6f)',r21))
  disp(sprintf('   sigy   = %11.6f      um (rms = %11.6f)',1e6*[sigv(3),sqrt(sigm(3,3))]))
  disp(sprintf('   sigyp  = %11.6f      ur (rms = %11.6f)',1e6*[sigv(4),sqrt(sigm(4,4))]))
  disp(sprintf('                                (r43 = %11.6f)',r43))
  disp(sprintf('   sigz   = %11.6f      mm (rms = %11.6f)',1e3*[sigv(5),sqrt(sigm(5,5))]))
  disp(sprintf('   sigdp  = %11.6f      pm (rms = %11.6f)',1e3*[sigv(6),sqrt(sigm(6,6))]))
  disp(sprintf('                                (r65 = %11.6f)',r65))
  disp(' ')
  disp(sprintf('   emitxn = %16.6e m',emitn(1)))
  disp(sprintf('   emitx  = %16.6e m',emit(1)))
  disp(sprintf('   betx   = %11.6f      m',twss(1)))
  disp(sprintf('   alfx   = %11.6f',twss(2)))
  disp(sprintf('   dLwx   = %11.6f      m',dLwx))
  if (ismember(dflag,[1,2,5,6]))
  disp(sprintf('   dx     = %11.6f      m',eta1(1)))
  disp(sprintf('   dpx    = %11.6f',eta1(2)))
  end
  if (ismember(dflag,[2,6]))
  disp(sprintf('   ddx    = %11.6f      m',eta2(1)))
  disp(sprintf('   ddpx   = %11.6f',eta2(2)))
  end
  disp(' ')
  disp(sprintf('   emityn = %16.6e m',emitn(2)))
  disp(sprintf('   emity  = %16.6e m',emit(2)))
  disp(sprintf('   bety   = %11.6f      m',twss(4)))
  disp(sprintf('   alfy   = %11.6f',twss(5)))
  disp(sprintf('   dLwy   = %11.6f      m',dLwy))
  if (ismember(dflag,[3,4,5,6]))
  disp(sprintf('   dy     = %11.6f      m',eta1(3)))
  disp(sprintf('   dpy    = %11.6f',eta1(4)))
  end
  if (ismember(dflag,[4,6]))
  disp(sprintf('   ddy    = %11.6f      m',eta2(3)))
  disp(sprintf('   ddpy   = %11.6f',eta2(4)))
  end
  disp(' ')
  disp(sprintf('   emit1n = %16.6e m',emitn(3)))
  disp(sprintf('   emit1  = %16.6e m',emit(3)))
  disp(sprintf('   emit2n = %16.6e m',emitn(4)))
  disp(sprintf('   emit2  = %16.6e m',emit(4)))
  disp(' ')
end
