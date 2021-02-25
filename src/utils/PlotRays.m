function PlotRays(rays,program,asym)

if (~exist('program','var'))
  program='TURTLE';
end
if (~exist('asym','var'))
  asym=0;
end

% convert units

xdat=1e6*rays(:,1);  % um
pxdat=1e6*rays(:,2); % urad
ydat=1e6*rays(:,3);  % um
pydat=1e6*rays(:,4); % urad
zdat=1e3*rays(:,5);  % mm
dpdat=1e3*rays(:,6); % pm
sig=cov(rays);
nbin=100;
nsig=3;

% x-px phase space plot

figure
clf,subplot,hold off

subplot(221)
h1=plot(xdat,pxdat,'.');
hold on
plot_ellipse(inv(nsig*(1e12*sig(1:2,1:2))),mean(xdat),mean(pxdat),'r-')
hold off
title(sprintf('%s rays',deblank(program)))
xlabel('x (um)')
ylabel('px (urad)')
subplot(222)

warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
[v,u]=hist(pxdat,nbin);
if (asym)
  [pxfit,q,dq,chi2]=agauss_fit(u,v,[],0);
else
  [pxfit,q,dq,chi2]=gauss_fit(u,v,[],0);
end
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
pxg=q(4);
h2=barh(u,v);
set(h2,'EdgeColor',[0,0,1],'FaceColor',[0,0,1])
hold on
plot(pxfit,u,'r-')
hold off
title(['\sigma_{px} = ',num2str(pxg),' urad'])
ylabel('px (urad)')

subplot(223)
[v,u]=hist(xdat,nbin);
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
if (asym)
  [xfit,q,dq,chi2]=agauss_fit(u,v,[],0);
else
  [xfit,q,dq,chi2]=gauss_fit(u,v,[],0);
end
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
xg=q(4);
h3=bar(u,v);
set(h3,'EdgeColor',[0,0,1],'FaceColor',[0,0,1])
hold on
plot(u,xfit,'r-')
hold off
title(['\sigma_x = ',num2str(xg),' um'])
xlabel('x (um)')

subplot(223),a=axis;v=a(1:2);
subplot(222),a=axis;v=[v,a(3:4)];
subplot(221),axis(v)

% y-py phase space plot

figure
clf,subplot,hold off

subplot(221)
h1=plot(ydat,pydat,'.');
hold on
plot_ellipse(inv(nsig*(1e12*sig(3:4,3:4))),mean(ydat),mean(pydat),'r-')
hold off
title(sprintf('%s rays',deblank(program)))
xlabel('y (um)')
ylabel('py (urad)')

subplot(222)
[v,u]=hist(pydat,nbin);
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
if (asym)
  [pyfit,q,dq,chi2]=agauss_fit(u,v,[],0);
else
  [pyfit,q,dq,chi2]=gauss_fit(u,v,[],0);
end
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
pyg=q(4);
h2=barh(u,v);
set(h2,'EdgeColor',[0,0,1],'FaceColor',[0,0,1])
hold on
plot(pyfit,u,'r-')
hold off
title(['\sigma_{py} = ',num2str(pyg),' urad'])
ylabel('py (urad)')

subplot(223)
[v,u]=hist(ydat,nbin);
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
if (asym)
  [yfit,q,dq,chi2]=agauss_fit(u,v,[],0);
else
  [yfit,q,dq,chi2]=gauss_fit(u,v,[],0);
end
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
yg=q(4);
h3=bar(u,v);
set(h3,'EdgeColor',[0,0,1],'FaceColor',[0,0,1])
hold on
plot(u,yfit,'r-')
hold off
title(['\sigma_y = ',num2str(yg),' um'])
xlabel('y (um)')

subplot(223),a=axis;v=a(1:2);
subplot(222),a=axis;v=[v,a(3:4)];
subplot(221),axis(v)

% z-dp phase space plot

figure
clf,subplot,hold off

subplot(221)
h1=plot(zdat,dpdat,'.');
hold on
plot_ellipse(inv(nsig*(1e6*sig(5:6,5:6))),mean(zdat),mean(dpdat),'r-')
hold off
title(sprintf('%s rays',deblank(program)))
xlabel('z (mm)')
ylabel('dp (pm)')

subplot(222)
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
[v,u]=hist(dpdat,nbin);
if (asym)
  [dpfit,q,dq,chi2]=agauss_fit(u,v,[],0);
else
  [dpfit,q,dq,chi2]=gauss_fit(u,v,[],0);
end
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
dpg=q(4);
h2=barh(u,v);
set(h2,'EdgeColor',[0,0,1],'FaceColor',[0,0,1])
hold on
plot(dpfit,u,'r-')
hold off
title(['\sigma_{dp} = ',num2str(dpg),' pm'])
ylabel('dp (pm)')

subplot(223)
[v,u]=hist(zdat,nbin);
warning off MATLAB:rankDeficientMatrix
warning off MATLAB:nearlySingularMatrix
if (asym)
  [zfit,q,dq,chi2]=agauss_fit(u,v,[],0);
else
  [zfit,q,dq,chi2]=gauss_fit(u,v,[],0);
end
warning on MATLAB:rankDeficientMatrix
warning on MATLAB:nearlySingularMatrix
zg=q(4);
h3=bar(u,v);
set(h3,'EdgeColor',[0,0,1],'FaceColor',[0,0,1])
hold on
plot(u,zfit,'r-')
hold off
title(['\sigma_z = ',num2str(zg),' mm'])
xlabel('z (mm)')

subplot(223),a=axis;v=a(1:2);
subplot(222),a=axis;v=[v,a(3:4)];
subplot(221),axis(v)
