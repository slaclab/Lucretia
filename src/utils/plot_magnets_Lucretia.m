function [h0,h1]=plot_magnets_Lucretia(BEAMLINE,colorflag,nameflag,fontsize,fontname)
%
% [h0,h1]=plot_magnets_Lucretia(BEAMLINE,colorflag,nameflag,fontsize,fontname);
%
% Add magnet schematic to a plot (compatible with Lucretia).
% Single-column subplots are OK.
%
% INPUTs:
%
%   BEAMLINE  = Lucretia BEAMLINE structure
%   colorflag = magnets in color (1) or not (0) (OPTIONAL) [default=1]
%   nameflag  = element names (1) or not (0) (OPTIONAL) [default=0]
%   fontsize  = fontsize for names (OPTIONAL) [default=6]
%   fontname  = fontname for names (OPTIONAL) [default='helvetica'];
%
% OUTPUTs:
%
%   h0 = handle(s) to original plot(s)
%   h1 = handle to magnet schematic

if (nargin==1),colorflag=1;end
names=((nargin>2)&&(nameflag~=0));
if (names)
  if (nargin<5),fontname='helvetica';end
  if (nargin<4),fontsize=6;end
end

ha=0.5;    % half-height of RF rectangle
hb=1;      % full height of bend rectangle
hq=4;      % full height of quadrupole rectangle
hs=3;      % full height of sextupole rectangle
ho=2;      % full height of octupole rectangle
hr=1;      % half-height of solenoid rectangle
tol=1e-6;  % used to unsplit devices

Nelem=length(BEAMLINE);
N=cell(Nelem,1);
S=zeros(Nelem,1); % S-position of element entrance (!)
L=zeros(Nelem,1);
for n=1:Nelem
  if (isfield(BEAMLINE{n},'Name')),N{n}=BEAMLINE{n}.Name;end
  if (isfield(BEAMLINE{n},'S')),S(n)=BEAMLINE{n}.S;end
  if (isfield(BEAMLINE{n},'L')),L(n)=BEAMLINE{n}.L;end
end
N=char(N);
S=[S(2:end);S(end)]; % S-position of element exit

% RF

ida=findcells(BEAMLINE,'Class','LCAV');
if (~isempty(ida))
  if (ida(1)==1),ida(1)=[];end
end
if (isempty(ida))
  Na=0;
else
  xa=zeros(size(ida));
  wa=zeros(size(ida));
  ya=zeros(size(ida));
  n=ida(1);
  Na=1;
  xa(Na)=S(n);
  wa(Na)=L(n);
  ya(Na)=ha;
  if (names),na=N(n,:);end
  for m=2:length(ida)
    n=ida(m);
    if (abs(S(n-1)-xa(Na))<tol)
      xa(Na)=S(n);
      wa(Na)=wa(Na)+L(n);
    else
      Na=Na+1;
      xa(Na)=S(n);
      wa(Na)=L(n);
      ya(Na)=ha;
      if (names),na=[na;N(n,:)];end
    end
  end
  xa(Na+1:end)=[];
  wa(Na+1:end)=[];
  ya(Na+1:end)=[];
  xa=xa-wa/2;
end

% bends

idb=findcells(BEAMLINE,'Class','SBEN');
if (~isempty(idb))
  if (idb(1)==1),idb(1)=[];end
end
if (isempty(idb))
  Nb=0;
else
  xb=zeros(size(idb));
  wb=zeros(size(idb));
  yb=zeros(size(idb));
  n=idb(1);
  Nb=1;
  xb(Nb)=S(n);
  wb(Nb)=L(n);
  if (BEAMLINE{n}.Tilt==0)
    yb(Nb)=hb;
  else
    yb(Nb)=-hb;
  end
  if (names),nb=N(n,:);end
  for m=2:length(idb)
    n=idb(m);
    if (abs(S(n-1)-xb(Nb))<tol)
      xb(Nb)=S(n);
      wb(Nb)=wb(Nb)+L(n);
    else
      Nb=Nb+1;
      xb(Nb)=S(n);
      wb(Nb)=L(n);
      if (BEAMLINE{n}.Tilt==0)
        yb(Nb)=hb;
      else
        yb(Nb)=-hb;
      end
      if (names),nb=[nb;N(n,:)];end
    end
  end
  xb(Nb+1:end)=[];
  wb(Nb+1:end)=[];
  yb(Nb+1:end)=[];
  xb=xb-wb/2;
end

% quads

idq=findcells(BEAMLINE,'Class','QUAD');
if (~isempty(idq))
  if (idq(1)==1),idq(1)=[];end
end
if (isempty(idq))
  Nq=0;
else
  xq=zeros(size(idq));
  wq=zeros(size(idq));
  yq=zeros(size(idq));
  n=idq(1);
  Nq=1;
  xq(Nq)=S(n);
  wq(Nq)=L(n);
  if (BEAMLINE{n}.B<0)
    yq(Nq)=-hq;
  else
    yq(Nq)=hq;
  end
  if (names),nq=N(n,:);end
  for m=2:length(idq)
    n=idq(m);
    if (abs(S(n-1)-xq(Nq))<tol)
      xq(Nq)=S(n);
      wq(Nq)=wq(Nq)+L(n);
    else
      Nq=Nq+1;
      xq(Nq)=S(n);
      wq(Nq)=L(n);
      if (BEAMLINE{n}.B<0)
        yq(Nq)=-hq;
      else
        yq(Nq)=hq;
      end
      if (names),nq=[nq;N(n,:)];end
    end
  end
  xq(Nq+1:end)=[];
  wq(Nq+1:end)=[];
  yq(Nq+1:end)=[];
  xq=xq-wq/2;
end

% sexts

ids=findcells(BEAMLINE,'Class','SEXT');
if (~isempty(ids))
  if (ids(1)==1),ids(1)=[];end
end
if (isempty(ids))
  Ns=0;
else
  xs=zeros(size(ids));
  ws=zeros(size(ids));
  ys=zeros(size(ids));
  n=ids(1);
  Ns=1;
  xs(Ns)=S(n);
  ws(Ns)=L(n);
  if (BEAMLINE{n}.B<0)
    ys(Ns)=-hs;
  else
    ys(Ns)=hs;
  end
  if (names),ns=N(n,:);end
  for m=2:length(ids)
    n=ids(m);
    if (abs(S(n-1)-xs(Ns))<tol)
      xs(Ns)=S(n);
      ws(Ns)=ws(Ns)+L(n);
    else
      Ns=Ns+1;
      xs(Ns)=S(n);
      ws(Ns)=L(n);
      if (BEAMLINE{n}.B<0)
        ys(Ns)=-hs;
      else
        ys(Ns)=hs;
      end
      if (names),ns=[ns;N(n,:)];end
    end
  end
  xs(Ns+1:end)=[];
  ws(Ns+1:end)=[];
  ys(Ns+1:end)=[];
  xs=xs-ws/2;
end

% octs

ido=findcells(BEAMLINE,'Class','OCTU');
if (~isempty(ido))
  if (ido(1)==1),ido(1)=[];end
end
if (isempty(ido))
  No=0;
else
  xo=zeros(size(ido));
  wo=zeros(size(ido));
  yo=zeros(size(ido));
  n=ido(1);
  No=1;
  xo(No)=S(n);
  wo(No)=L(n);
  if (BEAMLINE{n}.B<0)
    yo(No)=-ho;
  else
    yo(No)=ho;
  end
  if (names),no=N(n,:);end
  for m=2:length(ido)
    n=ido(m);
    if (abs(S(n-1)-xo(No))<tol)
      xo(No)=S(n);
      wo(No)=wo(No)+L(n);
    else
      No=No+1;
      xo(No)=S(n);
      wo(No)=L(n);
      if (BEAMLINE{n}.B<0)
        yo(No)=-ho;
      else
        yo(No)=ho;
      end
      if (names),no=[no;N(n,:)];end
    end
  end
  xo(No+1:end)=[];
  wo(No+1:end)=[];
  yo(No+1:end)=[];
  xo=xo-wo/2;
end

% solenoids

idr=findcells(BEAMLINE,'Class','SOLENOID');
if (~isempty(idr))
  if (idr(1)==1),idr(1)=[];end
end
if (isempty(idr))
  Nr=0;
else
  xr=zeros(size(idr));
  wr=zeros(size(idr));
  yr=zeros(size(idr));
  n=idr(1);
  Nr=1;
  xr(Nr)=S(n);
  wr(Nr)=L(n);
  yr(Nr)=hr;
  if (names),nr=N(n,:);end
  for m=2:length(idr)
    n=idr(m);
    if (abs(S(n-1)-xr(Nr))<tol)
      xr(Nr)=S(n);
      wr(Nr)=wr(Nr)+L(n);
    else
      Nr=Nr+1;
      xr(Nr)=S(n);
      wr(Nr)=L(n);
      yr(Nr)=hr;
      if (names),nr=[nr;N(n,:)];end
    end
  end
  xr(Nr+1:end)=[];
  wr(Nr+1:end)=[];
  yr(Nr+1:end)=[];
  xr=xr-wr/2;
end

if ((Na+Nb+Nq+Ns+No+Nr)==0)
  error('No magnets to plot')
end

% squeeze plot(s) to make room for magnet schematic

if (names)
  frac=0.25;  % fraction of plot height for magnet schematic
else
  frac=0.15;
end
bottom=[];
top=[];
hc=get(gcf,'Children');
h0=[];
for n=1:length(hc)
  if (strcmp(get(hc(n),'Type'),'axes'))
    v=get(hc(n),'Position');
    bottom=min([bottom,v(2)]);
    top=max([top,v(2)+v(4)]);
    h0=[h0;hc(n)];
  end
end
for n=1:length(h0)
  v=get(h0(n),'Position');
  if (v(2)>bottom),v(2)=v(2)*(1-frac);end
  v(4)=v(4)*(1-frac);
  set(h0(n),'Position',v)
end
height=top-bottom;

% get horizontal position and width, horizontal axis limits, and title
% (NOTE: we assume that all subplots have the same horizontal position
%        and limits, and that there is only one title among the subplots)

p0=get(h0(1),'Position');
axes(h0(1));
v0=axis;
n=0;
found=0;
while ((n<length(h0))&(~found))
  n=n+1;
  ht=get(h0(n),'Title');
  tt=get(ht,'String');
  found=(~isempty(tt));
end
if (found)
  set(ht,'String',[])
end

% create axes for magnet schematic

p1=[p0(1),top-height*frac,p0(3),height*frac];
h1=axes('Position',p1);
if (names)
  v1=[v0(1),v0(2),-6,12];
else
  v1=[v0(1),v0(2),-6,6];
end
plot(S([1,end]),[0,0],'k-')
axis(v1)
set(h1,'Visible','off');

% make the magnet schematic (use the "rectangle" command);
% add names if requested
% (NOTE: ignore zero-length elements)

hold on
if (Na>0)
  for n=1:Na
    x=xa(n)-wa(n)/2;
    w=wa(n);
    if (w>0)
      y=-ya(n);
      h=2*ya(n);
      h=rectangle('Position',[x,y,w,h],'FaceColor','none');
      if (colorflag),set(h,'FaceColor','y'),end
      if (names)
        text(xa(n),5,literal(na(n,:)),'Rotation',90, ...
          'FontSize',fontsize,'FontName',fontname, ...
          'HorizontalAlignment','left','VerticalAlignment','middle')
      end
    end
  end
end
if (Nb>0)
  for n=1:Nb
    x=xb(n)-wb(n)/2;
    w=wb(n);
    if (w>0)
      if (yb(n)<0)
        y=yb(n);
        h=abs(yb(n));
      else
        y=0;
        h=yb(n);
      end
      h=rectangle('Position',[x,y,w,h],'FaceColor','none');
      if (colorflag),set(h,'FaceColor','b'),end
      if (names)

%       shave off last character in name

        nc=length(deblank(nb(n,:)))-1;
        text(xb(n),5,literal(nb(n,1:nc)),'Rotation',90, ...
          'FontSize',fontsize,'FontName',fontname, ...
          'HorizontalAlignment','left','VerticalAlignment','middle')
      end
    end
  end
end
if (Nq>0)
  for n=1:Nq
    x=xq(n)-wq(n)/2;
    w=wq(n);
    if (w>0)
      if (yq(n)<0)
        y=yq(n);
        h=abs(yq(n));
      else
        y=0;
        h=yq(n);
      end
      h=rectangle('Position',[x,y,w,h],'FaceColor','none');
      if (colorflag),set(h,'FaceColor','r'),end
      if (names)
        text(xq(n),5,literal(nq(n,:)),'Rotation',90, ...
          'FontSize',fontsize,'FontName',fontname, ...
          'HorizontalAlignment','left','VerticalAlignment','middle')
      end
    end
  end
end
if (Ns>0)
  for n=1:Ns
    x=xs(n)-ws(n)/2;
    w=ws(n);
    if (w>0)
      if (ys(n)<0)
        y=ys(n);
        h=abs(ys(n));
      else
        y=0;
        h=ys(n);
      end
      h=rectangle('Position',[x,y,w,h],'FaceColor','none');
      if (colorflag),set(h,'FaceColor','g'),end
      if (names)
        text(xs(n),5,literal(ns(n,:)),'Rotation',90, ...
          'FontSize',fontsize,'FontName',fontname, ...
          'HorizontalAlignment','left','VerticalAlignment','middle')
      end
    end
  end
end
if (No>0)
  for n=1:No
    x=xo(n)-wo(n)/2;
    w=wo(n);
    if (w>0)
      if (yo(n)<0)
        y=yo(n);
        h=abs(yo(n));
      else
        y=0;
        h=yo(n);
      end
      h=rectangle('Position',[x,y,w,h],'FaceColor','none');
      if (colorflag),set(h,'FaceColor','c'),end
      if (names)
        text(xo(n),5,literal(no(n,:)),'Rotation',90, ...
          'FontSize',fontsize,'FontName',fontname, ...
          'HorizontalAlignment','left','VerticalAlignment','middle')
      end
    end
  end
end
if (Nr>0)
  for n=1:Nr
    x=xr(n)-wr(n)/2;
    w=wr(n);
    if (w>0)
      y=-yr(n);
      h=2*yr(n);
      h=rectangle('Position',[x,y,w,h],'FaceColor','none');
      if (colorflag),set(h,'FaceColor','m'),end
      if (names)
        text(xr(n),5,literal(nr(n,:)),'Rotation',90, ...
          'FontSize',fontsize,'FontName',fontname, ...
          'HorizontalAlignment','left','VerticalAlignment','middle')
      end
    end
  end
end
hold off

% stick the title over everything

if (found)
  title(tt)
  ht=get(h1,'Title');
  set(ht,'Visible','on')
end

set(gcf,'NextPlot','replace')
