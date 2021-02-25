frac=0.2;  % fraction of plot height for magnet schematic

v0=axis;
h0=gca;
p0=get(h0,'Position');
height=p0(4);
p0(4)=(1-frac)*height;
set(h0,'Position',p0);

p1=[p0(1),p0(2)+p0(4),p0(3),frac*height];
h1=axes('Position',p1);
v1=[v0(1),v0(2),-1,1];
plot(v1(1:2),[0,0],'b-')
axis(v1)
set(h1,'Visible','off');
