function plot_box(z,L,W)
x1=z-L/2;
x2=z+L/2;
y1=-W/2;
y2=W/2;
x=[x1;x1;x2;x2;x1];
y=[y1;y2;y2;y1;y1];
plot(x,y,'-b')