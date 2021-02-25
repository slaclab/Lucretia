function v=plims(x,y,f)
dx=max(x)-min(x);
dy=max(y)-min(y);
v=[min(x)-f*dx,max(x)+f*dx,min(y)-f*dy,max(y)+f*dy];