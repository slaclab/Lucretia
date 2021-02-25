function h=barmdw(x,y,w,s,c)
if (nargin<5)
  c='none';
end
if (nargin<4)
  s='k-';
end
held=ishold;
h=[];
for n=1:length(x)
  px=[x(n)-w(n)/2;x(n)-w(n)/2;x(n)+w(n)/2;x(n)+w(n)/2;x(n)-w(n)/2];
  py=[0;y(n);y(n);0;0];
  if (strcmp(c,'none'))
    h=[h;plot(px,py,s)];
  else
    h=[h;patch(px,py,c)];
  end
  if ((n==1)&(~held))
    hold on
  end
end
if (held)
  hold on
else
  hold off
end