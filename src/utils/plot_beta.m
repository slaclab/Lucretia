function h=plot_beta(s,twss,keyw,ds)

x=[];
y1=[];
y2=[];
for n=1:length(s)
  if (strcmp(keyw(n,:),'DRIF'))
    if (abs(s(n)-s(n-1))>ds)
      d=[s(n-1):ds:s(n)]';
      if (d(end)==s(n)),d(end)=[];end
      bx0=twss(n-1,2);
      ax0=twss(n-1,3);
      gx0=(1+ax0^2)/bx0;
      by0=twss(n-1,7);
      ay0=twss(n-1,8);
      gy0=(1+ay0^2)/by0;
      for m=2:length(d)
        l=d(m)-d(1);
        x=[x;d(m)];
        y1=[y1;bx0-2*l*ax0+gx0*l^2];
        y2=[y2;by0-2*l*ay0+gy0*l^2];
      end
    end
  end
  x=[x;s(n)];
  y1=[y1;twss(n,2)];
  y2=[y2;twss(n,7)];
end
h=plot(x,y1,'b-',x,y2,'g--');
    