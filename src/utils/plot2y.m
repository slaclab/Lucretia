%function dummy=plot2y(x,y1,y2,xt,y1t,y2t)
[AX,HL,HR]=plotyy(x,y1,x,y2);
set(AX(2),'YColor','k');
v=get(AX(1),'Position');
v(3)=1-2*v(1);
set(AX,'Position',v)
xlabel(xt)
axes(AX(1))
ylabel(y1t)
axes(AX(2))
ylabel(y2t,'Rotation',-90)
