function h=plot_magnet(coor,S,idi,ido,w,c)
if (idi==ido)
  zc=coor(ido,3)-w/2;
  xc=coor(ido,1)-w/2;
  h=rectangle('Position',[zc,xc,w,w],'Curvature',[1,1],'FaceColor',c);
else
  zc=mean(coor([idi;ido],3));
  xc=mean(coor([idi;ido],1));
  tc=mean(coor([idi;ido],4));
  dS=(S(ido)-S(idi))/2;
  z=dS*[-1,1,1,-1];
  x=(w/2)*[-1,-1,1,1];
  sintc=sin(tc);
  costc=cos(tc);
  t=[costc,-sintc;sintc,costc]*[z;x];
  z=zc+t(1,:)';
  x=xc+t(2,:)';
  h=patch(z,x,c);
end
