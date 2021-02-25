function figit(fig,w)
figure(fig)
enhance_plot('arial',14)
if (~exist('w')),w=8.0;end
set(gcf,'Units','inches')
set(gcf,'Position',[1,1,w,0.75*w])
