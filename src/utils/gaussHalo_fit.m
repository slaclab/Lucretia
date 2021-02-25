function [yfit,q,chisq_ndf] = gaussHalo_fit(x,y,NSIGHALO)   

x = x(:);
y = y(:);

arg1(:,1) = x;
arg1(:,2) = y;

% sort data to find the pedistal (the 5% population level)
% and the peak.  Use this data to identify data points 
% 1/e above pedistal

[~,indx]=sort(y);
npnts=length(y);
ymin=y(indx(round(npnts*0.05+1)));  % the 5% point
ymax=y(indx(round(npnts*0.98)));    % the 98% point

% Using prior ymin,ymax, find indicies for y>ymin + 1/e (ymax-ymin)

th=find(y > ymin + (ymax-ymin)/2.7);

% guess that the x0 is at the mean of the above threshold
% points

x_ymax=mean(x(th));

% and that the varience is somehow related to to distribution
% of x with y>threshold

xstd=1.3*std(x(th));
p = [x_ymax xstd 1 0.01 y(1) max(y)];
q = fminsearch(@(x) gaussHalo_min(arg1,NSIGHALO,x),p,optimset('MaxIter',5000,'MaxFunEvals',5000,'Display','off'));
yfit = gaussHaloFn(x,NSIGHALO,q(1),q(2),q(3),q(4),q(5),q(6));
chisq_ndf = norm( (y-yfit) )/sqrt(length(y) - length(q));        
chisq_ndf = chisq_ndf^2;                                           
