function f = gaussHalo_min(arg1, NSIGHALO, p)

x  = arg1(:,1); 
y  = arg1(:,2);

A  = gaussHaloFn(x,NSIGHALO,p(1),p(2),p(3),p(4),p(5),p(6));
f=sum((A-y).^2);

              
