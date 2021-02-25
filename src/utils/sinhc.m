function y=sinhc(x)

%  y=sinhc(x)
%
%   sinhc(x) = sinh(x)/x           for |x| >= 1e-3
%            = 1+(x^2)/3!+(x^4)/5! for |x| <  1e-3

[nrow,ncol]=size(x);
y=zeros(nrow,ncol);
for n=1:nrow
   for m=1:ncol
      if (abs(x(n,m))<1e-3)
         y(n,m)=1+((x(n,m)^2)/6)+((x(n,m)^4)/120);
      else
         y(n,m)=sinh(x(n,m))/x(n,m);
      end
   end
end