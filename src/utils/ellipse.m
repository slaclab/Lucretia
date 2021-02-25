function [x,y]=ellipse(X,p,q,N)

%  function [x,y]=ellipse(X,p,q)
%
%    Return coordinates which can be used to plot the ellipse described by the
%    2x2 symmetric matrix "X", centered at the point (p,q).
%
%  INPUTS:
%
%    X:     a 2x2 symmetric matrix which describes the ellipse as follows:
%
%            [x y]*X*[x y]' = 1,
%
%           or, with X = [a b 
%                         b c],
%
%	           ax^2 + 2bxy + cy^2  = 1;
%
%           for a beam ellipse, X=inv(sigma)=(1/emit)*[gamma alpha
%                                                      alpha  beta]
%
%    p:     ordinate of ellipse center
%    q:     abscissa of ellipse center
%
%  OUTPUTS:
%
%    x:     a 1001 element column vector of x-coordinates
%    y:     a 1001 element column vector of y-coordinates

%===============================================================================

[r,c]=size(X);
if ((r~=2)||(c~=2))
  error('X must be a 2x2 matrix')
end
if (abs(X(1,2)-X(2,1))>(abs(.001*X(1,2))+eps))
  error('X must be a symmetric matrix')
end

a=X(1,1);
b=X(1,2);
c=X(2,2);

if ~exist('N','var')
  N=20000;
end
theta=linspace(0,pi,ceil(N/2))';
C=cos(theta);
S=sin(theta);

r=sqrt((a*(C.^2)+2*b*(C.*S)+c*(S.^2)).^(-1));
r=[r,r];
C=[C,-C];
S=[S,-S];

x=r.*C+p;         
y=r.*S+q;         
