function [f,g]=fun(X)
f=exp(X(1))*(4*X(1)^2+2*X(2)^2+4*X(1)*X(2)+2*X(2)+1);
g(1)=1.5+X(1)*X(2)-X(1)-X(2);
g(2)=-X(1)*X(2)-10;
