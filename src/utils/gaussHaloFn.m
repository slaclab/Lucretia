function y = gaussHaloFn(x,haloSIGmin,x_bar,sig,a,b,A,B)

if sig<1e-3
  sig=1e-3;
end
y = A + B*exp(-( (x-x_bar).^2)/(2*sig^2)) + ...
  double((abs(x-x_bar)./sig)>haloSIGmin).*a.*abs(x-x_bar).^-b;
