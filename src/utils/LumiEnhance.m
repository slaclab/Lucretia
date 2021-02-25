
function H = LumiEnhance(D,beta,sigz)

nterm = length(D) ;

for count = 1:nterm
    
    d = D(count) ;
    b = beta(count) ;
    H(count) = 1+d^0.25 * (d^3/(1+d^3)) * ...
              ( log(sqrt(d)+1) + 2 * log(0.8 * b/sigz) ) ;
          
end