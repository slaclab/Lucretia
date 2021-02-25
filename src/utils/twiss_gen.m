function twiss = twiss_gen(twss)
% return twiss.(mux,betx,alfx,dx,dpx,muy,bety,alfy,dy,dpy)

twiss_names={'mux','betx','alfx','dx','dpx','muy','bety','alfy','dy','dpy'};

for f=1:length(twiss_names)
    evalc(['twiss.',twiss_names{f},'=twss(:,',num2str(f),')']);
end

return