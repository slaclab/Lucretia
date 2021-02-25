function out_dist=remove_flyers(in_dist,n_sigma)

% out_dist=remove_flyers(in_dist,n_sigma)
%
% Removes flyers from input distribution in_dist by culling values more than n_sigma from the mean
%
% INPUT:
%
%    in_dist  = input distribution
%    n_sigma  = number of sigma to cut at
%
% OUTPUT:
%
%    out_dist = decimated distribution
%

out_dist=in_dist;
id=find(abs(out_dist-mean(out_dist))>n_sigma*std(out_dist));
while ((length(out_dist)>0)&(length(id)>0))
   out_dist(id)=[];
   id=find(abs(out_dist-mean(out_dist))>n_sigma*std(out_dist));
end
