function [indx]=find_name(name,names)
indx=[];
[nn,ln]=size(names);
nameb=[name blanks(ln-length(name))];
for n=1:nn
   if strcmp(names(n,:),nameb)
      indx=[indx;n];
   end
end
