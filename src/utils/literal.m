function strl = literal(str);

%				strl = literal(str);
%				
%				Creates the TEX-literal equivalent of the input string by replacing
%				underscores within "str" with the new string '\_'.  This makes
%				the text appear in literal format on xlabel, etc. commands
%				rather than in TEX format (where the underscore is a subscript command).
%				
%			INPUTS:	str	Input character string (1 by n vector)
%			OUTPUTS:	strl:	Same as "str" with all '_' replaced by '\_'
%
%===================================================================================

i  = find(str=='_');
ni = length(i);
n  = length(str);
if ni > 0
  if i(1) == 1
    strl = [];
  else
    strl = str(1:(i(1)-1));
  end
  for j = 1:ni
    if i(j) < n
      if j < ni
        strl = [strl '\_' str((i(j)+1):(i(j+1)-1))];
      else
        strl = [strl '\_' str((i(j)+1):n)];
      end
    else
      strl = [strl '\_'];
    end
  end
else
  strl = str;
end