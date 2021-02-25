function s=prompt(str,options,deflt_option)

% s=prompt(str,options[,deflt_option]);
%
% Prompts user for input of ONE character from options string.  It uses only
% the first character of the user's response, converted to lower case, and
% returns this character which will then match at least one of the characters
% in the options string.  Each character in the options string is an acceptable
% choice.  If the user types a character not in the options string, the
% function displays a message and prompts again.
%
% INPUTs:
%
%   str          = the prompt string (the list of options and a colon are
%                  tacked on at the end)
%   options      = a character string of options 
%   deflt_option = one character from options which is the default selection
%                  when <CR> is the response to the prompt
%                  (OPTIONAL ... if not present, no default option is assumed;
%                  the <CR> response to the prompt returns an empty matrix)
%
% OUTPUT:
%
%   s            = the character chosen by the user which has been converted to
%                  lower case, is guaranteed to be only one character in
%                  length, and matches at least one of the characters in the
%                  options string
%
% EXAMPLE:
%
% » opts=['yn'];
% » s=prompt('  Plot this again?',opts);
%
%   Plot this again? [y/n]: 
%
%   (now the user may only enter 'y', or 'Y', or 'n', or 'N')

n=length(options);
if (n<=1)
  error('  < 2 options makes no sense')
end
options=lower(options);
if (~exist('deflt_option'))
  no_default=1;
else
  deflt_option=lower(deflt_option(1));
  id=find(deflt_option==options);
  if (length(id)==0)
    error('  "default_option" must be one of the provided "options"')
  end
  no_default=0;
end

opts=' (';
for j=1:(n-1)
  opts=[opts options(j) '/'];
end
opts=[opts options(n) ')'];
if (no_default)
  opts=[opts ': '];
else
  opts=[opts ' [' deflt_option ']: '];
end
str=['  ' str opts];

while (1)
  disp(' ')
  s=input(str,'s');
  ns=length(s);
  if (ns==0)
    if (no_default)
      s=[];
    else
      s=deflt_option;
    end
    return
  else
    s=lower(s(1));
    if (find(options==s))
      return
    else
      disp(' ')
      disp(['  Select only from the following: ' options])
    end
  end
end
