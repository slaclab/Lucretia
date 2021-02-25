function stat = AddMessageToStack(stin, msg)
%
% ADDMESSAGETOSTACK Add a message to a Lucretia message stack.
%
% stack_new = AddMessageToStack(stack_old, msg) adds text message msg to 
%    message stack stack_old, returning the resulting message stack in
%    stack_new.  
%
% See also LucretiaStatus, AddStackToStack
%

  stat = stin ;
  ptrin = length(stin) + 1 ;
  stat{ptrin} = msg ;
%  