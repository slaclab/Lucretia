function stat = AddStackToStack(s1,s2)
%
% ADDSTACKTOSTACK Combine 2 message stacks
%
% stackout = AddStackToStack(stack1, stack2) appends the contents of 
%    Lucretia message stack stack2 onto the contents of message stack
%    stack1. The combined stack is returned.
%
% See also:  LucretiaStatus, InitializeMessageStack, AddMessageToStack
%

stat = s1 ; ptr1 = length(s1) + 1 ;
for count = 2:length(s2)
  stat{ptr1+count-2} = s2{count} ;
end
