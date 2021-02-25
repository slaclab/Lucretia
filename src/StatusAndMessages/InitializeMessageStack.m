function stat = InitializeMessageStack( )
%
% INITIALIZEMESSAGESTACK Create a new Lucretia message stack.
%
% stack = InitializeMessageStack( ) creates a new cell array
%    for Lucretia status and messages and initializes the first cell to a
%    value of 1 (indicating no errors).
%
% See also:  LucretiaStatus, AddMessageToStack, AddStackToStack.
%
  stat{1} = 1 ;
%