function DisplayMessageStack( stack )
%
% DISPLAYMESSAGESTACK Display a Lucretia status and message stack.
%
% DisplayMessageStack( stack ) displays the status values in a Lucretia
%    status and message stack, followed by the messages.
%
% See also:  InitializeMessageStack, LucretiaStatus.
%
% Version date:  17-September-2007.

%=========================================================================

for count = 1:length(stack)
    disp(stack{count}) ;
end