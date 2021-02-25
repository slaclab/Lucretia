function stackout = AddStackToMasterStack( stack1, stack2, FunctionName )
%
% ADDSTACKTOMASTERSTACK Combine Lucretia status cell arrays to generate a
%    master cell array which summarizes others.
%
% stackout = AddMasterStackToStack( stack1, stack2, FunctionName ) adds
%    status cell array stack2 to status cell array stack1, but does so in
%    such a way that the resulting output, stackout, is a master status
%    cell array.  A master status cell array differs from a regular status
%    cell array in the following ways:
%
%    1.  The first cell in the array contains the values of all the status
%        cell arrays which the master is summarizing
%    2.  There is a message in the master array for every summarized cell
%        array.  If the cell array from a given function call has no
%        messages in it, then the message 'FunctionName:  OK' is added to
%        the master stack, where FunctionName is the text string 3rd
%        calling argument to AddStackToMasterStack.
%
% See also:  LucretiaStatus, AddMessageToStack, AddStackToStack.
%
% Version date:  13-Mar-2008.

% Modification History:
%
%   PT, 12-mar-2008:
%       do the right thing if stack2 is also a master stack!

  stackout = stack1 ;
  if (length(stackout) == 1)
      stackout{1} = stack2{1} ;
  else
      stackout{1} = [stackout{1} stack2{1}] ;
  end
  
  AllOK = (length(stack2{1}) == sum(stack2{1})) ;
  Stack2IsMaster = ( (AllOK & (length(stack2)>1)) | ...
                     (length(stack2{1})>1) ) ;
  
  stackout = AddStackToStack(stackout,stack2) ;
  if (AllOK)
      stackout = AddMessageToStack(stackout, ...
          [FunctionName,': OK']) ;
  end
  if (Stack2IsMaster)
      if (AllOK)
          stackout{1} = [stackout{1} 1] ;
      else
          stackout{1} = [stackout{1} 0] ;
      end
  end