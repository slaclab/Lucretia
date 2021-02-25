function [stat,varargout] = SetElementSlices( istart, iend )

% SETELEMENTSLICES Find the longitudinal "slices" of a single element
%
%   stat = SetElementSlices( istart, iend ) identifies all the instances of
%      elements which are "sliced" longitudinally (ie, more than one
%      instance in BEAMLINE represents the same physical element).  Each
%      element is then given a Slices field in BEAMLINE, which points at
%      all of the slices of that element.  Slices are identified by being
%      elements of the same class which have zero drift space between
%      them.  Return argument stat is a cell array, with stat{1} == 1 if no
%      errors occurred and stat{1} == 0 if errors occurred.
%
% [stat,slices] = SetElementSlices( istart, iend ) returns all of the
%    identified slice element lists in a cell array as well as a status.
%
% Version date:  09-Mar-2006.

% MOD:
%      09-mar-2006, PT:
%         support for TCAVs and solenoids.

%==========================================================================

  stat = InitializeMessageStack( ) ;
  slices = {} ; 
  ThisSlice = [] ;
  nslices = 0 ;
  global BEAMLINE ;
  
  if ( (istart < 0) | (iend > length(BEAMLINE)) )
      stat{1} = 0 ;
      stat = AddMessageToStack(stat,...
          'Error in range for SetElementSlices') ;
      return ;
  end
  
% set up a few cell arrays to make life simpler

  ValidClass = cell(10) ;
  ValidClass{1}  = 'QUAD' ;
  ValidClass{2}  = 'SEXT' ;
  ValidClass{3}  = 'OCTU' ;
  ValidClass{4}  = 'MULT' ;
  ValidClass{5}  = 'SBEN' ;
  ValidClass{6}  = 'LCAV' ;
  ValidClass{7}  = 'XCOR' ;
  ValidClass{8}  = 'YCOR' ;
  ValidClass{9}  = 'TCAV' ;
  ValidClass{10} = 'SOLENOID' ;
  
  AllOfClass = [] ;

% loop over classes and find all instances of each class  
  
  for ClassCount = 1:length(ValidClass)
      
      AllOfClass = findcells(BEAMLINE,'Class',...
                             ValidClass{ClassCount}) ;
      if (length(AllOfClass) == 0)
          continue ;
      end
      AOC_ptr = 0 ;
      ThisSlice = [] ;
      OldS = 0 ; 
      SliceOpen = 0 ;
      
% loop over members of the list
      
      while (AOC_ptr < length(AllOfClass))
          
          AOC_ptr = AOC_ptr + 1 ;
          elemno = AllOfClass(AOC_ptr) ;
          NewS = BEAMLINE{elemno}.S ; 
          
% keep looping until we find an element which is past the start point        
          
          if (elemno < istart)
              continue ;
          end

% if we have a ThisSlice open and this element should go into it, then 
% put it in

          if ( (~isempty(ThisSlice)) & (NewS == OldS) & (elemno <= iend) )
              ThisSlice = [ThisSlice elemno] ;
          else
              SliceOpen = 1 ;
          end

% If the slice is closed, or if we are on the last element in the valid
% range, file the slice if it is valid.  Note that a slice with only one
% element in it is not valid!

          if ( (SliceOpen ==1) | (elemno <=iend) )
              
            if (length(ThisSlice)>1)
              nslices = nslices + 1 ;
              slices{nslices} = ThisSlice ;
            end
            ThisSlice = [] ;
            
          end
          
% if this element is in range, update the OldS value

          if (elemno <= iend)
            OldS = BEAMLINE{elemno}.S + BEAMLINE{elemno}.L ;
            
% if in addition to being in range ThisRange is currently blank, then 
% this element must be the (potential) start of a new slice.  Put it into
% ThisSlice now.

            if (isempty(ThisSlice))
              ThisSlice = elemno ;
              SliceOpen = 0 ;
            end
            
          end
          
      end % while in range loop
      
  end % for-loop over eligible classes
  
% at this point we have accumulated a cell array full of slice data.  We
% can now commit the slices to the BEAMLINE elements

  for count = 1:nslices
      for count2 = 1:length(slices{count}) ;          
          BEAMLINE{slices{count}(count2)}.Slices = slices{count} ;          
      end      
  end

% finally, if the user wants the slice data back, give it to them

  if (nargout == 2)
      varargout{1} = slices ;
  end
  
  