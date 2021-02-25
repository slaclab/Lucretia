function [stat,knob] = MakeMultiKnob( varargin )
%
% MAKEMULTIKNOB Create a combination of accelerator parameters (a "knob")
%    which can be varied in a synchronized and linear manner.
%
% [stat,knob] = MakeMultiKnob( comment, Parameter1, coeff1, Parameter2, 
%    coeff2, ...) creates a set of accelerator parameters which can be
%    varied together in order to produce a particular effect (waist shift,
%    dispersion variation, etc).  Arguments:
%
%    comment:  a text string which describes the knob.
%
%    Parameter1...ParameterN:  strings which name the parameter to be
%      varied, for example 'PS(1).SetPt', 'KLYSTRON(1).PhaseSetPt', etc.
%      Valid parameter names are PS().SetPt, KLYSTRON().PhaseSetPt,
%      KLYSTRON().AmplSetPt, GIRDER{}.MoverSetPt().
%
%    coeff1...coeffN:  real coefficient values for the parameters.
%
% Return argument knob is a Matlab data structure which contains all the
%    necessary information for the knob.  Return argument stat is a
%    Lucretia status cell array (type help LucretiaStatus for more
%    information).  The first cell, stat{1}==1 if the knob was successfully
%    generated; stat{1} == 0 if the arguments are not well defined or if an
%    invalid parameter string is detected.
%
%  See also:  SetMultiKnob, IncrementMultiKnob, RestoreMultiKnob.
%

%==========================================================================

% Begin with argument checking and unpacking:

  knob = [] ;
  xfercell = varargin ;
  [stat,parstrings,units,coeffs,comment] = MKBCheckUnpackPars( xfercell ) ;
  if (stat{1}==0)
    return ;
  end
  
% looks like everything is okay

  knob.Comment = comment ;
  knob.Value = 0 ;
  for count = 1:length(parstrings)
    knob.Channel(count).Parameter = parstrings{count} ;
    knob.Channel(count).Unit = units(count) ;
    knob.Channel(count).Coefficient = coeffs(count) ;
  end
  
  return ;
  
end

%=========================================================================

function [stat,parstrings,units,coeffs,comment] = MKBCheckUnpackPars( xfercell )

global PS KLYSTRON GIRDER

% Check arguments and unpack them for MakeMultiKnob

  stat = InitializeMessageStack() ;
  parstrings = cell(0) ;
  coeffs = [] ;
  comment = [] ;
  units = [] ;
  
% are there an odd number of arguments?  If not, bail out

  if ( (mod(length(xfercell),2) == 0) | ...
       (length(xfercell) < 3)               )
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
        'Incorrect number of arguments in MakeMultiKnob') ;
    return ;
  end
  
% is the first argument a string (comment)?  If not, bail out

  if ( ~ischar(xfercell{1}) )
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'First argument in MakeMultiKnob not a string') ;
  end
  comment = xfercell{1} ;
  
% loop over remaining entries in xfercell

  nchannel = (length(xfercell)-1) / 2 ;
  for count = 1:nchannel
      
    parstrings{count} = xfercell{count*2} ;
    if (~ischar(parstrings{count}))
       stat{1} = 0 ;
       stat = AddMessageToStack(stat,...
           ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
       continue ;
    end
    lenpar = length(parstrings{count}) ;

% power supply    
    
      if ( strncmp(parstrings{count},'PS(',3) )
        for endparen = 5:lenpar
          if (strcmp(parstrings{count}(endparen),')') )
            break ;
          end
        end
        if (endparen == lenpar)
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        psno = str2num(parstrings{count}(4:endparen-1)) ;
        units(count) = psno ;
        if ( (~exist('PS')) )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ((psno > length(PS)) | (psno < 1) )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ( ~strcmp(parstrings{count}(endparen+1:lenpar),'.SetPt') )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        
% klystron

      elseif ( strncmp(parstrings{count},'KLYSTRON(',9) )
        for endparen = 11:lenpar
          if (strcmp(parstrings{count}(endparen),')') )
            break ;
          end
        end
        if (endparen == lenpar)
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        psno = str2num(parstrings{count}(10:endparen-1)) ;
        units(count) = psno ;
        if ( (~exist('KLYSTRON')) ) 
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ( (psno > length(KLYSTRON)) | (psno < 1) )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ( ~strcmp(parstrings{count}(endparen+1:lenpar),'.AmplSetPt') & ...
             ~strcmp(parstrings{count}(endparen+1:lenpar),'.PhaseSetPt')   )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        
% girder mover

      elseif ( strncmp(parstrings{count},'GIRDER{',7) )
        for endparen = 9:lenpar
          if (strcmp(parstrings{count}(endparen),'}') )
            break ;
          end
        end
        if (endparen == lenpar)
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        psno = str2num(parstrings{count}(8:endparen-1)) ;
        units(count) = psno ;
        if ( (~exist('GIRDER')) ) 
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ( (psno > length(GIRDER)) | (psno < 1) )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ( ~isfield(GIRDER{psno},'MoverSetPt') )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        
% things are slightly different here because we need to check the
% mover DOF is valid

        if ( ~strncmp(parstrings{count}(endparen+1:endparen+11),...
              '.MoverSetPt',11) )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        dofptr = endparen+13 ;
        dof = str2num(parstrings{count}(dofptr)) ;
        if (isempty(dof))
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end
        if ( dof > length(GIRDER{psno}.Mover) )
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
        end            
        
% if it's not a GIRDER, a KLYSTRON, or a PS, it's an error

      else
          stat{1} = 0 ;
          stat = AddMessageToStack(stat,...
              ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
          continue ;
      end            

% now get the coefficient

%    coef = str2num(xfercell{count*2+1}) ;
    coef = xfercell{count*2+1} ;
    if (isempty(coef))
      stat{1} = 0 ;
      stat = AddMessageToStack(stat,...
          ['Channel # ',num2str(count),' in MakeMultiKnob invalid']) ;
      continue ;
    end            

    coeffs(count) = coef ;
    
  end
  
end  