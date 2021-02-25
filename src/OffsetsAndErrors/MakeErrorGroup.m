function [stat,Group] = MakeErrorGroup( tablecell, range, errname, ...
                        clusters, comment )
% 
% MAKEERRORGROUP Generate a data structure for the application of
% Gaussian-distributed errors and misalignments to a set of elements.
%
% [stat,Group] = MakeErrorGroup( Table, Range, ErrName, Clusters, 
%    [Comment | DL] ) finds all elements of a given Lucretia data table which fit
%    certain categories (position within the table, element class, etc) and
%    returns a list of those elements, preserving information on alignment
%    blocks or element slices if requested.  This structure can be used by
%    functions such as ErrorGroupGaussErrors to apply errors and
%    misalignments to a set of elements, klystrons, power supplies, or
%    girders. Return argument stat is a Lucretia status cell array (type
%    help LucretiaStatus for more information), return argument Group
%    contains the error group. Calling arguments are defined as follows:
% 
%      Table:  either a string or a cell array of strings.  The first
%         string must be the name of the table of interest ('BEAMLINE',
%         'KLYSTRON', 'PS', 'GIRDER').  If the first string is BEAMLINE,
%         Table{2} and Table{3} are the element class and element name,
%         respectively, which are to be given errors.  If Table{2} and/or
%         Table{3} are themselves cell arrays of strings, then all of the
%         classes in Table{2} and/or all of the named elements in Table{3}
%         will be put into the error group.  Names in Table{3} can also be
%         Matlab regular expressions, see help for REGEXP.
%      Range:  1 x 2 vector with the initial and final table entries which
%         are to receive errors.
%      ErrName:  string which names the error which is to be set ('dB',
%         'dV', 'dPhase', 'dAmpl', 'dScale', 'Offset', 'ElecOffset',
%         'BPMOffset').  For gradient bends with 2 dB values, ErrName can
%         also be 'dB(1)' or 'dB(2)'.  Note that use of 'dB' will ignore
%         gradient bends with 2 dB values, and that use of dB(1) or dB(2)
%         will ignore all elements with only 1 dB value (including gradient
%         bends).
%      Clusters: scalar, if clusters == 2 then all slices (for errors) or
%         block members (for misalignments) will receive a common error set
%         (with appropriate adjustments to the misalignments for the fact
%         that block members with different S positions need different
%         position offsets if they receive angle offsets).  If clusters ==
%         0, each entry in BEAMLINE gets a unique error.  If clusters == 1,
%         the result is similar to cluster == 2, except that only block
%         members with the name and class requested will be misaligned and
%         all other block members will not be moved.
%     [Comment | DL]: text string which is attached to the group for user's
%         reference and information. Or a distributedLucretia object if
%         want to apply errors in distributed context.
%
% Return status:  +1 if executed successfully, 0 if an invalid combination
%    of arguments is supplied, -1 if there are no devices which meet the
%    user's criteria.
%
% See Also:  ErrorGroupGaussErrors, SetGaussianErrors, regexp, distributedLucretia
%
% Version date:  25-September-2007.

% MOD:
%      14-Feb-2012, GW:
%         Add distributedLucretia support
%      25-sep-2007, GW:
%         Changed rand function to use randnt instead of randn to enable
%         truncation of random number distribution (additional function
%         randnt placed in same directory as this file)
%      13-jun-2006, PT:
%         bugfix -- scalar errors had wrong ErrAccum dimension!
%      24-Feb-2006, PT:
%         support for error groups with multiple classes and/or names.
%      18-oct-2005, PT:
%         support for BPM scale factor errors.
%      29-sep-2005, PT:
%         support for magnets with multiple dB fields.
%      27-sep-2005, PT:
%         Add sextupole, octupole, multipole field errors to "dB" option.

%==========================================================================


  global BEAMLINE KLYSTRON PS GIRDER
  Group = [] ; 
  stat = InitializeMessageStack( ) ;

% before anything else, verify that the arguments are correct and
% consistent with one another

  [statarg,tabname,classname,eltname] = MEGVerifyArgs( tablecell, range, ...
      errname, clusters ) ;
  stat = AddStackToStack( stat, statarg ) ;
  if (statarg{1} ~= 1)
    stat{1} = 0 ;
    return ;
  end
  
% make a cluster list for the application of the errors

  if (~strcmp(tabname,'BEAMLINE'))
    ClusterList = SimpleClusterList(range) ;
  else
    ClusterList = BeamlineClusterList( range, classname, eltname, ...
                                       errname, clusters ) ;
  end
  if (isempty(ClusterList))
      stat{1} = -1 ;
      stat = AddMessageToStack(stat,...
          'No appropriate elements or devices found') ;
      return ;
  end
    
% generate appropriate command strings for rolling the errors and for
% applying them  

  [dimension,gener,adjust,apply,accum,statistics] = ...
      MakeMEGCommandStrings( tabname, errname ) ;
  
% assemble the group

  Group.comment = comment ;
  Group.table = tabname ;
  Group.class = classname ;
  Group.name = eltname ;
  Group.error = errname ;
  Group.dimension = dimension ;
  Group.gener = gener ;
  Group.adjust = adjust ;
  Group.apply = apply ;
  Group.accum = accum ;
  Group.statistics = statistics ;
  Group.ClusterList = ClusterList ;
  
%==========================================================================
%==========================================================================
%==========================================================================

% subfunction to unpack arguments and verify correctness

    function [statarg,tabname,classname,eltname] = MEGVerifyArgs( ...
      table, range, errstring, clusters )
  
  global BEAMLINE PS GIRDER KLYSTRON
 
  statarg = InitializeMessageStack( ) ;

% start by unpacking the names of the table, class, and element

  classname = [] ; eltname = [] ;
  if (ischar(table))
    tabname = table ;
  elseif (iscell(table))
      if (ischar(table{1}))
        tabname = table{1} ;
      else
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'First argument of MakeErrorGroup, cell 1, not a string') ;
        return ;
      end
      if (length(table) > 1)

% start by making sure that table{2} is either a string or a cell array of
% strings

        charok = ischar(table{2}) ;
        if (charok == 0)
         if (iscell(table{2}))
          for itab2 = 1:length(table{2})
           if (~ischar(table{2}{itab2}))
            charok = 0 ;
            break ;
           end
           charok = 1 ;
          end
         end
        end
        if (charok == 0)
          statarg{1} = 0 ;
          statarg = AddMessageToStack(statarg, ...
            'First argument of MakeErrorGroup, cell 2, not a string or cell array of strings') ;
          return ;
        end
        classname = upper(table{2}) ;
        if (~iscell(classname))
         classname = {classname} ;
        end
      end
      
      if (length(table) > 2)
        if (ischar(table{3}))
          eltname = upper(table{3}) ;
        else
          statarg{1} = 0 ;
          statarg = AddMessageToStack(statarg, ...
            'First argument of MakeErrorGroup, cell 3, not a string') ;
          return ;
        end
        if (~iscell(eltname))
          eltname = {eltname} ;
        end
      end
  else
      statarg{1} = 0 ;
      statarg = AddMessageToStack(statarg, ...
          'First argument of MakeErrorGroup not a string or cell array') ;
      return ;
  end
    
% check that the clusters flag is 2, 1 or 0

  if ( (~isscalar(clusters)) | (~isnumeric(clusters)) )
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
        'Argument 4 in MakeErrorGroup must be numeric scalar') ;
    return ;
  end
  if ( (clusters~=2) & (clusters~=1) & (clusters~=0) )
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
        'Argument 4 in MakeErrorGroup must be 2, 1 or 0') ;
    return ;
  end

% make sure the table name is valid

  if ( (~strcmp(tabname,'BEAMLINE')) & (~strcmp(tabname,'KLYSTRON')) & ...
       (~strcmp(tabname,'GIRDER')  ) & (~strcmp(tabname,'PS')      )       )
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
      'Invalid table name in MakeErrorGroup') ;
    return ;
  end
  
% check that the range argument is well-defined

  if ( (~isnumeric(range)) | (sum(size(range)==[1 2])~=2) )
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
      'Argument 2 in MakeErrorGroup must be 1 x 2 numeric vector') ;
    return ;
  end
  if (range(2) < range(1)) 
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
      'Argument 2 values in MakeErrorGroup improperly ordered') ;
    return ;
  end
  if (sum(floor(range)==range)~=2)
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
      'Argument 2 values in MakeErrorGroup not integers') ;
    return ;
  end
  if (range(1)<1)
    statarg{1} = 0 ;
    statarg = AddMessageToStack(statarg, ...
      'Argument 2 first value in MakeErrorGroup < 1') ;
    return ;
  end

% now we need to do table-specific verification of the table name, the
% error name, the second range entry, and the class name (if any).

  switch(tabname)
      
    case 'PS'
      if ( (~isempty(classname)) | (~isempty(eltname)) )
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 1 in MakeErrorGroup:  invalid combination') ;
        return ;
      end
      if (~strcmp(errstring,'dAmpl'))
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Arguments 1 and 3 in MakeErrorGroup:  invalid combination') ;
        return ;
      end
      if (range(2) > length(PS))
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 2 value 2 in MakeErrorGroup exceeds PS length') ;
        return ;
      end

%      
      
    case 'GIRDER'
      if ( (~isempty(classname)) | (~isempty(eltname)) )
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 1 in MakeErrorGroup:  invalid combination') ;
        return ;
      end
      if (~strcmp(errstring,'Offset'))
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Arguments 1 and 3 in MakeErrorGroup:  invalid combination') ;
        return ;
      end
      if (range(2) > length(GIRDER))
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 2 value 2 in MakeErrorGroup exceeds GIRDER length') ;
        return ;
      end

%      
      
    case 'KLYSTRON'
      if ( (~isempty(classname)) | (~isempty(eltname)) )
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 1 in MakeErrorGroup:  invalid combination') ;
        return ;
      end
      if ( (~strcmp(errstring,'dAmpl')) & (~strcmp(errstring,'dPhase')) )
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Arguments 1 and 3 in MakeErrorGroup:  invalid combination') ;
        return ;
      end
      if (range(2) > length(KLYSTRON))
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 2 value 2 in MakeErrorGroup exceeds KLYSTRON length') ;
        return ;
      end
  
% beamlines are a bit more complicated

    case 'BEAMLINE'
      if (range(2) > length(BEAMLINE))
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
          'Argument 2 value 2 in MakeErrorGroup exceeds BEAMLINE length') ;
        return ;
      end
      switch errstring
          
       case 'dB'
%        if ( (~isempty(classname)) & ...
%             (~strcmp(classname,'QUAD')) & ...
%             (~strcmp(classname,'SEXT')) & ...
%             (~strcmp(classname,'OCTU')) & ...
%             (~strcmp(classname,'MULT')) & ...             
%             (~strcmp(classname,'SBEN')) & ...
%             (~strcmp(classname,'XCOR')) & ...
%             (~strcmp(classname,'YCOR'))       )
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%         'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%         return ;
%        end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'QUAD') ; ...
                        strcmp(classname,'SEXT') ; ...
                        strcmp(classname,'OCTU') ; ...
                        strcmp(classname,'MULT') ; ...
                        strcmp(classname,'SBEN') ; ...
                        strcmp(classname,'XCOR') ; ...
                        strcmp(classname,'YCOR')] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
       case { 'dB(1)','dB(2)' } 
%        if ( (~isempty(classname)) & (~strcmp(classname,'SBEN')) )
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%          'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%         return ;
%        end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'SBEN')] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
       case 'dV'
%        if ( (~isempty(classname)) & ...
%             (~strcmp(classname,'LCAV'))       )
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%          'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%          return ;
%       end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'LCAV') ; ...
                        strcmp(classname,'TCAV') ] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
       case 'dPhase'
 %       if ( (~isempty(classname)) & ...
 %            (~strcmp(classname,'LCAV'))       )
 %         statarg{1} = 0 ;
 %         statarg = AddMessageToStack(statarg, ...
 %         'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
 %         return ;
 %       end
         if (~isempty(classname))
          CheckClass = [strcmp(classname,'LCAV') ; ...
                        strcmp(classname,'TCAV') ] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
       
       case 'BPMOffset'
%        if ( (~isempty(classname)) & ...
%             (~strcmp(classname,'LCAV'))       )
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%          'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%          return ;
%        end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'LCAV') ; ...
                        strcmp(classname,'TCAV') ] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
       case 'ElecOffset'
%        if ( (~isempty(classname)) & ...
%             (~strcmp(classname,'HMON')) & ...      
%             (~strcmp(classname,'VMON')) & ...      
%             (~strcmp(classname,'MONI'))       )
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%          'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%          return ;
%        end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'HMON') ; ...
                        strcmp(classname,'VMON') ; ...
                        strcmp(classname,'MONI')] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
       case 'dScale'
%        if ( (~isempty(classname)) & ...
%             (~strcmp(classname,'HMON')) & ...      
%             (~strcmp(classname,'VMON')) & ...      
%             (~strcmp(classname,'MONI'))       )
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%          'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%          return ;
%        end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'HMON') ; ...
                        strcmp(classname,'VMON') ; ...
                        strcmp(classname,'MONI')] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
       case 'Offset'
%        if ( (~isempty(classname)) & ...
%             ( (strcmp(classname,'DRIF')) | (strcmp(classname,'MARK')) ) )   
%          statarg{1} = 0 ;
%          statarg = AddMessageToStack(statarg, ...
%         'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
%          return ;
%        end
        if (~isempty(classname))
          CheckClass = [strcmp(classname,'SBEN') ; ...
                        strcmp(classname,'QUAD') ; ...
                        strcmp(classname,'SEXT') ; ...
                        strcmp(classname,'OCTU') ; ...
                        strcmp(classname,'MULT') ; ...
                        strcmp(classname,'XCOR') ; ...
                        strcmp(classname,'YCOR') ; ...
                        strcmp(classname,'LCAV') ; ...
                        strcmp(classname,'TCAV') ; ...
                        strcmp(classname,'HMON') ; ...
                        strcmp(classname,'VMON') ; ...
                        strcmp(classname,'MONI') ; ...
                        strcmp(classname,'INST') ; ...
                        strcmp(classname,'PROF') ; ...
                        strcmp(classname,'WIRE') ; ...
                        strcmp(classname,'BLMO') ; ...
                        strcmp(classname,'SLMO') ; ...
                        strcmp(classname,'IMON') ; ...
                        strcmp(classname,'COLL') ] ; 
          if (sum(sum(CheckClass)) ~= length(classname))
            statarg{1} = 0 ;
            statarg = AddMessageToStack(statarg, ...
           'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
           return ;
          end
        end
        
      otherwise
           
        statarg{1} = 0 ;
        statarg = AddMessageToStack(statarg, ...
        'Arguments 1 and 3 in MakeErrorGroup: invalid combination') ;
        return ;
        
      end % switch on errstring
      
%

  end % switch on tabname

% if we got here then the arguments are valid and internally consistent,
% and we can return.

  return ;
  
%  
%==========================================================================
%==========================================================================
%==========================================================================
 
% generate a simple list in the ClusterList format of the entries 
% specified by the range.  Since the simple list has no clustering, it
% is guaranteed that each list has only 1 entry and each S vector is zero.

function List = SimpleClusterList( range )
    
  for count1 = range(1):range(2)   
    count = count1+1-range(1) ;
    List(count).index = count1 ;
    List(count).dS = 0 ;
  end
  
%==========================================================================
%==========================================================================
%==========================================================================
 
function List = BeamlineClusterList( range, classname, eltname, ...
                                     errname, clusters ) 
                                 
  global BEAMLINE   
  List = [] ;
                                       
% make a shortcut showing what kinds of errors we are dealing with

  if (strcmp(errname,'dB(1)')) 
    errname = 'dB' ;
    errpos = 1 ;
  elseif (strcmp(errname,'dB(2)'))
    errname = 'dB' ;
    errpos = 2 ;
  elseif (strcmp(errname,'dB'))
    errname = 'dB' ;
    errpos = 0 ;
  end
  if (strcmp(errname,'Offset'))
      ErrorClass = 1 ;
  elseif ( (strcmp(errname,'dB')) | (strcmp(errname,'dV')) | ...
               (strcmp(errname,'dPhase')) )
      ErrorClass = 2 ;
  else
      ErrorClass = 3 ;
  end

% generate a cluster list which properly takes into account the range
% of beamline elements, the class and name screens, and any slice/block
% action

  ClusterPointer = 0 ; S0 = 0 ;
  for count = range(1):range(2)
       
      ThisList = [] ; SList = [] ; SAll = [] ; ThisAll = [] ;
      
% a drift can't be the first element in a block, and it can't be an element of 
% a slice list at all.  So if this is a drift, skip to next element.

      if ( strcmp(BEAMLINE{count}.Class,'DRIF') )
        continue ;
      end
      
% make a shortcut to the Slices or Block list, if appropriate

      if ( ErrorClass == 1 )
        if ( (clusters>0) & (isfield(BEAMLINE{count},'Block')) )
          clist = BEAMLINE{count}.Block ;
        else
          clist = [count count] ;
        end
      elseif ( ErrorClass == 2 )
        if ( (clusters>0) & (isfield(BEAMLINE{count},'Slices')) )
          clist = BEAMLINE{count}.Slices ;
        else
          clist = count ;
        end
      else % BPMOffset or ElecOffset
        clist = count ;
      end
      
% if the first entry in the clist isn't this element, loop

      if (clist(1) ~= count) 
        continue ;
      end
      
% if this is an offset, expand the clist

      if (ErrorClass == 1)
        if (isfield(BEAMLINE{clist(2)},'L'))
          L = BEAMLINE{clist(2)}.L ;
        else
          L = 0 ;
        end
        S0 = (BEAMLINE{clist(2)}.S + L + BEAMLINE{clist(1)}.S) / 2 ;
        clist = linspace(clist(1),clist(2),clist(2)-clist(1) + 1) ;
      end
      
% loop over elements in clist

      for count2 = 1:length(clist)
        elemptr = clist(count2) ;
        
% see whether the element passes the name and class screens, and whether it 
% has the desired field

        if (~isfield(BEAMLINE{elemptr},errname))
          continue ;
        end
        
% if this is a dB-type error, and the user's request for its shape is not
% consistent with the shape of the field in the element, continue

        if (strcmp(errname,'dB'))
           if ( (length(BEAMLINE{elemptr}.dB) == 1) & ...
                (errpos ~= 0)                            )
             continue ;
           end
           if ( (length(BEAMLINE{elemptr}.dB) > 1) & ...
                ( (errpos == 0) | ...
                  (errpos > length(BEAMLINE{elemptr}.dB)) ...
                 )                                             )
             continue ;
           end
        end
        
        if ( (strcmp(errname,'BPMOffset')) & ...
             (BEAMLINE{elemptr}.NBPM == 0)       )
          continue ;
        end
        Sctr = BEAMLINE{elemptr}.S ;
        if (isfield(BEAMLINE{elemptr},'L'))
          Sctr = Sctr + BEAMLINE{elemptr}.L/2 ;
        end
        ThisAll = [ThisAll elemptr] ;
        SAll = [SAll Sctr] ;
%        if ( (~isempty(classname)) & ...
%            (~strcmp(BEAMLINE{elemptr}.Class,classname)) )
%          continue ;
%        end
        if (~isempty(classname))
          ClassCheck = strcmp(BEAMLINE{elemptr}.Class,classname) ;
          if (sum(sum(ClassCheck)) == 0)
            continue ;
          end
        end
%        if ( (~isempty(eltname)) & ...
%            (~strcmp(BEAMLINE{elemptr}.Name,eltname)) )
%          continue ;
%        end
        if (~isempty(eltname))
%          EltCheck = strcmp(BEAMLINE{elemptr}.Name,eltname) ;
          EltCheck = regexp(BEAMLINE{elemptr}.Name,eltname) ;
          FoundAMatch = [] ;
          for count = 1:length(EltCheck) 
            FoundAMatch = [FoundAMatch EltCheck{count}] ;
          end
          if (isempty(FoundAMatch))
            continue ;
          end
        end
        
% if we made it through the screens (as they say in Star Trek), add this one to
% the current list.

        ThisList = [ThisList elemptr] ;
        SList = [SList Sctr] ;
        
      end % inner loop over cluster members
      
% if the list is not blank, then we found elements of interest.  Add them to the 
% metalist

      if (~isempty(ThisList))
        ClusterPointer = ClusterPointer + 1 ;
        if ( (ErrorClass==1) & (clusters==2) )
          ThisList = ThisAll ;
          SList = SAll ;
        end
        List(ClusterPointer).index = ThisList ;
        List(ClusterPointer).dS = SList - S0 ;
       end
      
  end % outer loop over elements

%==========================================================================
%==========================================================================
%==========================================================================

% subfunction to generate command strings for use in rolling and applying
% Gaussian errors
 
function [dimension,gener,adjust,apply,accum,statistics] = ...
          MakeMEGCommandStrings( tabname, errname ) ;
      
% the dimensioning string depends only on the type of error
  %#function randnt

  switch errname
      
      case 'Offset'
          dimension = 'ErrAccum = zeros(length(group.ClusterList),6);' ;
      case 'BPMOffset'
          dimension = 'ErrAccum = zeros(2,length(group.ClusterList));' ;
      case 'ElecOffset'
          dimension = 'ErrAccum = zeros(length(group.ClusterList),2);' ;
      otherwise
          dimension = 'ErrAccum = zeros(length(group.ClusterList),1);' ;
          
  end
  
% some of the string contents depend only on the type of error involved    
  if isempty(which('randnt')); error('Need randnt function in search path!'); end;
  switch errname
      
      case 'Offset'          
          gener = 'newerr = randnt(randTrunc,1,6) .* rmsval ;' ;
      case 'BPMOffset'
          gener = 'indx=group.ClusterList(ClusterCount).index;';
          gener = [gener,...
              'nb=BEAMLINE{indx}.NBPM ;newerr = randnt(randTrunc,2,nb);'] ;
          gener = [gener,...
           'newerr = [newerr(1,:)*rmsval(1);newerr(2,:)*rmsval(2)];'];
      case 'ElecOffset'
          gener = 'newerr = randnt(randTrunc,1,2) .* rmsval ;' ;
      otherwise
          gener = 'newerr = randnt(randTrunc,1) * rmsval ;' ;
          
  end
  
% the adjustment string depends on the error name and the table name
  
  switch errname
      
      case 'BPMOffset'
        adjust = 'm=[meanval(1)*ones(1,nb);meanval(2)*ones(1,nb)];' ;
        adjust = [adjust,...
          'k=[keepold(1)*ones(1,nb);keepold(2)*ones(1,nb)];'] ;
      case 'Offset'
          if (strcmp(tabname,'BEAMLINE'))
            adjust = 'thiserr=newerr;thismean=meanval;';
            adjust = [adjust,...
              'thiserr(1)=thiserr(1)+thiserr(2)*ds;'];
            adjust = [adjust,...
              'thiserr(3)=thiserr(3)+thiserr(4)*ds;'];
            adjust = [adjust,...
              'thismean(1)=thismean(1)+thismean(2)*ds;'];
            adjust = [adjust,...
              'thismean(3)=thismean(3)+thismean(4)*ds;'];
          else
            adjust = ['thiserr=newerr;thismean=meanval;'] ;
          end
      otherwise
          adjust = [] ;

  end
              
% the apply string depends on the error name and the table name

  switch tabname
      
      case 'BEAMLINE'
          appstr = 'BEAMLINE{' ; p = '}';
      case 'KLYSTRON'
          appstr = 'KLYSTRON(' ; p = ')';
      case 'GIRDER'
          appstr = 'GIRDER{' ; p = '}';
      case 'PS'
          appstr = 'PS(' ; p = ')';
          
  end
  
  apply = [appstr,'indx',p,'.',errname,'=',...
           appstr,'indx',p,'.',errname,'.*'     ] ;
  
  switch errname
      
      case 'BPMOffset'
          apply = [apply,'k+m+newerr;'];
      case 'Offset'
          apply = [apply,'keepold+thismean+thiserr;'] ;
       otherwise
          apply = [apply,'keepold+meanval+newerr;'] ;
         
  end
  
% the accum, orig, and statistics strings depend only on the error name

  switch errname
      
      case 'BPMOffset'
          accum = 'ErrAccum(:,ClusterCount) = newerr+meanval ;' ;
          statistics = 'ErrMean=mean(ErrAccum'');ErrStd=std(ErrAccum'');' ;
      otherwise
          accum = 'ErrAccum(ClusterCount,:) = newerr+meanval ;' ;
          statistics = 'ErrMean=mean(ErrAccum);ErrStd=std(ErrAccum);' ;
          
  end
%  

  
