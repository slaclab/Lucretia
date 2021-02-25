function stat = AssignToGirder( elemlist, gnum, is_long )

% ASSIGNTOGIRDER Assign elements to a girder
%
%   stat = AssignToGirder( element_list, G_number, is_long ) assigns the
%      BEAMLINE elements in the element_list to the girder indexed by
%      G_number.  The elements are added to any existing ones which are
%      already assigned to the girder.  If G_number is greater than the
%      present length of the GIRDER array, it will be extended; if the
%      GIRDER array is missing, it will be created; if G_number==0, all
%      existing girder assignments for members of element_list will be
%      cleared; if G_number==-1, the elements will be assigned to a new
%      girder just past the end of the current GIRDER array (ie, to
%      GIRDER{length(GIRDER}+1). Girders added by AssignToGirder will be
%      generated without movers.  Argument is_long indicates whether the
%      girder is intended to be a "long" girder which attaches to the
%      ground at both ends (is_long==1) or a "short" girder which attaches
%      to the ground at the longitudinal center (is_long == 0).  Return
%      argument stat is a Lucretia status and message cell array (type help
%      LucretiaStatus for more information).  In the event of error, no
%      assignments will be performed.
%
% Returned status values:  +1 for successful completion, 0 if an element in
% the element list is already assigned to another girder, -1 if the element
% list had no valid girderizable elements (ie all drifts and markers).
%
% Version date:  06-April-2007.

% MOD:
%       PT, 06-apr-2007:
%          add G_Number == -1 feature to tell routine to create a new
%          girder just past the end of the current array, and put the
%          selected elements onto it.

%==========================================================================

% is GIRDER around?  If not, create it

  stat = InitializeMessageStack( ) ;
  global BEAMLINE ;
  global GIRDER ;
  
% make sure that the new devices are not yet assigned to girders
% and are valid for assignment

  n_valid = 0 ;
  elemlist = elemlist(:) ;
  oldgird = [] ;
  for count = 1:length(elemlist)
      elemno = elemlist(count) ;
      if (isfield(BEAMLINE{elemno},'Girder'))
          n_valid = n_valid + 1 ;
          if ( (~isempty(BEAMLINE{elemno}.Girder)) & ...
                    (BEAMLINE{elemno}.Girder ~= 0) & ...
                    (BEAMLINE{elemno}.Girder ~= gnum) & ...
                    (gnum ~=0)                              )
              stat{1} = 0 ;
              stat = AddMessageToStack(stat,...
              ['Element # ',num2str(elemno), ...
              ' already has conflicting Girder assignment in AssignToGirder']) ;
          end
          if (gnum==0)
            if (isempty(BEAMLINE{elemno}.Girder))
              oldgird = [oldgird 0] ;
            else
              oldgird = [oldgird BEAMLINE{elemno}.Girder] ;
            end
          end
      else
          if (gnum==0)
            oldgird = [oldgird 0] ;
          end
      end
  end
  
% if we got this far, abort execution before we do anything we're going to
% regret

  if (stat{1} == 0)
      return ;
  end
  if (n_valid == 0)
    stat{1} = -1 ;
    stat = AddMessageToStack(stat,...
        'No valid elements in elemlist in AssignToGirder') ;
    return ;
  end
  
% if the girder number is zero, clear assignments now and return

  if (gnum==0)
    for count = 1:length(elemlist)
      elemno = elemlist(count) ; girdno = oldgird(count) ;
      if (girdno == 0)
        continue ;
      end      
      BEAMLINE{elemno}.Girder = 0 ;
      elist = find(GIRDER{girdno}.Element==elemno) ;
      GIRDER{girdno}.Element(elist) = [] ;
    end
    return ;
  end

% if we've gotten here, then we really have something to add to the girder
% list  
  
  if (isempty(GIRDER))
    GIRDER = cell(0) ;
  end
  
% if the user wants a new girder just past the end of the existing girder
% list, set that up now

  if (gnum == -1)
      gnum = length(GIRDER) + 1 ;
  end
  
% if the GIRDER array doesn't go out far enough, extend it

  if (length(GIRDER) < gnum)
    [statcall,GIRDER{gnum}] = GirderStruc( ) ;
  end
  
% note that although a DRIF or a MARK can't actually be put on a GIRDER,
% they do have a role to play in finding the S positions of the girder

  elist = sort([GIRDER{gnum}.Element elemlist]) ;
  e1 = elist(1) ; e2 = elist(length(elist)) ;
  gS = [BEAMLINE{e1}.S BEAMLINE{e2}.S] ;
  if (isfield(BEAMLINE{e2},'L'))
     gS(2) = gS(2) + BEAMLINE{e2}.L ;
  end

  if (is_long)
    GIRDER{gnum}.S = gS ;
  else
    GIRDER{gnum}.S = mean(gS) ;
  end
  
% now perform the assignments:  first at the BEAMLINE...

  elist_nodrifmark = [] ;
  for count = 1:length(elemlist)
      elemno = elemlist(count) ;
      if ( (~strcmp(BEAMLINE{elemno}.Class,'DRIF')) & ...
           (~strcmp(BEAMLINE{elemno}.Class,'MARK'))       )
        BEAMLINE{elemno}.Girder = gnum ;
        elist_nodrifmark = [elist_nodrifmark elemno] ;
      end
  end
  
% ...now at the girder

  elist = [GIRDER{gnum}.Element elist_nodrifmark] ;
  elist = sort(elist) ;
  
% eliminate duplicates

  elast = elist(1) ;
  elist2 = elast ;
  for count = 2:length(elist)
      if (elist(count) ~= elast)
          elast = elist(count) ;
          elist2 = [elist2 elast] ;
      end
  end
    
  GIRDER{gnum}.Element = elist2 ;
      