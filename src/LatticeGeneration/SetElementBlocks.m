function [stat,varargout] = SetElementBlocks( istart, iend )

% SETELEMENTBLOCKS Find the longitudinal alignment "blocks" of an
%    accelerator.
%
%   stat = SetElementBlocks( istart, iend ) identifies all the instances of
%      elements which are contiguous (ie, not separated by finite drifts),
%      which should therefore be (potentially) uniformly misaligned with
%      one another.  Each element is given a Block field in BEAMLINE, which
%      points at the start and end elements in that block.  Return argument
%      stat is a Lucretia status cell array (type help LucretiaStatus for
%      more information).
%
% [stat,blocks] = SetElementBlocks( istart, iend ) returns all of the
%    identified block lists in a cell array as well as a status.
%
% Return status:  +1 if successful, 0 if invalid arguments supplied.
%
% Version date:  11-may-2006

% Mod:
%       PT, 11-may-2006:
%           allow MARKERs to be members of Blocks.

%==========================================================================

  global BEAMLINE 
  stat = InitializeMessageStack( ) ;
  blocks = {} ; 
  
% check start and stop indicies

  if ( (istart<1) | (istart>length(BEAMLINE)) | ...
       (iend<1)   | (iend>length(BEAMLINE))   | ...
       (iend < istart)                              )
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'Invalid start/end indices in SetElementBlocks') ;
    return ;
  end
  
% move the start and endpoints of the search so that they point at the
% first and last non-drift elements in the region of interest

  while ( (strcmp(BEAMLINE{istart}.Class,'DRIF'))   & ...
          (istart < iend)                                  )
      istart = istart + 1 ;
  end
  while ( (strcmp(BEAMLINE{iend}.Class,'DRIF'))   & ...
          (iend > istart)                                  )
      iend = iend - 1 ;
  end
  
% since all blocks are started and ended by drifts, find all of the drifts
% in the desired range  

  dlist = findcells(BEAMLINE,'Class','DRIF') ;
  dindx = find((dlist>istart)&(dlist<iend)) ;
  dlist = dlist(dindx) ;
  if (length(dlist)==0)
      return ;
  end
  
% loop over the total number of regions between drifts

  thisblock = [] ; 
  for count = 0:length(dlist)
      
% potential block termini

      if (count==0)
          subblock = [istart dlist(1)-1] ;
      elseif (count==length(dlist))
          subblock = [dlist(count)+1 iend] ;
      else
          subblock = [dlist(count)+1 dlist(count+1)-1] ;
      end
      
% Because we discarded the leading and trailing drifts in the region, if
% any, we know that when count==0 and when count==length(dlist) that
% subblock is pointing to non-drift elements.  For all other values of
% count, it's possible that subblock is pointing at 2 consecutive drifts.
% The indication that this has happened is that the first entry in subblock
% is > the last entry.  In this case blank out the subblock and loop.

      if (subblock(2)<subblock(1))
          subblock = [] ;
          continue ;
      end      
      
% otherwise add the subblock to the master block

      thisblock = [thisblock subblock] ;
      
% we are ready to consider adding this block to the block list if the drift
% at the end of the block has nonzero length, or if we are on the last pass
% through the loop (count == length(dlist)).  As it turns out it is simpler
% to test for the opposite case:  if we are not on the last iteration of
% the loop, AND the drift at the end of the block has zero length, loop.
      
      if (count < length(dlist))
        if (BEAMLINE{dlist(count+1)}.L == 0)
          continue ;
        end
      end
      
% we still may not want to commit it to the list, however!  A single
% element sandwiched between 2 nonzero length drifts does not constitute a
% block.  If this is the case with the current block, clear it and
% continue.  Otherwise commit it and continue.

      if (max(thisblock) > min(thisblock))
        blocks{length(blocks)+1} = [min(thisblock) max(thisblock)] ;
      end
      thisblock = [] ;
      
  end

% now put the block information into the elements, being careful not to put
% a block data entry into a drift or a marker
%
% On second thought (PT, 11-may-2006), putting a marker into a block is OK

  for count = 1:length(blocks)
      thisblock = blocks{count} ;
      for count2 = thisblock(1):thisblock(length(thisblock))
          if ( (~strcmp(BEAMLINE{count2}.Class,'DRIF'))  ...
                                                              )
             BEAMLINE{count2}.Block = thisblock ;
          end
      end
  end
%
  if (nargout == 2)
      varargout{1} = blocks ;
  end
%