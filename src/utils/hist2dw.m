%function [mHist wHist] = hist2dw ([vY, vX, vW], vYEdge, vXEdge)
%2 Dimensional Histogram
%Counts number of points in the bins defined by vYEdge, vXEdge.
% And weights by the vW vector
% mHist is standard 2d histogram- wHist is weighted by vW
%size(vX) == size(vY) == size(vW) == [n,1]
%size(mHist) & size(wHist) == [length(vYEdge) -1, length(vXEdge) -1]
%
%EXAMPLE
%   mYX = rand(100,2);
%   vXEdge = linspace(0,1,10);
%   vYEdge = linspace(0,1,20);
%   mHist2d = hist2d(mYX,vYEdge,vXEdge);
%
%   nXBins = length(vXEdge);
%   nYBins = length(vYEdge);
%   vXLabel = 0.5*(vXEdge(1:(nXBins-1))+vXEdge(2:nXBins));
%   vYLabel = 0.5*(vYEdge(1:(nYBins-1))+vYEdge(2:nYBins));
%   pcolor(vXLabel, vYLabel,mHist2d); colorbar
function [mHist wHist] = hist2dw (mX, vYEdge, vXEdge)
[nRow, nCol] = size(mX);
if nCol < 3
    error ('mX has less than three columns!')
elseif length(mX(:,1))~=length(mX(:,2)) || length(mX(:,1))~=length(mX(:,3)) || length(mX(:,2))~=length(mX(:,3))
    error ('vY, vX and vW must all be the same lengths')
end

nRow = length (vYEdge)-1;
nCol = length (vXEdge)-1;

vRow = mX(:,1);
vCol = mX(:,2);
vW = mX(:,3);

mHist = zeros(nRow,nCol);

for iRow = 1:nRow
    rRowLB = vYEdge(iRow);
    rRowUB = vYEdge(iRow+1);
    
    [mIdxRow] = find (vRow > rRowLB & vRow <= rRowUB);
    vColFound = vCol(mIdxRow);
    vWFound = vW(mIdxRow);
    
    if (~isempty(vColFound))
        
        
        [vFound, vFound_bin] = histc (vColFound, vXEdge);
        
        nFound = (length(vFound)-1);
        wghtVec = zeros(nFound,1);
        for f=1:nFound
          wghtVec(f)=sum(vWFound(vFound_bin==f));
        end
        
        if (nFound ~= nCol)
            [nFound nCol]
            error ('hist2d error: Size Error')
        end
        
        [nRowFound, nColFound] = size (vFound);
        
        nRowFound = nRowFound - 1;
        nColFound = nColFound - 1;
        
        if nRowFound == nCol
            mHist(iRow, :)= vFound(1:nFound)';
        elseif nColFound == nCol
            mHist(iRow, :)= vFound(1:nFound);
        else
            error ('hist2d error: Size Error')
        end
        wHist(iRow, :) = wghtVec;
    end
    
end


