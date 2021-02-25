function M = TMapStruc( Name, L )
%
% TMAPSTRUC Create a Lucretia transfer map element structure
%
% M = TMapStruc( Name, L ) creates a Lucretia transfer map element
%     structure with a given name and length.
%     The M.R response matrix is initialised as a 6x6 identity matix.
%     M.T, M.U, M.V and M.W fields and M.Tinds, M.Uinds, M.Vinds, M.Winds
%     fields are provided to make use of higher order maps.
%     It is up to the user to them set the elements of the transport matrices desired.
%     M.T etc must be stored as an [N x 1] vector of T element strengths.
%     A corresponding M.Tinds etc must also have the same [N x 1] vector length (type double).
%     These contain the higher order map indices (e.g. T(2,3,6)=236, W(1,3,3,4,4,6)=133446 ).
%==============================================================================================

if ~exist('Order','var')
  Order=1;
end
M.Name = Name ;
M.L = L ;
M.S = 0 ; M.P = 0 ;
M.Class = 'TMAP' ;
M.Offset = zeros(6,1) ;
M.R = eye(6) ;
M.T = []; M.Tinds = [];
M.U = []; M.Uinds = [];
M.V = []; M.Vinds = [];
M.W = []; M.Winds = [];
