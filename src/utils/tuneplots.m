function tuneplots(varargin)
% tuneplots(order, superperiodicity, [x y], [x0 x1 y0 y1])
% 
% === INPUTS ===
% order - order of the resonance lines to be drawn
% superperiodicity - the superperiodicity of the lattice
% [x y] - vector of horizontal and vertical tunes. If you want to plot 
% [x1 y1] and [x2 y2] then use [x1 x2 y1 y2]
% [x0 x1 y0 y1] - axes limits
%
% ex. tuneplots(3, 1, [5.1 4.3], [4.9 6.1 3.9 5.1]);
%
% If no arguments are passed, it runs with old parameters for the
% Australian Synchrotron storage ring lattice.

% Define some default values if no parameters are placed.
if nargin < 3
    x = [13.3];
    y = [5.2];
else
    x = varargin{3}(1:end/2);
    y = varargin{3}(end/2 + 1:end);
end
if nargin < 2
    N = 14;    % Superperiod
else
    N = varargin{2};
end
if nargin < 1
    order = 3;
else
    order = varargin{1};
end
if nargin < 4    
    x0 = 0;
    x1 = N;
    y0 = 0;
    y1 = N;
else
    x0 = varargin{4}(1);
    x1 = varargin{4}(2);
    y0 = varargin{4}(3);
    y1 = varargin{4}(4);
end    

ref0 = min([x0 y0]);
ref1 = max([x1 y1 N]);
a = 0;
b = 0;
fig = figure;
axis([x0 x1 y0 y1]);
hold on

line = 0:ref1/2:ref1;
half = 0:0.5:ref1;

for i = 1:length(half)
    plot([line(1) line(end)], [half(i) half(i)], 'r-');
    plot([half(i) half(i)], [line(1) line(end)], 'r-');
end

cmax = 100; %2*round(ref1)
for m=1:order
    switch m
        case 1
            lines='b-';
        case 2
            lines='b--';
        case 3
            lines='b-.';
        case 4
            lines='b:';
    end
    for i=0:m
        a = m - i;
        b = i;
        
%         if mod(b,2) == 1
%             continue
%         end
        
        % a = +ve; b = +ve
        nux = line;
        if b ~= 0
            for c=0:cmax
                nuy = (c*N - a*nux)/b;
                plot(nux, nuy,lines);
%                 nuy = (a*nux - c*N)/b;
%                 plot(nux, nuy, 'r-');
            end
            %plot(nux, -nuy, 'r');
        end      
        nuy = line;
        if a ~= 0
            for c=0:cmax
                nux = (c*N - b*nuy)/a;
                plot(nux, nuy,lines);
%                 nux = (c*N + b*nuy)/a;
%                 plot(nux, nuy, 'r-');
            end
            %                 nux = (c*N + b*nuy)/a;
            %                 plot(nux, nuy, 'r');
        end
        
        % a = -ve; b = -ve
        a = -a;
        b = -b;
%         nux = line;
%         if b ~= 0
%             for c=0:cmax
%                 nuy = (c*N - a*nux)/b;
%                 plot(nux, nuy,lines);
%                 nuy = (a*nux - c*N)/b;
%                 plot(nux, nuy, 'r-');
%             end
%             %                 plot(nux, -nuy, 'r');
%         end      
%         nuy = line;
%         if a ~= 0
%             for c=0:cmax
%                 nux = (c*N - b*nuy)/a;
%                 plot(nux, nuy,lines);
%                 nux = (c*N + b*nuy)/a;
%                 plot(nux, nuy, 'r-');
%             end
%             %                 nux = (c*N + b*nuy)/a;
%             %                 plot(nux, nuy, 'r');
%         end
        
        % a = +ve; b = -ve
        a = -a;
        nux = line;
        if b ~= 0
            for c=0:cmax
                nuy = (c*N - a*nux)/b;
                plot(nux, nuy,lines);
%                 nuy = (a*nux - c*N)/b;
%                 plot(nux, nuy, 'r-');
            end
            %                 plot(nux, -nuy, 'r');
        end      
        nuy = line;
        if a ~= 0
            for c=0:cmax
                nux = (c*N - b*nuy)/a;
                plot(nux, nuy,lines);
%                 nux = (c*N + b*nuy)/a;
%                 plot(nux, nuy, 'r-');
            end
            %                 nux = (c*N + b*nuy)/a;
            %                 plot(nux, nuy, 'r');
        end
        
        % a = -ve; b = +ve
        a = -a;
        b = -b;
        nux = line;
        if b ~= 0
            for c=0:cmax
                nuy = (c*N - a*nux)/b;
                plot(nux, nuy,lines);
%                 nuy = (a*nux - c*N)/b;
%                 plot(nux, nuy, 'r-');
            end
            %                 plot(nux, -nuy, 'r');
        end      
        nuy = line;
        if a ~= 0
            for c=0:cmax
                nux = (c*N - b*nuy)/a;
                plot(nux, nuy,lines);
%                 nux = (c*N + b*nuy)/a;
%                 plot(nux, nuy, 'r-');
            end
            %                 nux = (c*N + b*nuy)/a;
            %                 plot(nux, nuy, 'r');
        end
    end
end

% plot points on the diagram.
plot(x, y, 'ro-');
%axis([13.05 13.45 5.05 5.45])

hold off
