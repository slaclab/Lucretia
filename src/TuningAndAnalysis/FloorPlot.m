classdef FloorPlot < handle
  % Plotting functions for physical accelerator layout
  
  properties % BEAMLINE Classes to draw (should be given dx and dy fields by constructor)
    QUAD
    SEXT
    SBEN
    LCAV
    MARK
  end
  properties % world properties
    WORLD
    INITC=[0 0 0]; % initial coordinate offset to apply
    INITANG=[0 0 0]; % initial angular offset to apply
  end
  
  methods
    function obj=FloorPlot()
      % Magnet sizes
      obj.QUAD.dx=0.15;
      obj.QUAD.dy=0.15;
      obj.QUAD.col='r';
      obj.SBEN.dx=0.1;
      obj.SBEN.dy=0.1;
      obj.SBEN.col='b';
      obj.SEXT.dx=0.1;
      obj.SEXT.dy=0.1;
      obj.SEXT.col='g';
      obj.LCAV.dx=0.1;
      obj.LCAV.dy=0.1;
      obj.LCAV.col='y';
      obj.MARK.dx=0.1;
      obj.MARK.dy=0.1;
      obj.MARK.col='m';
      obj.WORLD.dx=0.5;
      obj.WORLD.dy=0.5;
      obj.WORLD.dz=0.5;
    end
    function plot2dM(obj,mind,pdim,dx)
      global BEAMLINE
      % plot a marker
      hold on
      C=BEAMLINE{mind}.Coordi;
      th=BEAMLINE{mind}.Anglei;
      C(3)=C(3)-obj.INITC(3);
      C(pdim)=C(pdim)-obj.INITC(pdim);
      x1=C(3)-dx*sin(th);
      x2=C(3)+dx*sin(th);
      y1=C(pdim)-dx*cos(th);
      y2=C(pdim)+dx*cos(th);
      line([x1 x2],[y1 y2],'LineWidth',2,'Color','m');
      if isfield(BEAMLINE{mind},'CommonName')
        tstr=BEAMLINE{mind}.CommonName;
      else
        tstr=BEAMLINE{mind}.Name;
      end
      text(x1,y2+dx*0.2,tstr,'Color','m')
    end
    function plot2d(obj,pdim,ind1,ind2,cmd)
      global BEAMLINE
      % 'rectangle' function defines bottom-left as origin
      % Get maximum extent of world and draw world rectangle
      if ~exist('ind1','var') || ~exist('ind2','var')
        ind1=1; ind2=length(BEAMLINE);
      end
      lab={'dx' 'dy' 'dz'};
      if exist('cmd','var') && strcmpi(cmd,'add')
        hold on
        ax=axis;
        max1=max(cellfun(@(x) x.Coordf(3),BEAMLINE(ind1:ind2)))-obj.INITC(3);
        min1=min(cellfun(@(x) x.Coordi(3),BEAMLINE(ind1:ind2)))-obj.INITC(3);
        max2=max(cellfun(@(x) x.Coordf(pdim),BEAMLINE(ind1:ind2)))-obj.INITC(pdim);
        min2=min(cellfun(@(x) x.Coordf(pdim),BEAMLINE(ind1:ind2)))-obj.INITC(pdim);     
        ax(1)=min([ax(1) min1-obj.WORLD.(lab{3})]);
        ax(2)=max([ax(2) max1+obj.WORLD.(lab{3})]);
        ax(3)=min([ax(3) min2-obj.WORLD.(lab{pdim})]);
        ax(4)=max([ax(4) max2+obj.WORLD.(lab{pdim})]);
        axis(ax);
      else
        figure
        max1=max(cellfun(@(x) x.Coordf(3),BEAMLINE(ind1:ind2)))-obj.INITC(3);
        min1=min(cellfun(@(x) x.Coordi(3),BEAMLINE(ind1:ind2)))-obj.INITC(3);
        max2=max(cellfun(@(x) x.Coordf(pdim),BEAMLINE(ind1:ind2)))-obj.INITC(pdim);
        min2=min(cellfun(@(x) x.Coordf(pdim),BEAMLINE(ind1:ind2)))-obj.INITC(pdim);     
        axis([min1-obj.WORLD.(lab{3}) max1+obj.WORLD.(lab{3}) min2-obj.WORLD.(lab{pdim}) max2+obj.WORLD.(lab{pdim})]);
        axis tight
        axis
        hold on
      end
      lastCoord=[];
      pp=properties(obj);
      aper=[];
      for ibl=ind1:ind2
        if ismember(BEAMLINE{ibl}.Class,pp)
          type=BEAMLINE{ibl}.Class;
          th1=BEAMLINE{ibl}.Anglei(pdim)-obj.INITANG(pdim);
          th2=BEAMLINE{ibl}.Anglef(pdim)-obj.INITANG(pdim);
          if isfield(BEAMLINE{ibl},'EdgeAngle')
            th1=th1-BEAMLINE{ibl}.EdgeAngle(1);
            th2=th2+BEAMLINE{ibl}.EdgeAngle(2);
          end
          if isfield(BEAMLINE{ibl},'aper')
            aper=BEAMLINE{ibl}.aper;
          elseif isfield(BEAMLINE{ibl},'HGAP')
            aper=max(BEAMLINE{ibl}.HGAP);
          elseif isempty(aper)
            aper=0.000000001;
          end
          if length(aper)==1; aper=ones(1,2).*aper; end
          if any(aper==1)
            aper=[0.015 0.015];
          end
          % Draw beampipe from aperture to aperture
          Ci=BEAMLINE{ibl}.Coordi-obj.INITC;
          Cf=BEAMLINE{ibl}.Coordf-obj.INITC;
          xm2=Ci(3)-aper(pdim)*sin(th1);
          xp2=Ci(3)+aper(pdim)*sin(th1);
          ym2=Ci(pdim)-aper(pdim)*cos(th1);
          yp2=Ci(pdim)+aper(pdim)*cos(th1);
          if ~isempty(lastCoord) && (Ci(3)-lastCoord(3))>0
            xm1=lastCoord(3)-lastCoord(6+pdim)*sin(lastCoord(3+pdim));
            xp1=lastCoord(3)+lastCoord(6+pdim)*sin(lastCoord(3+pdim));
            ym1=lastCoord(pdim)-lastCoord(6+pdim)*cos(lastCoord(3+pdim));
            yp1=lastCoord(pdim)+lastCoord(6+pdim)*cos(lastCoord(3+pdim));
            fill([xp1 xm1 xm2 xp2],[ym1 yp1 yp2 ym2],'k');
          end
          % Draw object + beam pipe in object
          xm3=Cf(3)-aper(pdim)*sin(th2);
          xp3=Cf(3)+aper(pdim)*sin(th2);
          ym3=Cf(pdim)-aper(pdim)*cos(th2);
          yp3=Cf(pdim)+aper(pdim)*cos(th2);
          fill([xp2 xm2 xm3 xp3],[ym2 yp2 yp3 ym3],'k');
          xm1=Ci(3)-obj.(type).(lab{pdim})*sin(th1);
          xp1=Ci(3)+obj.(type).(lab{pdim})*sin(th1);
          ym1=Ci(pdim)-obj.(type).(lab{pdim})*cos(th1);
          yp1=Ci(pdim)+obj.(type).(lab{pdim})*cos(th1);
          xm2=Cf(3)-obj.(type).(lab{pdim})*sin(th2);
          xp2=Cf(3)+obj.(type).(lab{pdim})*sin(th2);
          ym2=Cf(pdim)-obj.(type).(lab{pdim})*cos(th2);
          yp2=Cf(pdim)+obj.(type).(lab{pdim})*cos(th2);
          fill([xp1 xm1 xm2 xp2],[ym1 yp1 yp2 ym2],obj.(type).col,'EdgeColor','none');
          % Store last coordinate and aperture
          lastCoord=[Cf BEAMLINE{ibl}.Anglef-obj.INITANG aper(1) aper(2)];
        end
      end
      hold off
      grid on
      ylab={'X / m' 'Y / m'};
      xlabel('Z / m'); ylabel(ylab(pdim));
    end
  end
  
end

