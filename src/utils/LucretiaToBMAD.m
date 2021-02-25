classdef LucretiaToBMAD < handle
  % LucretiaToAT - Convert current Lucretia environment to BMAD
  % LCAV elements are converted to RFCAV elements. In this case, you must
  % define RFCAV properties not present in LCAV by first calling
  % defineRFCAV method
  % BMAD lattice location is stored in global variable BMADLAT
  
  properties(SetAccess=private)
    rfcavData
    title
    rfon=0;
    rfV=0;
    Initial % Lucretia Initial structure for first element
  end
  properties(Constant)
    maxwid=90; % maximum line width for lattice file
  end
  properties
    doSymp=false; % Use smplectic tracking methods, or use Bmad_Standard
    particle_type='positron';
    geomtype='Open'; % Open or Closed
    BMADLAT
    TaylorOrder=5;
  end
  properties(Constant)
    Cb=1e9/2.99792458e8;       % rigidity conversion (T-m/GeV)
  end
  
  methods
    function obj=LucretiaToBMAD(Title,Initial)
        % LUCRETIATOBMAD Converson tool for generating BMAD deck
        %  BMAD = LucretiaToBMAD(Title,Initial)
        %  Provide Title for BMAD deck and Lucretia Initial structure
        %   corresponding to first element
      obj.title=Title;
      obj.Initial=Initial;
    end
    function defineRFCAV(obj,HarmonicNumber,rfon)
      % defineRFCAV(HarmonicNumber,rfon)
      % - set required RFCAV element attributes not included in Lucretia
      % - use HarmonicNumber=0 to use Lucretia value for freq and phase,
      %   else use pi phase and this harmonic number for rf cavity
      % - rfon=1 switches RF on, otherwise off
      % LCAV element
      if nargin<1
        error('Incorrect argument format')
      end
      obj.rfcavData.HarNum=HarmonicNumber;
      if exist('rfon','var') && rfon
        obj.rfon=1;
      end
    end
    function convert(obj)
      global BEAMLINE
      if isempty(obj.BMADLAT)
        error('BMADLAT global not set')
      end
      fid=fopen(fullfile(obj.BMADLAT,sprintf('%s.bmad',obj.title)),'w');
      if fid<=0
        error('Error Opening BMAD lattice file for writing')
      end
      fprintf(fid,'! %s\n',obj.title);
      fprintf(fid,'! ===============================================\n');
      fprintf(fid,'parameter[e_tot] = %.15g * 1e9\n',obj.Initial.Momentum);
      fprintf(fid,'parameter[geometry] = %s\n',obj.geomtype);
      fprintf(fid,'beam_start[emittance_a] = 1e-9\n');
      fprintf(fid,'beam_start[emittance_b] = 1e-9\n');
      fprintf(fid,'beginning[beta_a] = %.15g\n',obj.Initial.x.Twiss.beta);
      fprintf(fid,'beginning[beta_b] = %.15g\n',obj.Initial.y.Twiss.beta);
      fprintf(fid,'beginning[alpha_a] = %.15g\n',obj.Initial.x.Twiss.alpha);
      fprintf(fid,'beginning[alpha_b] = %.15g\n',obj.Initial.y.Twiss.alpha);
      fprintf(fid,'beginning[p0c] = %.15g * 1e9\n',obj.Initial.Momentum);
      fprintf(fid,'parameter[particle] = %s\n',obj.particle_type);
      fprintf(fid,'parameter[taylor_order] = %d\n',obj.TaylorOrder);
      fprintf(fid,'! ===============================================\n');
      names=cellfun(@(x) x.Name,BEAMLINE,'UniformOutput',false);
      linestr=sprintf('%s: line = (',obj.title);
      for iele=1:length(BEAMLINE)
        if iele==1
          linestr=sprintf('%s%s',linestr,BEAMLINE{iele}.Name);
        else
          linestr=sprintf('%s,%s',linestr,BEAMLINE{iele}.Name);
        end
        nm=ismember(names,BEAMLINE{iele}.Name); % Only write out unique elements
        if sum(nm)>1 && iele~=find(nm,1)
          continue
        end
        bmele=sprintf('%s:',BEAMLINE{iele}.Name);
        switch BEAMLINE{iele}.Class
          case 'DRIF'
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            bmele=sprintf('%s drift, l = %.15g, tracking_method = %s',bmele,BEAMLINE{iele}.L,tm);
          case 'SBEN'
%             B=GetTrueStrength(iele);
            if length(BEAMLINE{iele}.B)>1
              K1 = BEAMLINE{iele}.B(2) / (obj.Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L) ;
            else
              K1 = 0;
            end
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            if isnan(K1); K1=0; end
%             ea=BEAMLINE{iele}.EdgeAngle; if length(ea)==1; ea=[ea ea]; end;
%             fi=BEAMLINE{iele}.FINT; if length(fi)==1; fi=[fi fi]; end;
            ang=BEAMLINE{iele}.Angle;
%             bmele=sprintf('%s bend_sol_quad, l = %.15g, angle = %.15g, k1 = %.15g',bmele,BEAMLINE{iele}.L,ang,K1);
            if BEAMLINE{iele}.FINT(1) && BEAMLINE{iele}.FINT(2)
              fringe='both_ends';
            elseif BEAMLINE{iele}.FINT(1)
              fringe='entrance_end';
            elseif BEAMLINE{iele}.FINT(2)
              fringe='exit_end';
            else
              fringe='exit_end';
            end
            bmele=sprintf('%s sbend, l = %.15g, angle = %.15g, k1 = %.15g, tracking_method = %s, fringe_at = %s',...
              bmele,BEAMLINE{iele}.L,ang,K1,tm,fringe);
            if length(BEAMLINE{iele}.B)>2
              bmele=sprintf('%s, k2=%.15g',bmele,BEAMLINE{iele}.B(3))/ (obj.Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L);
            end
          case 'QUAD'
            K1 = BEAMLINE{iele}.B / (obj.Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L) ;
            if isnan(K1); K1=0; end
            if obj.doSymp
              bmele=sprintf('%s Bend_Sol_Quad, l = %.15g, k1 = %.15g, tilt = %.15g, tracking_method = Symp_Lie_Bmad, fringe_at = no_end',bmele,BEAMLINE{iele}.L, K1, BEAMLINE{iele}.Tilt) ;
            else
              bmele=sprintf('%s quad, l = %.15g, k1 = %.15g, tilt = %.15g, tracking_method = Bmad_Standard, fringe_at = no_end',bmele,BEAMLINE{iele}.L, K1, BEAMLINE{iele}.Tilt) ;
            end
          case 'SEXT'
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            K2 = BEAMLINE{iele}.B / (2*obj.Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L) ;
            if isnan(K2); K2=0; end
            bmele=sprintf('%s quad, l = %.15g, k2 = %.15g, tilt = %.15g, tracking_method = %s, fringe_at = no_end',bmele,BEAMLINE{iele}.L, K2, BEAMLINE{iele}.Tilt, tm) ;
          case 'LCAV'
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            if isempty(obj.rfcavData) || ~isfield(obj.rfcavData,'HarNum')
              warning('Lucretia:LucretiaToBMAD_convertError','No RFCAV data defined- treating LCAV element %s as Drift',BEAMLINE{iele}.Name);
              bmele=sprintf('%s drift, l = %.15g, tracking_method = %s',bmele,BEAMLINE{iele}.L,tm);
            else
              if obj.rfcavData.HarNum>0 % Use specified harmonic number for cavity and set pi phase
                bmele=sprintf('%s rfcavity, l = %.15g, voltage = %.15g, harmon = %.15g, phi0 = %.15g, tracking_method = %s',...
                  bmele,BEAMLINE{iele}.L,BEAMLINE{iele}.Volt*1e6,obj.rfcavData.HarNum,pi,tm) ;
              else % Use lucretia specified cavity frequency and phase
                bmele=sprintf('%s lcavity, l = %.15g, voltage = %.15g, rf_frequency = %.15g, phi0 = %.15g, tracking_method = Bmad_Standard',...
                  bmele,BEAMLINE{iele}.L,BEAMLINE{iele}.Volt*1e6,BEAMLINE{iele}.Freq*1e6,deg2rad(BEAMLINE{iele}.Phase)/(2*pi)) ;
                if obj.doSymp
                  bmele=sprintf('%s, Symplectify=t',bmele);
                end
              end
            end
            
          case 'SOLENOID'
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            KS = BEAMLINE{iele}.B / (obj.Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L) ;
            bmele=sprintf('%s solenoid, l = %.15g, ks = %.15g, tracking_method = %s',bmele,BEAMLINE{iele}.L,KS,tm);
          case 'TMAP' % only R and T matrix supported by AT (note dims 5 and 6 switched between Lucretia and AT)
            bmele=sprintf('%s taylor, tracking_method = Taylor',bmele);
            R=BEAMLINE{iele}.R;
            for i1=1:6
              for i2=1:6
                if R(i1,i2)~=0 && (i1==i2 && R(i1,i2)~=1)
                  bmele=sprintf('%s, {%d: %.15g | %d}',bmele,i1,R(i1,i2),i2);
                end
              end
            end
            if ~isempty(BEAMLINE{iele}.T)
              for iT=1:length(BEAMLINE{iele}.T)
                ns=num2str(BEAMLINE{iele}.Tinds(iT));
                bmele=sprintf('%s, {%s: %.15g | %s}',bmele,ns(1),BEAMLINE{iele}.T(iT),ns(2:3));
              end
            end
            if ~isempty(BEAMLINE{iele}.U)
              for iU=1:length(BEAMLINE{iele}.U)
                ns=num2str(BEAMLINE{iele}.Uinds(iU));
                bmele=sprintf('%s, {%s: %.15g | %s}',bmele,ns(1),BEAMLINE{iele}.U(iU),ns(2:4));
              end
            end
            if ~isempty(BEAMLINE{iele}.V)
              for iV=1:length(BEAMLINE{iele}.V)
                ns=num2str(BEAMLINE{iele}.Vinds(iV));
                bmele=sprintf('%s, {%s: %.15g | %s}',bmele,ns(1),BEAMLINE{iele}.V(iV),ns(2:5));
              end
            end
            if ~isempty(BEAMLINE{iele}.W)
              for iW=1:length(BEAMLINE{iele}.W)
                ns=num2str(BEAMLINE{iele}.Winds(iW));
                bmele=sprintf('%s, {%s: %.15g | %s}',bmele,ns(1),BEAMLINE{iele}.W(iW),ns(2:6));
              end
            end
          case 'MULT'
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            if BEAMLINE{iele}.L>0
              bmele=sprintf('%s multipole, tracking_method = %s, l = %.15g',bmele,tm,BEAMLINE{iele}.L);
            else
              bmele=sprintf('%s multipole, tracking_method = %s',bmele,tm);
            end
            ind=1;
            for index=BEAMLINE{iele}.PoleIndex
              bmele=sprintf('%s, k%dl = %.15g, t%d = %.15g',bmele,index,BEAMLINE{iele}.B(ind),index,BEAMLINE{iele}.Tilt(ind));
            end
          case 'MARK'
            bmele=sprintf('%s mark',bmele);
          otherwise
            warning('Lucretia:LucretiaToBMAD_convertError','Unsupported Class: %s, treating as marker or drift: element %s...\n',BEAMLINE{iele}.Class,BEAMLINE{iele}.Name);
            if obj.doSymp
              tm='Symp_Lie_PTC';
            else
              tm='Bmad_Standard';
            end
            if isfield(BEAMLINE{iele},'L')
              bmele=sprintf('%s drift, l = %.15g, tracking_method = %s',bmele,BEAMLINE{iele}.L,tm);
            else
              bmele=sprintf('%s mark',bmele);
            end
        end
        bmelenew='';
        while length(bmele)>=obj.maxwid
          pars=regexp(bmele,',');
          ispl=pars(find(pars<obj.maxwid,1,'last'));
          bmelenew=sprintf('%s%s &\n',bmelenew,bmele(1:ispl));
          bmele=bmele(ispl+1:end);
        end
        bmelenew=[bmelenew bmele]; %#ok<AGROW>
        fprintf(fid,'%s\n',bmelenew);
      end
      fprintf(fid,'! ===============================================\n');
      linestr=sprintf('%s)\n',linestr);
      lstr='';
      while length(linestr)>=obj.maxwid
        pars=regexp(linestr,',');
        ispl=pars(find(pars<obj.maxwid,1,'last'));
        lstr=sprintf('%s%s &\n',lstr,linestr(1:ispl));
        linestr=linestr(ispl+1:end);
      end
      lstr=[lstr linestr];
      fprintf(fid,'%s\n',lstr);
      fprintf(fid,sprintf('use, %s',obj.title));
      fclose(fid);
    end
  end
  
end

