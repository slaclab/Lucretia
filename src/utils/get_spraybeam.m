function bunchout = get_spraybeam(E, dE)

bunch=load('spray.dat');
bunchout=[];

for f=1:length(bunch)
    if ( bunch(f,6)>(-30+E-(dE*E)) ) && ( bunch(f,6)<(-30+E+(dE*E)) )
        bunchout=[bunchout; bunch(f,:)];
    end
end

bunchout(:,6)=(30+bunchout(:,6)-E)/E;

return