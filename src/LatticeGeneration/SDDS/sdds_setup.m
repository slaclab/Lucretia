sddspath=regexprep(which('DeckTool'),'LatticeGeneration/DeckTool.m','LatticeGeneration/SDDS');
addpath(sddspath);
javaaddpath(fullfile(sddspath,'SDDS.jar'), '-end');
import SDDS.java.SDDS.*