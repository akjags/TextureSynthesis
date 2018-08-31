%% generatePhaseScramble.m
%
%   Function to generate phase scrambled (spectrally matched noise) images
%    and save them out to a directory
%
%    by: akshay jagadeesh
%  date: 08/18/2018
%
function generatePhaseScramble(origImPath, outDirPath)

orig_ims = dir([origImPath '/*.jpg']);
orig_ims = {orig_ims.name};

for imI = 1:length(orig_ims)
  
  imPath = [origImPath '/' orig_ims{imI}];
  disp(sprintf('Phase scrambling image: %s', imPath));
  
  im = phaseScrambleIm(imPath)./256;
  
  saveName = sprintf('%s/noise_%s', outDirPath, orig_ims{imI});
  imwrite(im, saveName, 'JPG');
  disp(sprintf('Saved noise_%s to %s', orig_ims{imI}, outDirPath));
end

