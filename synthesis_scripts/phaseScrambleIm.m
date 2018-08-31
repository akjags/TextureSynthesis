%% phaseScrambleIm.m
%
%   Function to scramble the phase of an image.
%    Takes fourier transform, adds random phase, and returns inverse
%    fourier transform.
%
%    by: akshay jagadeesh
%  date: 08/18/2018
function imf = phaseScrambleIm(imPath, plotFig)

if ieNotDefined('plotFig')
  plotFig = 0;
end

%%
im = double(imread(imPath));
randIm = getHalfFourier(rand(256,256));
d = getHalfFourier(im(:,:,1));
randPhase = rand(size(d.phase))*2*pi - pi;

for i = 1:size(im,3)
  d = getHalfFourier(im(:,:,i));
  d.phase = randPhase;
  imf(:,:,i) = reconstructFromHalfFourier(d);
end

if plotFig
  figure;
  subplot(122); imshow(im./256); title('Original');
  subplot(121); imshow(imf./256); title('Phase Scrambled');
end

%%
function d = getHalfFourier(im)

% make sure there are an odd number of pixels
if iseven(size(im,1)),im = im(1:end-1,:);end
if iseven(size(im,2)),im = im(:,1:end-1);end

% take fourier transform of image
imf = fft2(im);

% get input dimensions
d.originalDims = size(im);

% get one half of fourier image
imfh = fftshift(imf);
imfh = imfh(1:d.originalDims(1),1:ceil(d.originalDims(2)/2));

% extract dc form half fourier image
d.dc = imfh(ceil(d.originalDims(1)/2),end);
halfFourier = imfh(1:(prod(size(imfh))-ceil(d.originalDims(1)/2)));

d.mag = abs(halfFourier);
d.phase = angle(halfFourier);
d.n = length(d.phase);

%%
function im = reconstructFromHalfFourier(d)

d.halfFourier = d.mag.*(cos(d.phase)+i*sin(d.phase));

% first make the last column of the half fourier space which includes
% the dc and should have the frequency components replicated corectly
halfFourier = [d.halfFourier d.dc];
halfFourier(end+1:end+floor(d.originalDims(1)/2)) = conj(d.halfFourier(end:-1:end-floor(d.originalDims(1)/2)+1));
halfFourier = reshape(halfFourier,d.originalDims(1),ceil(d.originalDims(2)/2));

% replicate the frequency components to make the negative frequencies which
% are the complex conjugate of the positive frequncies
halfFourier2 = fliplr(flipud(conj(halfFourier(:,1:floor(d.originalDims(2)/2)))));
im = ifft2(ifftshift([halfFourier halfFourier2]));
