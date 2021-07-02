% [LEV,IND] = spyrBand(PYR,INDICES,LEVEL,BAND)
%
% Access a band from a steerable pyramid.
% 
%   LEVEL indicates the scale (finest = first, coarsest = spyrHt(INDICES)).
% 
%   BAND (optional, default=first) indicates which subband
%     (first = vertical, rest proceeding anti-clockwise).

% Eero Simoncelli, 6/96.

function res =  spyrBand(pyr,pind,level,band)

if (exist('level') ~= 1)
  level = 1;
end

if (exist('band') ~= 1)
  band = 1;
end

nbands = spyrNumBands(pind);
if ((band > nbands) | (band < 1))
  error(sprintf('Bad band number (%d) should be in range [1,%d].', band, nbands));
end	

maxLev = spyrHt(pind);
if ((level > maxLev) | (level < 1))
  error(sprintf('Bad level number (%d), should be in range [1,%d].', level, maxLev));
end

firstband = 1 + band + nbands*(level-1);
res  = pyrBand(pyr, pind, firstband);

