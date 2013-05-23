function Dists = cleanDists(Dists, limit)
% function Dists = cleanDists(Dists, limit)
if(~exist('limit', 'var'))
    limit = 5;
end

inds = find(Dists > limit);
Dists(inds) = limit;
inds = find(Dists < -limit);
Dists(inds) = -limit;
