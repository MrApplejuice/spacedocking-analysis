function Dists = cleanDists(Dists, upper_limit, lower_limit)
% function Dists = cleanDists(Dists, upper_limit, lower_limit)
if(~exist('upper_limit', 'var'))
    upper_limit = 10;
    lower_limit = 0;
end

inds = find(Dists > upper_limit);
Dists(inds) = upper_limit;
inds = find(Dists < lower_limit);
Dists(inds) = lower_limit;
