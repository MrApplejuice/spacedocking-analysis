function sliceClustering(Y, T, n_slices)
% function sliceClustering(Y, T, n_slices)

if(~exist('n_slices', 'var') || isempty(n_slices))
    n_slices = 9;
end

RANGE_SLICING = true;

n_samples = length(T);
[V, inds] = sort(T, 'descend');

n_plots = ceil(sqrt(n_slices));
n_slices = n_plots * n_plots;

if(RANGE_SLICING)
    min_T = min(T);
    max_T = max(T);
    T_interval = (max_T - min_T) / n_slices;
else
    n_inds_slice = floor(n_samples / n_slices);
end

figure();
for s = 1:n_slices
    subplot(n_plots, n_plots, s);
    if(RANGE_SLICING)
        start_slice = (s-1) * T_interval;
        end_slice = s * T_interval;
        inds_slice = find(T >= start_slice & T <= end_slice);
    else
        start_slice = (s-1) * n_inds_slice + 1;
        end_slice = s * n_inds_slice;
        inds_slice = inds(start_slice:end_slice);
    end
    scatter(Y(inds_slice,1), Y(inds_slice,2), 5, T(inds_slice));
    colorbar;
end