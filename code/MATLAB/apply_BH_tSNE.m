function apply_BH_tSNE(X, T, n_selected, perplexity, theta, dims)
% function apply_BH_tSNE(X, T, n_selected, perplexity, theta, dims)

addpath('../py2/bh_tsne');

if(~exist('perplexity', 'var') || isempty(perplexity))
    perplexity = 10;
end
if(~exist('theta', 'var') || isempty(theta))
    theta = 0.5;
end
if(~exist('dims', 'var') || isempty(dims))
    dims = 30;
end

if(~exist('T', 'var') || isempty(T))
    load('orientations.txt')
    load('responses.txt')
    load('sizes.txt')
end
if(~exist('X', 'var') || isempty(X))
    load('X.mat')
end

n_samples = size(X,1);

% select a subset in the interest of not running out of memory:
if(~exist('n_selected') || isempty(n_selected))
    n_selected = n_samples;
else
    inds = randperm(n_samples);
    inds = inds(1:n_selected);
    X = X(inds,:);
    save('inds', 'inds');
    orientations = orientations(inds);
    responses = responses(inds);
    sizes = sizes(inds);
end

curr = pwd;
cd ../py2/bh_tsne
pause(1);
% run tSNE:
Y = fast_tsne(X, dims, perplexity, theta);
cd(curr); pause(1);

% save the results:
save('Y', 'Y');

if(~exist('T', 'var') || isempty(T))
    % plot the related feature properties:
    figure();
    scatter(Y(:,1),Y(:,2),5,responses);
    title('Responses');

    figure();
    scatter(Y(:,1),Y(:,2),5,sizes);
    title('Sizes');

    figure();
    scatter(Y(:,1),Y(:,2),5,orientations);
    title('Orientations');
else
    figure();
    scatter(Y(:,1),Y(:,2),5,T);
    title(['Perpl ' num2str(perplexity) ' theta ' num2str(theta) ' dims ' num2str(dims)]);
end



