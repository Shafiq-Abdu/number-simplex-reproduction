%% ============================================================
% SPARSE LDA DIAGNOSTIC — YFM neuron 63
% ============================================================

clear;
clc;


%% ============================================================
% Find project root from THIS script
% ============================================================

scriptPath = mfilename('fullpath');
scriptFolder = fileparts(scriptPath);

% test.m is inside:
% number-simplex-reproduction/scripts/
%
% Therefore parent folder is project root.

projectRoot = fileparts(scriptFolder);

csvFile = fullfile( ...
    projectRoot, ...
    'tables', ...
    'diagnostic_YFM_neuron63_sparse_lda.csv');


fprintf('Project root:\n%s\n\n', projectRoot);

fprintf('Loading:\n%s\n\n', csvFile);


%% ============================================================
% Load exported Python data
% ============================================================

T = readtable(csvFile);

X = T.X;
y = T.y;

fprintf('Observations: %d\n', length(y));
fprintf('Total spikes: %.0f\n', sum(X));
fprintf('Nonzero observations: %d\n', nnz(X));


%% ============================================================
% 4-fold stratified cross-validation
% ============================================================

rng(0);

yCat = categorical(y);

cv = cvpartition( ...
    yCat, ...
    'KFold', ...
    4);

gamma = 0.2;

nCorrect = 0;
nPredictions = 0;
nSuccessfulFolds = 0;


fprintf('\n');
fprintf('============================================================\n');
fprintf('MATLAB FITCDISCR TEST\n');
fprintf('============================================================\n');


for fold = 1:cv.NumTestSets

    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    Xtrain = X(trainIdx, :);
    Xtest  = X(testIdx, :);

    ytrain = yCat(trainIdx);
    ytest  = yCat(testIdx);


    fprintf('\nFold %d\n', fold);
    fprintf('Training spikes: %.0f\n', sum(Xtrain));
    fprintf('Test spikes: %.0f\n', sum(Xtest));
    fprintf('Training variance: %.10f\n', var(Xtrain));


    try

        mdl = fitcdiscr( ...
            Xtrain, ...
            ytrain, ...
            'DiscrimType', 'linear', ...
            'Gamma', gamma, ...
            'Prior', 'uniform');

        ypred = predict( ...
            mdl, ...
            Xtest);

        foldCorrect = sum(ypred == ytest);
        foldN = length(ytest);

        fprintf('fitcdiscr: SUCCESS\n');
        fprintf( ...
            'Fold accuracy: %.10f\n', ...
            foldCorrect / foldN);

        nCorrect = nCorrect + foldCorrect;
        nPredictions = nPredictions + foldN;
        nSuccessfulFolds = nSuccessfulFolds + 1;


    catch ME

        fprintf('fitcdiscr: FAILED\n');
        fprintf( ...
            'Identifier: %s\n', ...
            ME.identifier);

        fprintf( ...
            'Message: %s\n', ...
            ME.message);

    end

end


%% ============================================================
% Final result
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('RESULT\n');
fprintf('============================================================\n');

fprintf( ...
    'Successful folds: %d / %d\n', ...
    nSuccessfulFolds, ...
    cv.NumTestSets);

fprintf( ...
    'Predictions obtained: %d / %d\n', ...
    nPredictions, ...
    length(y));


if nPredictions == length(y)

    accuracy = nCorrect / nPredictions;

    fprintf( ...
        'Cross-validated accuracy: %.10f\n', ...
        accuracy);

else

    fprintf( ...
        'Complete CV accuracy could not be calculated.\n');

end