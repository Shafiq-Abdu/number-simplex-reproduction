%% ============================================================
% REGIONAL SPECIFICITY OF TEMPORAL NUMBER CODING
%
% Model:
% IsTuned ~ Region + (1 | Subject)
%
% Distribution: Binomial
% Link: Logit
% ============================================================

clear;
clc;


%% ------------------------------------------------------------
% 1. FIND PROJECT ROOT ROBUSTLY
% ------------------------------------------------------------

script_path = mfilename('fullpath');
script_dir = fileparts(script_path);
project_root = fileparts(script_dir);

csv_path = fullfile( ...
    project_root, ...
    'tables', ...
    'arithmetic_temporal_region_glmm_input.csv' ...
);


fprintf('Reading file:\n%s\n\n', csv_path);


%% ------------------------------------------------------------
% 2. LOAD DATA
% ------------------------------------------------------------

T = readtable( ...
    csv_path, ...
    'TextType', 'string' ...
);


fprintf('Number of neurons: %d\n', height(T));
fprintf('Number of tuned neurons: %d\n', sum(T.IsTuned));
fprintf('Number of subjects: %d\n\n', numel(unique(T.Subject)));


%% ------------------------------------------------------------
% 3. CONVERT VARIABLES TO CATEGORICAL
% ------------------------------------------------------------

T.Subject = categorical(T.Subject);

T.Region = categorical( ...
    T.Region, ...
    {'hpc', 'ent', 'amy', 'para-hpc'} ...
);


%% ------------------------------------------------------------
% 4. FIT BINOMIAL GENERALIZED LINEAR MIXED MODEL
%
% Fixed effect:
% Region
%
% Random effect:
% Subject-specific intercept
% ------------------------------------------------------------

glme = fitglme( ...
    T, ...
    'IsTuned ~ Region + (1|Subject)', ...
    'Distribution', 'Binomial', ...
    'Link', 'logit' ...
);


%% ------------------------------------------------------------
% 5. DISPLAY MODEL RESULTS
% ------------------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf('GLMM RESULTS\n');
fprintf('============================================================\n\n');

disp(glme);


%% ------------------------------------------------------------
% 6. OVERALL REGION TEST
% ------------------------------------------------------------

anova_results = anova(glme);

fprintf('\n');
fprintf('============================================================\n');
fprintf('ANOVA: OVERALL REGION EFFECT\n');
fprintf('============================================================\n\n');

disp(anova_results);