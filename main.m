%% MAIN.M
% Main entry point for the fundus cataract analysis pipeline.
%
% This flat version assumes that all .m files are placed in the same
% directory: main.m, config_default.m, and all run_*.m files.
% Edit config_default.m before running this file.

clear; clc; close all;

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot);

cfg = config_default(projectRoot);

fprintf('\n=== Fundus cataract pipeline ===\n');
fprintf('Project root: %s\n', cfg.projectRoot);
fprintf('Results root: %s\n', cfg.outputRoot);

if cfg.run.extractLBP
    run_lbp_extraction(cfg);
end

if cfg.run.plotIntensityHistograms
    run_intensity_histogram_plots(cfg);
end

if cfg.run.plotGlobalLBP
    run_lbp_global_plots(cfg);
end

if cfg.run.binaryClassification
    run_binary_classification(cfg);
end

if ~cfg.run.extractLBP && ~cfg.run.plotIntensityHistograms && ...
        ~cfg.run.plotGlobalLBP && ~cfg.run.binaryClassification
    warning('No stage is enabled. Open config_default.m and set one or more cfg.run.* flags to true.');
end

fprintf('\nDone.\n');
