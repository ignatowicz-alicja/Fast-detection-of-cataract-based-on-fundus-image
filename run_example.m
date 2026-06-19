%% Example run
% This example shows how to load the default configuration and enable one stage.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'config'));
addpath(genpath(fullfile(projectRoot, 'src')));

cfg = config_default(projectRoot);

% Example: run only the global LBP plots.
cfg.run.extractLBP              = false;
cfg.run.plotIntensityHistograms = false;
cfg.run.plotGlobalLBP           = true;
cfg.run.binaryClassification    = false;

run_lbp_global_plots(cfg);
