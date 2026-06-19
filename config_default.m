function cfg = config_default(projectRoot)
%CONFIG_DEFAULT Project configuration for the flat GitHub version.
%
% Edit this file instead of editing paths inside the processing code.
% Keep this file in the same directory as main.m and all run_*.m files.
%
% Expected dataset layout for image datasets:
%   data/<dataset_name>/<class_name>/*.png|*.jpg|...
%
% Expected precomputed LBP layout:
%   features/lbp/<class_name>/<lbp_config>/X_LBP_LOCAL.mat
%
% Expected histogram CSV file:
%   features/hist_features.csv

if nargin < 1 || isempty(projectRoot)
    projectRoot = fileparts(mfilename('fullpath'));
end

cfg = struct();
cfg.projectRoot = string(projectRoot);
cfg.randomSeed  = 42;

cfg.dataRoot    = fullfile(cfg.projectRoot, "data");
cfg.featureRoot = fullfile(cfg.projectRoot, "features");
cfg.outputRoot  = fullfile(cfg.projectRoot, "results");

%% Stage switches
% Turn on only the stages you want to run.
cfg.run.extractLBP              = false;
cfg.run.plotIntensityHistograms = false;
cfg.run.plotGlobalLBP           = false;
cfg.run.binaryClassification    = true;

%% Class labels
cfg.labels.cataract = "cataract";
cfg.labels.normal   = "normal";

%% Image extensions
cfg.imageExtensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"];

%% LBP extraction stage
cfg.lbpExtraction = struct();
cfg.lbpExtraction.outRoot = fullfile(cfg.outputRoot, "LBP");
cfg.lbpExtraction.basePaths = [
    fullfile(cfg.dataRoot, "eye_diseases_classification")
    fullfile(cfg.dataRoot, "cataract_dataset")
];
cfg.lbpExtraction.maxPerClass = [1100, 100];
cfg.lbpExtraction.pList = [4 8 16];
cfg.lbpExtraction.radius = 1;
cfg.lbpExtraction.useRiu2 = true;
cfg.lbpExtraction.grids = {[4 4]; [8 8]};
cfg.lbpExtraction.saveFigs = true;
cfg.lbpExtraction.showFigs = false;
cfg.lbpExtraction.figWidth = 2880;
cfg.lbpExtraction.figHeight = 1800;
cfg.lbpExtraction.exportResolution = 450;
cfg.lbpExtraction.saveDebugMasks = true;
cfg.lbpExtraction.maxDebugPerClass = 12;
cfg.lbpExtraction.saveOriginals = true;
cfg.lbpExtraction.savePatchPreview = true;

cfg.lbpExtraction.maskParams = struct();
cfg.lbpExtraction.maskParams.minAreaFrac   = 0.05;
cfg.lbpExtraction.maskParams.innerMarginPx = 2;
cfg.lbpExtraction.maskParams.erodePixels   = 0;

cfg.lbpExtraction.seg = struct();
cfg.lbpExtraction.seg.nonBlackThr       = 0.01;
cfg.lbpExtraction.seg.gaussSigmaNB      = 1.5;
cfg.lbpExtraction.seg.edgeMethod        = "canny";
cfg.lbpExtraction.seg.edgeSigma         = 2.0;
cfg.lbpExtraction.seg.sigmaSmoothEdges  = 3.0;
cfg.lbpExtraction.seg.edgeDilate        = 2;
cfg.lbpExtraction.seg.edgeClose         = 18;
cfg.lbpExtraction.seg.radiusFracRange   = [0.30 0.60];
cfg.lbpExtraction.seg.circleSensitivity = 0.95;
cfg.lbpExtraction.seg.circleEdgeThr     = 0.05;

%% Intensity histogram plotting stage
cfg.intensityHistograms = struct();
cfg.intensityHistograms.inputRoot = fullfile(cfg.dataRoot, "eye_diseases_classification");
cfg.intensityHistograms.outDir    = fullfile(cfg.outputRoot, "intensity_histograms");
cfg.intensityHistograms.maxPerClass = 300;
cfg.intensityHistograms.maxIntensity = 230;
cfg.intensityHistograms.binStep = 1;
cfg.intensityHistograms.smoothWindow = 5;
cfg.intensityHistograms.ranges = [0 10; 0 60; 61 120; 121 180; 181 230];
cfg.intensityHistograms.fullRange = [0 230];
cfg.intensityHistograms.figWidth = 2880;
cfg.intensityHistograms.figHeight = 1800;
cfg.intensityHistograms.exportResolution = 600;
cfg.intensityHistograms.lineWidth = 2.8;
cfg.intensityHistograms.axisFont = 20;
cfg.intensityHistograms.titleFont = 22;
cfg.intensityHistograms.legendFont = 14;
cfg.intensityHistograms.maxXLabels = 25;

%% Global LBP plot stage
cfg.globalLbpPlots = struct();
cfg.globalLbpPlots.lbpRoot = fullfile(cfg.featureRoot, "lbp");
cfg.globalLbpPlots.lbpConfig = "lbp_local_grid4x4_P8_R1_riu2";
cfg.globalLbpPlots.lbpMat = "X_LBP_LOCAL.mat";
cfg.globalLbpPlots.outDir = fullfile(cfg.outputRoot, "global_lbp_plots");
cfg.globalLbpPlots.numBaseBins = 10;

%% Binary classification stage
cfg.binary = struct();
cfg.binary.lbpRoot   = fullfile(cfg.featureRoot, "lbp");
cfg.binary.lbpConfig = "lbp_local_grid4x4_P8_R1_riu2";
cfg.binary.lbpMat    = "X_LBP_LOCAL.mat";
cfg.binary.histCsvFiles = [fullfile(cfg.featureRoot, "hist_features.csv"), ""];

cfg.binary.imageRoot = fullfile(cfg.dataRoot, "ODIR-5K_NORM");
% Optional path remapping. Leave empty unless old absolute paths are stored in MAT/CSV files.
cfg.binary.pathRemapFrom = "";
cfg.binary.pathRemapTo   = "";

cfg.binary.outDir = fullfile(cfg.outputRoot, "binary_classification_" + cfg.binary.lbpConfig);
cfg.binary.kFolds = 5;
cfg.binary.holdout = 0.20;
cfg.binary.boxConstraint = 10;
cfg.binary.dlDoCV = false;

cfg.binary.runLBP      = true;
cfg.binary.runHIST     = true;
cfg.binary.runLBPHIST  = true;
cfg.binary.runRawDL    = true;

cfg.binary.classicModels = ["SVM_RBF", "RF"];
cfg.binary.dlModels      = ["RESNET50_TL", "MOBILENETV2_TL"];

cfg.binary.cnnInputSize = [224 224];
cfg.binary.cnnMiniBatch = 32;
cfg.binary.cnnMaxEpochs = 25;
cfg.binary.cnnLearnRate = 1e-4;
cfg.binary.cnnFreezeBackbone = true;

%% Shared fundus ROI parameters used in binary classification and timing
cfg.roi = struct();
cfg.roi.nonBlackThr = 0.03;
cfg.roi.gaussSigma  = 2.0;
cfg.roi.minAreaFrac = 0.05;
cfg.roi.closeRadius = 12;
cfg.roi.fillHoles   = true;
cfg.roi.useEdges    = true;
cfg.roi.edgeMethod  = "canny";
cfg.roi.edgeSigma   = 2.0;
cfg.roi.edgeDilate  = 2;
cfg.roi.edgeClose   = 20;
cfg.roi.erodePixels = 0;
cfg.roi.innerMarginPx = 2;
cfg.roi.cropToMaskBBox = true;

%% Publication formatting
cfg.pub = struct();
cfg.pub.FontName      = "DejaVu Sans";
cfg.pub.FontSizeAxes  = 15;
cfg.pub.FontSizeTitle = 17;
cfg.pub.LineWidth     = 2.2;
cfg.pub.FigW          = 950;
cfg.pub.FigH          = 750;
cfg.pub.DPI           = 300;
end
