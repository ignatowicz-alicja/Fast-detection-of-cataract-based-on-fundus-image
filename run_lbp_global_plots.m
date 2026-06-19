function run_lbp_global_plots(cfg)
%RUN_LBP_GLOBAL_PLOTS Save global LBP histogram plots.
%
% This function loads precomputed LBP matrices for cataract and normal
% classes and exports separate comparison plots for selected channels.

if nargin < 1
    error('Configuration structure cfg is required. Run main.m or call config_default first.');
end

LBP_ROOT   = cfg.globalLbpPlots.lbpRoot;
LBP_CONFIG = cfg.globalLbpPlots.lbpConfig;
LBP_MAT    = cfg.globalLbpPlots.lbpMat;

CAT_LABEL    = cfg.labels.cataract;
NORMAL_LABEL = cfg.labels.normal;

catMatPath  = fullfile(LBP_ROOT, CAT_LABEL, LBP_CONFIG, LBP_MAT);
normMatPath = fullfile(LBP_ROOT, NORMAL_LABEL, LBP_CONFIG, LBP_MAT);

if ~exist(catMatPath, 'file') || ~exist(normMatPath, 'file')
    warning('LBP MAT files were not found. Check cfg.globalLbpPlots.lbpRoot and cfg.globalLbpPlots.lbpConfig.');
    fprintf('Missing or unavailable:\n  %s\n  %s\n', catMatPath, normMatPath);
    return;
end

fprintf('Loading LBP data from: %s\n', LBP_ROOT);
catData = load(catMatPath, "X");   X_cat = catData.X;
normData = load(normMatPath, "X"); X_norm = normData.X;

X_cat  = X_cat(any(X_cat ~= 0, 2) & all(isfinite(X_cat), 2), :);
X_norm = X_norm(any(X_norm ~= 0, 2) & all(isfinite(X_norm), 2), :);

mean_cat_raw  = mean(X_cat, 1);
mean_norm_raw = mean(X_norm, 1);

num_base_bins = cfg.globalLbpPlots.numBaseBins;
num_blocks = length(mean_cat_raw) / num_base_bins;

if abs(num_blocks - round(num_blocks)) > 1e-12
    error('The LBP feature vector length is not divisible by num_base_bins. Check LBP configuration.');
end

global_cat  = sum(reshape(mean_cat_raw, [num_base_bins, num_blocks]), 2)';
global_norm = sum(reshape(mean_norm_raw, [num_base_bins, num_blocks]), 2)';

global_cat = (global_cat / sum(global_cat)) * 100;
global_norm = (global_norm / sum(global_norm)) * 100;

PUB.FontName      = "DejaVu Sans";
PUB.FontSizeAxes  = 16;
PUB.FontSizeTitle = 18;

max_y = max(max(global_cat), max(global_norm)) * 1.1;
x_labels = 0:(num_base_bins-1);

outDir = cfg.globalLbpPlots.outDir;
if ~exist(outDir, 'dir'), mkdir(outDir); end

fig1 = figure('Color', 'w', 'Position', [100, 100, 800, 500]);
ax1 = axes(fig1);
bar(ax1, x_labels, global_norm, 0.7, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'k', 'LineWidth', 1.2);
title(ax1, 'Global LBP Histogram: Normal Eye', 'FontName', PUB.FontName, 'FontSize', PUB.FontSizeTitle);
xlabel(ax1, 'LBP Pattern Index', 'FontName', PUB.FontName, 'FontSize', PUB.FontSizeAxes);
ylabel(ax1, 'Frequency (%)', 'FontName', PUB.FontName, 'FontSize', PUB.FontSizeAxes);
grid(ax1, 'on');
ax1.FontName = PUB.FontName; ax1.FontSize = PUB.FontSizeAxes; ax1.Box = 'on'; ax1.TickDir = 'out'; ax1.LineWidth = 1.2;
xticks(ax1, x_labels);
xlim(ax1, [-0.6, num_base_bins-0.4]); ylim(ax1, [0, max_y]);
exportgraphics(fig1, fullfile(outDir, "Global_LBP_Histogram_Normal.png"), "Resolution", 300);
exportgraphics(fig1, fullfile(outDir, "Global_LBP_Histogram_Normal.pdf"), "ContentType", "vector");
close(fig1);

fig2 = figure('Color', 'w', 'Position', [150, 150, 800, 500]);
ax2 = axes(fig2);
bar(ax2, x_labels, global_cat, 0.7, 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'k', 'LineWidth', 1.2);
title(ax2, 'Global LBP Histogram: Cataract', 'FontName', PUB.FontName, 'FontSize', PUB.FontSizeTitle);
xlabel(ax2, 'LBP Pattern Index', 'FontName', PUB.FontName, 'FontSize', PUB.FontSizeAxes);
ylabel(ax2, 'Frequency (%)', 'FontName', PUB.FontName, 'FontSize', PUB.FontSizeAxes);
grid(ax2, 'on');
ax2.FontName = PUB.FontName; ax2.FontSize = PUB.FontSizeAxes; ax2.Box = 'on'; ax2.TickDir = 'out'; ax2.LineWidth = 1.2;
xticks(ax2, x_labels);
xlim(ax2, [-0.6, num_base_bins-0.4]); ylim(ax2, [0, max_y]);
exportgraphics(fig2, fullfile(outDir, "Global_LBP_Histogram_Cataract.png"), "Resolution", 300);
exportgraphics(fig2, fullfile(outDir, "Global_LBP_Histogram_Cataract.pdf"), "ContentType", "vector");
close(fig2);

fprintf('\nSaved global LBP plots in:\n  %s\n', outDir);
end
