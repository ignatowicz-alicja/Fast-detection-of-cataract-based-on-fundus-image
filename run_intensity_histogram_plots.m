function run_intensity_histogram_plots(cfg)
%RUN_INTENSITY_HISTOGRAM_PLOTS Plot RGB and grayscale intensity histograms.
%
% The function computes intensity histograms inside the fundus ROI mask
% and saves publication-style figures for all configured classes/channels.

if nargin < 1
    error('Configuration structure cfg is required. Run main.m or call config_default first.');
end

rng(cfg.randomSeed);

%% === SETTINGS FROM CONFIG ===
inputRoot = cfg.intensityHistograms.inputRoot;
outDir    = cfg.intensityHistograms.outDir;
if ~exist(outDir,"dir"), mkdir(outDir); end

if ~exist(inputRoot, "dir")
    warning("Input directory does not exist: %s", inputRoot);
    return;
end

MAX_PER_CLASS = cfg.intensityHistograms.maxPerClass;
MAX_INT = cfg.intensityHistograms.maxIntensity;
BIN_STEP = cfg.intensityHistograms.binStep;
edges = 0:BIN_STEP:(MAX_INT+1);
bins  = edges(1:end-1) + BIN_STEP/2;
nBins = numel(bins);
imgExt = cellstr(cfg.imageExtensions);
SMOOTH_WIN = cfg.intensityHistograms.smoothWindow;
RANGES = cfg.intensityHistograms.ranges;
FULL_RANGE = cfg.intensityHistograms.fullRange;
FIG_W = cfg.intensityHistograms.figWidth;
FIG_H = cfg.intensityHistograms.figHeight;
EXPORT_RES = cfg.intensityHistograms.exportResolution;
LINE_W = cfg.intensityHistograms.lineWidth;
AXIS_FONT = cfg.intensityHistograms.axisFont;
TITLE_FONT = cfg.intensityHistograms.titleFont;
LEGEND_FONT = cfg.intensityHistograms.legendFont;
MAX_X_LABELS = cfg.intensityHistograms.maxXLabels;

p = cfg.roi;

%% === LIST CLASSES ===
d = dir(inputRoot);
d = d([d.isdir]);
d = d(~ismember({d.name},{'.','..'}));
classNames = string({d.name});
nC = numel(classNames);

fprintf("Found classes (%d): %s\n", nC, strjoin(classNames, ", "));

%% === STORAGE ===
C_R = zeros(nC, nBins);
C_G = zeros(nC, nBins);
C_B = zeros(nC, nBins);
C_Y = zeros(nC, nBins);

%% === COMPUTE HISTOGRAMS PER CLASS (COUNTS) ONLY INSIDE MASK ===
for c = 1:nC
    className = classNames(c);
    classPath = fullfile(inputRoot, className);

    files = dir(classPath);
    files = files(~[files.isdir]);

    keep = false(size(files));
    for i = 1:numel(files)
        [~,~,e] = fileparts(files(i).name);
        keep(i) = any(strcmpi(e, imgExt));
    end
    files = files(keep);

    if isempty(files)
        warning("No images in %s", classPath);
        continue;
    end

    nAll = numel(files);
    if nAll > MAX_PER_CLASS
        files = files(randperm(nAll, MAX_PER_CLASS));
        fprintf("[%s] Sampling %d/%d images\n", className, MAX_PER_CLASS, nAll);
    else
        fprintf("[%s] Using all %d images\n", className, nAll);
    end

    hR = zeros(1,nBins); hG = zeros(1,nBins); hB = zeros(1,nBins); hY = zeros(1,nBins);
    ok = 0; fail = 0;

    for i = 1:numel(files)
        imgPath = fullfile(classPath, files(i).name);
        try
            I = imread(imgPath);
            I = force_rgb(I);
            Iu8 = toUint8_robust(I);
            Id  = im2double(Iu8);

            mask = compute_fundus_mask(Id, p);
            if nnz(mask) < 1000
                fail = fail + 1;
                continue;
            end

            R = Iu8(:,:,1); G = Iu8(:,:,2); B = Iu8(:,:,3);
            Y = rgb2gray(Iu8);

            rVals = double(R(mask)); rVals(rVals > MAX_INT) = MAX_INT;
            gVals = double(G(mask)); gVals(gVals > MAX_INT) = MAX_INT;
            bVals = double(B(mask)); bVals(bVals > MAX_INT) = MAX_INT;
            yVals = double(Y(mask)); yVals(yVals > MAX_INT) = MAX_INT;

            hR = hR + histcounts(rVals, edges);
            hG = hG + histcounts(gVals, edges);
            hB = hB + histcounts(bVals, edges);
            hY = hY + histcounts(yVals, edges);

            ok = ok + 1;
        catch
            fail = fail + 1;
        end
    end

    if ok == 0
        warning("No valid images for %s (failed=%d)", className, fail);
        continue;
    end

    fprintf("  -> %s: ok=%d  failed=%d\n", className, ok, fail);

    C_R(c,:) = hR;
    C_G(c,:) = hG;
    C_B(c,:) = hB;
    C_Y(c,:) = hY;
end

%% Optional smoothing
if SMOOTH_WIN > 1
    C_R = movmean(C_R, SMOOTH_WIN, 2);
    C_G = movmean(C_G, SMOOTH_WIN, 2);
    C_B = movmean(C_B, SMOOTH_WIN, 2);
    C_Y = movmean(C_Y, SMOOTH_WIN, 2);
end

%% === 2D PLOTS: per channel per range + FULL RANGE ===
plotAndSaveAllRanges2D("R",    C_R, bins, classNames, RANGES, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);
plotAndSaveAllRanges2D("G",    C_G, bins, classNames, RANGES, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);
plotAndSaveAllRanges2D("B",    C_B, bins, classNames, RANGES, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);
plotAndSaveAllRanges2D("Gray", C_Y, bins, classNames, RANGES, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);

% FULL range 2D (0..230)
plotAndSaveSingleRange2D("R",    C_R, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);
plotAndSaveSingleRange2D("G",    C_G, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);
plotAndSaveSingleRange2D("B",    C_B, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);
plotAndSaveSingleRange2D("Gray", C_Y, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, ...
    LINE_W, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MAX_X_LABELS);

%% === 3D PLOTS (ALL CLASSES) ===
plotAndSave3D("R",    C_R, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);
plotAndSave3D("G",    C_G, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);
plotAndSave3D("B",    C_B, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);
plotAndSave3D("Gray", C_Y, bins, classNames, FULL_RANGE, outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);

plotAndSave3D("R",    C_R, bins, classNames, [0 10], outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);
plotAndSave3D("G",    C_G, bins, classNames, [0 10], outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);
plotAndSave3D("B",    C_B, bins, classNames, [0 10], outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);
plotAndSave3D("Gray", C_Y, bins, classNames, [0 10], outDir, FIG_W, FIG_H, EXPORT_RES, AXIS_FONT, TITLE_FONT);

fprintf("\nDone. Full-res 2D + 3D plots saved to: %s\n", outDir);

%% ===================== FUNCTIONS =====================


end

function plotAndSaveAllRanges2D(channelName, C, bins, classNames, ranges, outDir, ...
    figW, figH, exportRes, lineW, axisFont, titleFont, legendFont, maxXlabels)

    for k = 1:size(ranges,1)
        plotAndSaveSingleRange2D(channelName, C, bins, classNames, ranges(k,:), outDir, ...
            figW, figH, exportRes, lineW, axisFont, titleFont, legendFont, maxXlabels);
    end
end

function plotAndSaveSingleRange2D(channelName, C, bins, classNames, range, outDir, ...
    figW, figH, exportRes, lineW, axisFont, titleFont, legendFont, maxXlabels)

    xMin = range(1); xMax = range(2);
    m = (bins >= xMin) & (bins <= xMax);
    x = bins(m);

    fig = figure("Visible","off","Units","pixels","Position",[50 50 figW figH],"Color","w");
    ax = axes(fig); 
    set(ax, "Color","w");
    hold(ax, "on");

    cols = lines(size(C,1));
    for i = 1:size(C,1)
        plot(ax, x, C(i,m), "LineWidth", lineW, "Color", cols(i,:));
    end

    xlim(ax, [xMin xMax]);
    apply_dense_grid_no_warnings(ax, xMin, xMax, maxXlabels);

    xlabel(ax, "Pixel intensity", "FontSize", axisFont);
    ylabel(ax, "Number of pixels", "FontSize", axisFont);
    title(ax, sprintf("%s channel (counts, masked fundus)  [%d..%d]", channelName, xMin, xMax), ...
        "Interpreter","none", "FontSize", titleFont);

    lg = legend(ax, classNames, "Interpreter","none", "Location","eastoutside");
    lg.FontSize = legendFont;
    lg.Color = "w";        % white legend background
    lg.EdgeColor = "k";

    ax.FontSize = axisFont;

    outName = sprintf("overlay2D_%s_%d_%d_full.png", channelName, xMin, xMax);
    exportgraphics(fig, fullfile(outDir, outName), "Resolution", exportRes, "BackgroundColor","white");
    close(fig);
end

function plotAndSave3D(channelName, C, bins, classNames, range, outDir, ...
    figW, figH, exportRes, axisFont, titleFont)

    xMin = range(1); xMax = range(2);
    m = (bins >= xMin) & (bins <= xMax);
    x = bins(m);

    Y = 1:numel(classNames);
    [Xg, Yg] = meshgrid(x, Y);
    Zg = C(:,m);

    fig = figure("Visible","off","Units","pixels","Position",[50 50 figW figH],"Color","w");
    ax = axes(fig); %#ok<LAXES>
    set(ax, "Color","w");
    hold(ax, "on");

    s = surf(ax, Xg, Yg, Zg);
    s.EdgeColor = "none";
    view(ax, 45, 25);
    grid(ax, "on");
    ax.XMinorTick = "on"; ax.YMinorTick = "on"; ax.ZMinorTick = "on";
    ax.XMinorGrid = "on"; ax.YMinorGrid = "on"; ax.ZMinorGrid = "on";
    ax.MinorGridLineStyle = ":"; ax.MinorGridAlpha = 0.35; ax.GridAlpha = 0.25;

    xlabel(ax, "Pixel intensity", "FontSize", axisFont);
    ylabel(ax, "Class", "FontSize", axisFont);
    zlabel(ax, "Number of pixels", "FontSize", axisFont);

    ylim(ax, [1 numel(classNames)]);
    yticks(ax, 1:numel(classNames));
    yticklabels(ax, classNames);

    title(ax, sprintf("%s channel (3D counts, masked fundus)  [%d..%d]", channelName, xMin, xMax), ...
        "Interpreter","none", "FontSize", titleFont);

    ax.FontSize = axisFont;

    outName = sprintf("overlay3D_%s_%d_%d_full.png", channelName, xMin, xMax);
    exportgraphics(fig, fullfile(outDir, outName), "Resolution", exportRes, "BackgroundColor","white");
    close(fig);
end

function apply_dense_grid_no_warnings(ax, xMin, xMax, maxXlabels)
    % Dense helper lines without warnings:
    % - keep number of labeled major ticks limited (no "too many tick labels" warnings)
    % - turn ON minor grid for denser lines

    grid(ax, "on");
    ax.XMinorTick = "on";
    ax.YMinorTick = "on";
    ax.XMinorGrid = "on";
    ax.YMinorGrid = "on";
    ax.MinorGridLineStyle = ":";
    ax.MinorGridAlpha = 0.35;
    ax.GridAlpha = 0.25;

    span = max(1, xMax - xMin);
    % choose a "nice" major tick step so labels <= maxXlabels
    rawStep = span / max(1, (maxXlabels - 1));
    step = nice_step(rawStep);

    t0 = ceil(xMin/step)*step;
    t1 = floor(xMax/step)*step;
    if t1 >= t0
        xticks(ax, t0:step:t1);
    else
        xticks(ax, [xMin xMax]);
    end
end

function step = nice_step(x)
    % round up to a "nice" step: 1,2,5 * 10^k
    if x <= 0, step = 1; return; end
    k = floor(log10(x));
    b = x / 10^k;
    if b <= 1
        step = 1 * 10^k;
    elseif b <= 2
        step = 2 * 10^k;
    elseif b <= 5
        step = 5 * 10^k;
    else
        step = 10 * 10^k;
    end
    % for small ranges allow step=1 exactly
    if step < 1, step = 1; end
end

function mask = compute_fundus_mask(Id, p)
    Id = force_rgb(Id);
    V = max(Id, [], 3);
    Vsm = imgaussfilt(V, p.gaussSigma);

    nonBlack = Vsm > p.nonBlackThr;
    nonBlack = imclose(nonBlack, strel('disk', p.closeRadius));
    if p.fillHoles, nonBlack = imfill(nonBlack, "holes"); end
    nonBlack = remove_small_and_keep_largest(nonBlack, p.minAreaFrac);

    if p.useEdges
        edgeFilled = build_edge_filled(Vsm, nonBlack, p);
        overlap = nnz(edgeFilled & nonBlack) / max(1, nnz(nonBlack));
        if overlap > 0.60
            mask0 = nonBlack | edgeFilled;
        else
            mask0 = nonBlack;
        end
    else
        mask0 = nonBlack;
    end

    mask = mask0;
    mask = imfill(mask, "holes");
    mask = bwareafilt(mask, 1);

    if p.innerMarginPx > 0
        mask = imerode(mask, strel('disk', p.innerMarginPx));
    end
    if p.erodePixels > 0
        mask = imerode(mask, strel('disk', p.erodePixels));
    end

    mask = imfill(mask, "holes");
    mask = bwareafilt(mask, 1);
end

function I = force_rgb(I)
    if ismatrix(I)
        I = repmat(I, [1 1 3]);
    elseif size(I,3) > 3
        I = I(:,:,1:3);
    end
end

function Iu8 = toUint8_robust(I)
    if isa(I,"uint8"), Iu8 = I; return; end
    if isa(I,"uint16"), Iu8 = uint8(double(I)/65535*255); return; end
    I = double(I);
    mx = max(I(:)); mn = min(I(:));
    if mx <= 255 && mn >= 0
        Iu8 = uint8(round(min(max(I,0),255)));
    else
        In = (I-mn)/max(mx-mn, eps);
        Iu8 = uint8(round(In*255));
    end
end

function bw = remove_small_and_keep_largest(bw, minAreaFrac)
    bw = imfill(bw, "holes");
    bw = bwareaopen(bw, max(1000, round(minAreaFrac * numel(bw))));
    bw = bwareafilt(bw, 1);
    bw = imfill(bw, "holes");
end

function edgeFilled = build_edge_filled(Vsm, nonBlack, p)
    if p.edgeMethod == "canny"
        E = edge(Vsm, "canny", [], p.edgeSigma);
    else
        G = imgradient(Vsm, "sobel");
        Gn = mat2gray(G);
        t = graythresh(Gn);
        t = max(t, 0.05);
        E = (Gn > t);
    end

    E = bwareaopen(E, 200);
    if p.edgeDilate > 0
        E = imdilate(E, strel('disk', p.edgeDilate));
    end
    E = imclose(E, strel('disk', p.edgeClose));

    edgeFilled = imfill(E, "holes");

    nbDil = imdilate(nonBlack, strel('disk', 10));
    edgeFilled = edgeFilled & nbDil;

    edgeFilled = bwareafilt(edgeFilled, 1);
    edgeFilled = imfill(edgeFilled, "holes");
end
