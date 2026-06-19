function run_lbp_extraction(cfg)
%RUN_LBP_EXTRACTION Extract global and local LBP features inside fundus ROI.
%
% Input:
%   cfg - configuration structure created by config_default.m.
%
% Output:
%   Saves LBP feature matrices, image paths, metadata, debug masks and
%   optional publication-style plots to cfg.lbpExtraction.outRoot.

if nargin < 1
    error('Configuration structure cfg is required. Run main.m or call config_default first.');
end

rng(cfg.randomSeed);

%% ===================== SETTINGS FROM CONFIG =====================
OUT_ROOT = cfg.lbpExtraction.outRoot;
BASE_PATHS = string(cfg.lbpExtraction.basePaths);
MAX_PER_CLASS = cfg.lbpExtraction.maxPerClass;
IMG_EXT = string(cfg.imageExtensions);

P_LIST = cfg.lbpExtraction.pList;
params.R = cfg.lbpExtraction.radius;
params.useRiu2 = cfg.lbpExtraction.useRiu2;
GRIDS = cfg.lbpExtraction.grids;

maskParams = cfg.lbpExtraction.maskParams;
seg = cfg.lbpExtraction.seg;

SAVE_FIGS = cfg.lbpExtraction.saveFigs;
SHOW_FIGS = cfg.lbpExtraction.showFigs;
FIG_W = cfg.lbpExtraction.figWidth;
FIG_H = cfg.lbpExtraction.figHeight;
EXPORT_RES = cfg.lbpExtraction.exportResolution;
SAVE_DEBUG_MASKS = cfg.lbpExtraction.saveDebugMasks;
MAX_DEBUG_PER_CLASS = cfg.lbpExtraction.maxDebugPerClass;
SAVE_ORIGINALS = cfg.lbpExtraction.saveOriginals;
SAVE_PATCH_PREVIEW = cfg.lbpExtraction.savePatchPreview;

if ~exist(OUT_ROOT, "dir"), mkdir(OUT_ROOT); end

ensure_dir(OUT_ROOT);

%% ============================================================
% MAIN LOOP OVER BASES
%% ============================================================
for b = 1:numel(BASE_PATHS)
    baseRoot = string(BASE_PATHS(b));
    if ~exist(baseRoot,"dir")
        warning("Base path does not exist: %s (skipping)", baseRoot);
        continue;
    end

    baseName = sanitize_name(get_last_folder_name(baseRoot));
    baseOut  = fullfile(OUT_ROOT, baseName);
    ensure_dir(baseOut);

    maxN = MAX_PER_CLASS(min(b, numel(MAX_PER_CLASS)));

    fprintf("\n==============================\n");
    fprintf("BASE %d/%d: %s\n", b, numel(BASE_PATHS), baseRoot);
    fprintf("Base folder name: %s\n", baseName);
    fprintf("Max images per class: %d\n", maxN);
    fprintf("==============================\n");

    classNames = list_subfolders(baseRoot);
    if isempty(classNames)
        warning("No class subfolders found in base: %s", baseRoot);
        continue;
    end
    fprintf("Detected classes (%d): %s\n", numel(classNames), strjoin(classNames,", "));

    %% ============================================================
    % LOOP OVER LBP P VALUES
    %% ============================================================
    for pIdx = 1:numel(P_LIST)
        params.P = P_LIST(pIdx);
        nbins = get_nbins(params);

        fprintf("\n---- LBP SETTINGS: P=%d, R=%d, riu2=%d (nbins=%d) ----\n", ...
            params.P, params.R, params.useRiu2, nbins);

        % Storage across classes (per grid)
        allGridData = struct();
        for g = 1:numel(GRIDS)
            tag = grid_tag(GRIDS{g});
            allGridData.(tag).X_all = [];
            allGridData.(tag).y_cataract = [];
            allGridData.(tag).classId = [];
            allGridData.(tag).classNames = string(classNames);
            allGridData.(tag).meanHistByClass = struct();
        end

        %% ============================================================
        % LOOP OVER CLASSES
        %% ============================================================
        for c = 1:numel(classNames)
            className = string(classNames{c});
            classDir  = fullfile(baseRoot, className);
            if ~exist(classDir,"dir"), continue; end

            files = list_images(classDir, IMG_EXT);
            if isempty(files)
                warning("No images in: %s", classDir);
                continue;
            end

            % sample
            nAll = numel(files);
            if nAll > maxN
                idx = randperm(nAll, maxN);
                files = files(idx);
                fprintf("  -> %s: sampled %d/%d\n", className, maxN, nAll);
            else
                fprintf("  -> %s: using %d\n", className, nAll);
            end

            classOut = fullfile(baseOut, sanitize_name(className));
            ensure_dir(classOut);

            % Debug dirs once per class
            dbgRoot = fullfile(classOut, "debug_masks_P" + params.P);
            if SAVE_DEBUG_MASKS, ensure_dir(dbgRoot); end
            if SAVE_ORIGINALS && SAVE_DEBUG_MASKS, ensure_dir(fullfile(dbgRoot,"originals")); end

            debugSaved = 0;

            %% ============================================================
            % LOOP OVER PATCH GRIDS
            %% ============================================================
            for g = 1:numel(GRIDS)
                patchGrid = GRIDS{g};
                tag = grid_tag(patchGrid);

                gridOut = fullfile(classOut, ...
                    "lbp_local_" + tag + "_P" + params.P + "_R" + params.R + "_riu2");
                featDir = fullfile(gridOut, "features");
                figDir  = fullfile(gridOut, "figs");

                ensure_dir(gridOut);
                ensure_dir(featDir);
                if SAVE_FIGS, ensure_dir(figDir); end

                nPatches = patchGrid(1)*patchGrid(2);
                featDim  = nPatches * 4 * nbins;

                X = zeros(numel(files), featDim, "double");
                pathsTxt = strings(numel(files),1);

                % class-level mean GLOBAL hist (L1) per channel
                sumR = zeros(1, nbins);
                sumG = zeros(1, nbins);
                sumB = zeros(1, nbins);
                sumY = zeros(1, nbins);
                validCount = 0;

                %% ============================================================
                % LOOP OVER IMAGES
                %% ============================================================
                for i = 1:numel(files)
                    imgPath = string(files{i});
                    pathsTxt(i) = imgPath;

                    try
                        I = imread(imgPath);
                        I = force_rgb(I);

                        % ------- FUNDUS MASK (NEW METHOD) -------
                        [maskFinal, dbg] = fundus_mask_roi(I, maskParams, seg);

                        if nnz(maskFinal) < 1000
                            continue;
                        end

                        % channels
                        Rch = I(:,:,1); Gch = I(:,:,2); Bch = I(:,:,3);
                        Ych = rgb2gray(I);

                        % ------- LOCAL LBP FEATURES -------
                        [featLocal, meta] = lbp_local_features(Rch,Gch,Bch,Ych, maskFinal, params, patchGrid);
                        X(i,:) = featLocal;

                        % ------- GLOBAL LBP HISTS (L1) -------
                        hR = lbp_hist_norm(Rch, maskFinal, params);
                        hG = lbp_hist_norm(Gch, maskFinal, params);
                        hB = lbp_hist_norm(Bch, maskFinal, params);
                        hY = lbp_hist_norm(Ych, maskFinal, params);

                        sumR = sumR + hR(:)'; sumG = sumG + hG(:)';
                        sumB = sumB + hB(:)'; sumY = sumY + hY(:)';
                        validCount = validCount + 1;

                        % save per-image feature
                        [~, stem, ~] = fileparts(char(imgPath));
                        stem = sanitize_name(stem);

                        save(fullfile(featDir, stem + "_lbp_" + tag + "_P" + params.P + ".mat"), ...
                            "featLocal","meta","params","patchGrid","imgPath","hR","hG","hB","hY");

                        % ------- DEBUG EXPORT (pipeline + ROI image used for LBP) -------
                        if SAVE_DEBUG_MASKS && debugSaved < MAX_DEBUG_PER_CLASS && g==1
                            debugSaved = debugSaved + 1;

                            if SAVE_ORIGINALS
                                copyfile(char(imgPath), fullfile(dbgRoot,"originals", stem + "_orig" + get_ext(imgPath)));
                            end

                            outDbg = fullfile(dbgRoot, stem + "_DBG.png");
                            save_debug_pipeline_figure(I, maskFinal, dbg, outDbg, SHOW_FIGS, FIG_W, FIG_H, EXPORT_RES);
                        end

                        % ------- PATCH PREVIEW (1 image/class/grid/P) -------
                        if SAVE_PATCH_PREVIEW && SAVE_FIGS
                            previewPath = fullfile(figDir, ...
                                "PATCH_PREVIEW_" + sanitize_name(className) + "_" + tag + "_P" + params.P + ".png");
                            if ~exist(previewPath,"file")
                                save_patch_preview(I, Ych, maskFinal, meta, params, previewPath, SHOW_FIGS, FIG_W, FIG_H, EXPORT_RES);
                            end
                        end

                    catch ME
                        if mod(i,50)==0
                            fprintf("    failed %d/%d (%s,P=%d): %s\n", i, numel(files), tag, params.P, ME.message);
                        end
                    end
                end

                % save class-level feature matrix
                save(fullfile(gridOut,"X_LBP_LOCAL.mat"), "X","pathsTxt","params","className","patchGrid");
                writelines(pathsTxt, fullfile(gridOut,"paths.txt"));

                % mean global hist per class
                if validCount > 0
                    meanR = sumR/validCount;
                    meanG = sumG/validCount;
                    meanB = sumB/validCount;
                    meanY = sumY/validCount;
                else
                    meanR = zeros(1,nbins); meanG = meanR; meanB = meanR; meanY = meanR;
                end

                if SAVE_FIGS
                    save_class_hist_figure(meanR,meanG,meanB,meanY, nbins, ...
                        baseName, className, tag, params, ...
                        fullfile(figDir, "LBP_meanHist_" + tag + "_P" + params.P + ".png"), ...
                        SHOW_FIGS, FIG_W, FIG_H, EXPORT_RES);
                end

                % store for base-level comparisons
                allGridData.(tag).X_all = [allGridData.(tag).X_all; X];
                allGridData.(tag).classId = [allGridData.(tag).classId; repmat(c, size(X,1), 1)];

                isCat = strcmpi(className,"cataract");
                allGridData.(tag).y_cataract = [allGridData.(tag).y_cataract; repmat(double(isCat), size(X,1), 1)];

                allGridData.(tag).meanHistByClass.(sanitize_name(className)).R    = meanR;
                allGridData.(tag).meanHistByClass.(sanitize_name(className)).G    = meanG;
                allGridData.(tag).meanHistByClass.(sanitize_name(className)).B    = meanB;
                allGridData.(tag).meanHistByClass.(sanitize_name(className)).Gray = meanY;
            end
        end

        %% ============================================================
        % AFTER ALL CLASSES: comparisons + feature importance (per grid)
        %% ============================================================
        for g = 1:numel(GRIDS)
            patchGrid = GRIDS{g};
            tag = grid_tag(patchGrid);

            baseFigOut = fullfile(baseOut, ...
                "BASE_ANALYSIS_" + tag + "_P" + params.P + "_R" + params.R + "_riu2");
            ensure_dir(baseFigOut);

            % (1) ALL CLASSES overlays (same axis range for all)
            save_compare_all_classes(allGridData.(tag), nbins, baseName, tag, params, baseFigOut, ...
                SHOW_FIGS, FIG_W, FIG_H, EXPORT_RES);

            % (2) Cataract vs rest
            save_compare_cataract_vs_rest(allGridData.(tag), nbins, baseName, tag, params, baseFigOut, ...
                SHOW_FIGS, FIG_W, FIG_H, EXPORT_RES);

            % (3) Feature importance on LOCAL features
            run_feature_importance(allGridData.(tag), params, patchGrid, baseName, tag, baseFigOut, FIG_W, FIG_H, EXPORT_RES);
        end
    end
end

fprintf("\nDONE. Results saved in: %s\n", OUT_ROOT);

%% ========================= FUNCTIONS =========================


end

function files = list_images(folder, exts)
folder = char(folder);
d = dir(folder);
files = {};
for k = 1:numel(d)
    if d(k).isdir, continue; end
    [~,~,e] = fileparts(d(k).name);
    if any(strcmpi(string(e), string(exts)))
        files{end+1} = fullfile(folder, d(k).name); %#ok<AGROW>
    end
end
end

function names = list_subfolders(rootDir)
d = dir(rootDir);
d = d([d.isdir]);
names = {};
for i = 1:numel(d)
    nm = d(i).name;
    if strcmp(nm,".") || strcmp(nm,".."), continue; end
    names{end+1} = nm; %#ok<AGROW>
end
names = sort(names);
end

function ensure_dir(p)
if ~exist(p,"dir")
    [ok,msg,msgID] = mkdir(p);
    if ~ok
        error("mkdir failed for '%s': %s (%s)", p, msg, msgID);
    end
end
end

function name = get_last_folder_name(p)
p = string(p);
p = regexprep(p,"[/\\]+$","");
parts = split(p, filesep);
parts = parts(parts ~= "");
last = parts(end);
if strcmpi(last,"dataset") && numel(parts) >= 2
    name = parts(end-1);
else
    name = last;
end
end

function s = sanitize_name(s)
s = string(s);
s = regexprep(s, "[^\w\-]+", "_");
s = regexprep(s, "_+", "_");
s = regexprep(s, "^_+|_+$", "");
if strlength(s)==0, s="unnamed"; end
s = matlab.lang.makeValidName(char(s));
if isempty(s), s="unnamed"; end
s = char(s);
end

function nb = get_nbins(params)
if params.useRiu2
    nb = params.P + 2;
else
    nb = 2^params.P;
end
end

function tag = grid_tag(patchGrid)
tag = sprintf("grid%dx%d", patchGrid(1), patchGrid(2));
end

function I = force_rgb(I)
if ismatrix(I)
    I = repmat(I,[1 1 3]);
elseif size(I,3) > 3
    I = I(:,:,1:3);
end
end

function ext = get_ext(p)
[~,~,ext] = fileparts(char(p));
end

%% ============================================================
% FUNDUS MASK ROI (returns final mask + debug intermediates)
%% ============================================================
function [maskFinal, dbg] = fundus_mask_roi(Iu8, maskParams, seg)
I = im2double(force_rgb(Iu8));
[H,W,~] = size(I);

% robust intensity
V = max(I,[],3);

% nonBlack mask
Vnb = imgaussfilt(V, seg.gaussSigmaNB);
nonBlack = Vnb > seg.nonBlackThr;
nonBlack = imclose(nonBlack, strel("disk", 10, 0));
nonBlack = imfill(nonBlack, "holes");
nonBlack = keep_largest_component(nonBlack, [H W]);

% edges (Sobel or Canny) on smoothed V
Vs = imgaussfilt(V, seg.sigmaSmoothEdges);
if seg.edgeMethod == "canny"
    E = edge(Vs, "canny", [], seg.edgeSigma);
    Gmag = [];
else
    Gmag = imgradient(Vs, "sobel");
    Gn = mat2gray(Gmag);
    t  = graythresh(Gn);
    t  = max(t, 0.05);
    E  = (Gn > t);
end

% restrict edges to near fundus region (avoid corner junk)
E = E & imdilate(nonBlack, strel("disk", 3, 0));
E = bwareaopen(E, 200);

% connect + fill edges => candidate region
E2 = E;
if seg.edgeDilate > 0
    E2 = imdilate(E2, strel("disk", seg.edgeDilate, 0));
end
E2 = imclose(E2, strel("disk", seg.edgeClose, 0));
edgeFilled = imfill(E2, "holes");
edgeFilled = keep_largest_component(edgeFilled, [H W]);
edgeFilled = edgeFilled & nonBlack;
edgeFilled = imfill(edgeFilled, "holes");
edgeFilled = keep_largest_component(edgeFilled, [H W]);

% circle from edges
rMin = max(5, round(seg.radiusFracRange(1)*min(H,W)));
rMax = max(rMin+5, round(seg.radiusFracRange(2)*min(H,W)));

[centers, radii] = imfindcircles(E, [rMin rMax], ...
    "Sensitivity", seg.circleSensitivity, "EdgeThreshold", seg.circleEdgeThr);

if isempty(radii)
    st = regionprops(edgeFilled, "Centroid", "EquivDiameter");
    if isempty(st)
        cx = W/2; cy = H/2; r = 0.45*min(H,W);
    else
        cx = st(1).Centroid(1);
        cy = st(1).Centroid(2);
        r  = 0.5 * st(1).EquivDiameter;
    end
else
    [~, idx] = max(radii);
    cx = centers(idx,1);
    cy = centers(idx,2);
    r  = radii(idx);
end

[X,Y] = meshgrid(1:W, 1:H);
disk = ((X-cx).^2 + (Y-cy).^2) <= r^2;

mask0 = disk & edgeFilled;
mask0 = imfill(mask0, "holes");
mask0 = keep_largest_component(mask0, [H W]);

% cleanup (consistent across all images)
maskFinal = mask0;
maskFinal = imfill(maskFinal, "holes");
maskFinal = keep_largest_component(maskFinal, [H W]);

minA = max(500, round(maskParams.minAreaFrac * numel(maskFinal)));
maskFinal = bwareaopen(maskFinal, minA);
maskFinal = keep_largest_component(maskFinal, [H W]);
maskFinal = imfill(maskFinal, "holes");

% INSIDE shrink to remove black borders / corner residues
if isfield(maskParams,"innerMarginPx") && maskParams.innerMarginPx > 0
    maskFinal = imerode(maskFinal, strel("disk", maskParams.innerMarginPx, 0));
end
if isfield(maskParams,"erodePixels") && maskParams.erodePixels > 0
    maskFinal = imerode(maskFinal, strel("disk", maskParams.erodePixels, 0));
end
maskFinal = imfill(maskFinal, "holes");
maskFinal = keep_largest_component(maskFinal, [H W]);

% debug pack
dbg = struct();
dbg.V = V;
dbg.Vs = Vs;
dbg.nonBlack = nonBlack;
dbg.E = E;
dbg.E2 = E2;
dbg.edgeFilled = edgeFilled;
dbg.disk = disk;
dbg.mask0 = mask0;
dbg.cx = cx; dbg.cy = cy; dbg.r = r;
dbg.rMin = rMin; dbg.rMax = rMax;
dbg.Gmag = Gmag;
dbg.edgeMethod = seg.edgeMethod;
end

function bw = keep_largest_component(bw, imsz)
% deterministic "largest component" without ties warnings:
% tie-breaker: choose component with centroid closest to image center.
if ~any(bw(:))
    return;
end
CC = bwconncomp(bw);
if CC.NumObjects <= 1
    return;
end

stats = regionprops(CC, "Area", "Centroid");
areas = [stats.Area];
maxA = max(areas);
cand = find(areas == maxA);

if numel(cand) == 1
    keepIdx = cand;
else
    cx0 = imsz(2)/2; cy0 = imsz(1)/2;
    d2 = zeros(size(cand));
    for k = 1:numel(cand)
        c = stats(cand(k)).Centroid;
        d2(k) = (c(1)-cx0)^2 + (c(2)-cy0)^2;
    end
    [~,kmin] = min(d2);
    keepIdx = cand(kmin);
end

bw2 = false(size(bw));
bw2(CC.PixelIdxList{keepIdx}) = true;
bw = bw2;
end

%% ============================================================
% DEBUG: pipeline figure (includes ROI image used for LBP)
%% ============================================================
function save_debug_pipeline_figure(Iu8, maskFinal, dbg, outPath, showFigs, figW, figH, exportRes)

I = force_rgb(Iu8);
G = rgb2gray(I);

% ROI images (THIS is what LBP effectively sees)
roiGray = uint8(double(G) .* double(maskFinal));
roiRGB  = I;
for ch = 1:3
    tmp = roiRGB(:,:,ch);
    tmp(~maskFinal) = 0;
    roiRGB(:,:,ch) = tmp;
end

% boundary overlay on gray
bw = bwperim(maskFinal);
overlay = repmat(mat2gray(G), [1 1 3]);
overlay(:,:,1) = max(overlay(:,:,1), bw);

fig = figure("Visible", ternary(showFigs,"on","off"), ...
    "Units","pixels","Position",[30 30 figW figH], "Color","w");

tiledlayout(3,4,"Padding","compact","TileSpacing","compact");

nexttile; imshow(I); title("Original (RGB)","Interpreter","none");
nexttile; imshow(dbg.nonBlack); title("nonBlack mask","Interpreter","none");

if dbg.edgeMethod == "sobel" && ~isempty(dbg.Gmag)
    nexttile; imshow(mat2gray(dbg.Gmag)); title("Sobel |grad|","Interpreter","none");
else
    nexttile; imshow(dbg.Vs,[]); title("Smoothed V (max RGB)","Interpreter","none");
end

nexttile; imshow(dbg.E); title("Edges (thresholded)","Interpreter","none");

nexttile; imshow(dbg.edgeFilled); title("edgeFilled (close+fill)","Interpreter","none");
nexttile; imshow(dbg.disk); title(sprintf("disk (cx=%.1f,cy=%.1f,r=%.1f)", dbg.cx, dbg.cy, dbg.r),"Interpreter","none");
nexttile; imshow(dbg.mask0); title("mask0 = disk ∩ edgeFilled","Interpreter","none");
nexttile; imshow(maskFinal); title("Final mask (used for LBP)","Interpreter","none");

nexttile; imshow(overlay); title("Overlay (final boundary)","Interpreter","none");
nexttile; imshow(roiRGB); title("ROI RGB (masked, used for LBP)","Interpreter","none");
nexttile; imshow(roiGray); title("ROI Gray (masked, used for LBP)","Interpreter","none");

% show bounding circle range used
nexttile;
axis off;
text(0,0.8, sprintf("Circle range: r=[%d..%d] px", dbg.rMin, dbg.rMax), "FontSize",12);
text(0,0.6, sprintf("Edge method: %s", string(dbg.edgeMethod)), "FontSize",12);
text(0,0.4, sprintf("mask pixels: %d", nnz(maskFinal)), "FontSize",12);

exportgraphics(fig, outPath, "Resolution", exportRes, "BackgroundColor","white");
if ~showFigs, close(fig); end
end

%% ============================================================
% LBP features
%% ============================================================
function h = lbp_hist_norm(Ich, mask, params)
counts = lbp_hist_counts(Ich, mask, params);
s = sum(counts);
if s > 0, h = counts / s; else, h = counts; end
end

function [featVec, meta] = lbp_local_features(Rch,Gch,Bch,Ych, mask, params, patchGrid)
nbins = get_nbins(params);
[H,W] = size(mask);

stats = regionprops(mask,'BoundingBox');
if isempty(stats)
    featVec = zeros(1, patchGrid(1)*patchGrid(2)*4*nbins);
    meta = struct("empty",true);
    return;
end
bb = stats(1).BoundingBox;
x0 = max(1, floor(bb(1)));
y0 = max(1, floor(bb(2)));
x1 = min(W, ceil(bb(1)+bb(3)-1));
y1 = min(H, ceil(bb(2)+bb(4)-1));

Gh = patchGrid(1); Gw = patchGrid(2);
patchW = floor((x1-x0+1)/Gw);
patchH = floor((y1-y0+1)/Gh);

featList = zeros(Gh*Gw, 4*nbins);
patchInfo = [];
idx = 0;

for rr = 1:Gh
    for cc = 1:Gw
        idx = idx + 1;
        xs = x0 + (cc-1)*patchW;
        ys = y0 + (rr-1)*patchH;

        if cc < Gw, xe = xs + patchW - 1; else, xe = x1; end
        if rr < Gh, ye = ys + patchH - 1; else, ye = y1; end

        pmask = mask(ys:ye, xs:xe);

        if nnz(pmask) < 200
            featList(idx,:) = zeros(1,4*nbins);
        else
            hR = lbp_hist_norm(Rch(ys:ye,xs:xe), pmask, params);
            hG = lbp_hist_norm(Gch(ys:ye,xs:xe), pmask, params);
            hB = lbp_hist_norm(Bch(ys:ye,xs:xe), pmask, params);
            hY = lbp_hist_norm(Ych(ys:ye,xs:xe), pmask, params);
            featList(idx,:) = [hR(:); hG(:); hB(:); hY(:)]';
        end

        patchInfo = [patchInfo; rr,cc,xs,ys,xe,ye,nnz(pmask)]; %#ok<AGROW>
    end
end

featVec = reshape(featList', 1, []);
meta = struct();
meta.bbox = [x0 y0 x1 y1];
meta.patchGrid = patchGrid;
meta.patchInfo = patchInfo;  % [rr cc xs ys xe ye nPix]
end

function h = lbp_hist_counts(Ich, mask, params)
I = im2double(Ich);
P = params.P; R = params.R;

codes = lbp_codes_circular(I, P, R);

border = ceil(R);
if size(mask,1) <= 2*border || size(mask,2) <= 2*border
    h = zeros(1, get_nbins(params));
    return;
end

mask2 = mask(1+border:end-border, 1+border:end-border);
vals  = codes(mask2);

nbins = get_nbins(params);
if params.useRiu2
    labels = lbp_riu2_map(uint16(vals), P); % 0..P+1
    h = histcounts(double(labels), -0.5:1:(nbins-0.5));
else
    h = histcounts(double(vals), -0.5:1:((2^P)-0.5));
end
h = double(h(:))';
end

function codes = lbp_codes_circular(I, P, R)
[H,W] = size(I);
border = ceil(R);

[Xc,Yc] = meshgrid((1+border):(W-border), (1+border):(H-border));
gc = I((1+border):(H-border), (1+border):(W-border));

codes = zeros(size(gc), "uint16");

for p = 0:(P-1)
    theta = 2*pi*p/P;
    dx = R*cos(theta);
    dy = -R*sin(theta);

    Xn = Xc + dx;
    Yn = Yc + dy;

    gp = interp2(I, Xn, Yn, "linear");
    bit = gp >= gc;

    codes = bitor(codes, uint16(bit) * uint16(2^p));
end
end

function labels = lbp_riu2_map(codes, P)
persistent lutP lut;
if isempty(lut) || isempty(lutP) || lutP ~= P
    lutP = P;
    lut = zeros(2^P,1,"uint16");
    for v = 0:(2^P-1)
        bits = bitget(uint16(v), 1:P);
        trans = sum(bits(1:end-1) ~= bits(2:end)) + (bits(end) ~= bits(1));
        if trans <= 2
            lut(v+1) = uint16(sum(bits));
        else
            lut(v+1) = uint16(P+1);
        end
    end
end
labels = lut(double(codes)+1);
end

%% ============================================================
% PLOTS (HIGH RES, dense grid, identical axis for all classes)
%% ============================================================
function save_class_hist_figure(meanR,meanG,meanB,meanY, nbins, baseName, className, tag, params, outPath, showFigs, figW, figH, exportRes)
fig = figure("Visible", ternary(showFigs,"on","off"), ...
    "Units","pixels","Position",[30 30 figW figH], "Color","w");
tiledlayout(2,2,"Padding","compact","TileSpacing","compact");

x = 0:(nbins-1);

nexttile; bar(x, meanR); grid on; ax = gca; apply_dense_grid(ax, 0, nbins-1, 25);
title("LBP Mean Hist - R"); xlabel("LBP label"); ylabel("Mean (L1)");

nexttile; bar(x, meanG); grid on; ax = gca; apply_dense_grid(ax, 0, nbins-1, 25);
title("LBP Mean Hist - G"); xlabel("LBP label"); ylabel("Mean (L1)");

nexttile; bar(x, meanB); grid on; ax = gca; apply_dense_grid(ax, 0, nbins-1, 25);
title("LBP Mean Hist - B"); xlabel("LBP label"); ylabel("Mean (L1)");

nexttile; bar(x, meanY); grid on; ax = gca; apply_dense_grid(ax, 0, nbins-1, 25);
title("LBP Mean Hist - Gray"); xlabel("LBP label"); ylabel("Mean (L1)");

sgtitle(sprintf("LBP riu2 | P=%d R=%d | Base=%s | Class=%s | %s", params.P, params.R, baseName, className, tag), "Interpreter","none");

exportgraphics(fig, outPath, "Resolution", exportRes, "BackgroundColor","white");
if ~showFigs, close(fig); end
end

function save_compare_all_classes(gridData, nbins, baseName, tag, params, outDir, showFigs, figW, figH, exportRes)
chNames = ["R","G","B","Gray"];
classNames = gridData.classNames;

for ch = 1:numel(chNames)
    CH = chNames(ch);
    M = zeros(numel(classNames), nbins);
    for i = 1:numel(classNames)
        cn = sanitize_name(classNames{i});
        if isfield(gridData.meanHistByClass, cn)
            M(i,:) = gridData.meanHistByClass.(cn).(CH);
        end
    end

    fig = figure("Visible", ternary(showFigs,"on","off"), ...
        "Units","pixels","Position",[30 30 figW figH], "Color","w");
    ax = axes(fig); %#ok<LAXES>
    hold(ax,"on");

    x = 0:(nbins-1);
    cols = lines(size(M,1));
    for i = 1:size(M,1)
        plot(ax, x, M(i,:), "LineWidth", 2.2, "Color", cols(i,:));
    end

    xlim(ax,[0 nbins-1]);
    apply_dense_grid(ax, 0, nbins-1, 25);
    xlabel(ax,"LBP label"); ylabel(ax,"Mean (L1)");
    title(ax, sprintf("ALL CLASSES | CH=%s | %s | %s | P=%d", CH, baseName, tag, params.P), "Interpreter","none");

    lg = legend(ax, string(classNames), "Location","eastoutside", "Interpreter","none");
    lg.Color = "w"; lg.EdgeColor = "k";

    exportgraphics(fig, fullfile(outDir, "COMPARE_ALLCLASSES_" + tag + "_CH_" + CH + "_P" + params.P + ".png"), ...
        "Resolution", exportRes, "BackgroundColor","white");
    if ~showFigs, close(fig); end
end
end

function save_compare_cataract_vs_rest(gridData, nbins, baseName, tag, params, outDir, showFigs, figW, figH, exportRes)
chNames = ["R","G","B","Gray"];
classNames = string(gridData.classNames);

catIdx = find(strcmpi(classNames,"cataract"), 1);
if isempty(catIdx), return; end

for ch = 1:numel(chNames)
    CH = chNames(ch);

    catName = sanitize_name(classNames(catIdx));
    catHist = gridData.meanHistByClass.(catName).(CH);

    restH = zeros(1, nbins);
    k = 0;
    for i = 1:numel(classNames)
        if i==catIdx, continue; end
        cn = sanitize_name(classNames(i));
        if isfield(gridData.meanHistByClass, cn)
            restH = restH + gridData.meanHistByClass.(cn).(CH);
            k = k + 1;
        end
    end
    if k > 0, restH = restH / k; end

    fig = figure("Visible", ternary(showFigs,"on","off"), ...
        "Units","pixels","Position",[30 30 figW figH], "Color","w");
    ax = axes(fig); %#ok<LAXES>
    hold(ax,"on");

    x = 0:(nbins-1);
    plot(ax, x, restH, "LineWidth", 2.8);
    plot(ax, x, catHist, "LineWidth", 2.8);

    xlim(ax,[0 nbins-1]);
    apply_dense_grid(ax, 0, nbins-1, 25);

    xlabel(ax,"LBP label"); ylabel(ax,"Mean (L1)");
    title(ax, sprintf("CATARACT vs REST | CH=%s | %s | %s | P=%d", CH, baseName, tag, params.P), "Interpreter","none");
    lg = legend(ax, ["rest (mean)", "cataract"], "Location","best");
    lg.Color="w"; lg.EdgeColor="k";

    exportgraphics(fig, fullfile(outDir, "COMPARE_CAT_VS_REST_" + tag + "_CH_" + CH + "_P" + params.P + ".png"), ...
        "Resolution", exportRes, "BackgroundColor","white");
    if ~showFigs, close(fig); end
end
end

function apply_dense_grid(ax, xMin, xMax, maxXlabels)
grid(ax,"on");
ax.XMinorTick = "on"; ax.YMinorTick = "on";
ax.XMinorGrid = "on"; ax.YMinorGrid = "on";
ax.MinorGridLineStyle = ":"; ax.MinorGridAlpha = 0.35;
ax.GridAlpha = 0.25;

span = max(1, xMax - xMin);
rawStep = span / max(1,(maxXlabels-1));
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
if x <= 0, step = 1; return; end
k = floor(log10(x));
b = x / 10^k;
if b <= 1
    step = 1*10^k;
elseif b <= 2
    step = 2*10^k;
elseif b <= 5
    step = 5*10^k;
else
    step = 10*10^k;
end
if step < 1, step = 1; end
end

%% ============================================================
% Feature importance (local features)
%% ============================================================
function run_feature_importance(gridData, params, patchGrid, baseName, tag, outDir, figW, figH, exportRes)
X = gridData.X_all;
y = gridData.y_cataract;

if isempty(X) || numel(unique(y)) < 2
    return;
end

% random forest importance
nTrees = 200;
rf = TreeBagger(nTrees, X, y, ...
    'Method','classification', ...
    'OOBPrediction','on', ...
    'OOBPredictorImportance','on');

imp = rf.OOBPermutedPredictorDeltaError(:);

% fisher score
mu1 = mean(X(y==1,:), 1);
mu0 = mean(X(y==0,:), 1);
v1  = var(X(y==1,:), 0, 1);
v0  = var(X(y==0,:), 0, 1);
fisher = (mu1 - mu0).^2 ./ max(eps, (v1 + v0));

[impSorted, idxImp] = sort(imp,'descend');
[fisSorted, idxFis] = sort(fisher(:),'descend');

topK = min(80, size(X,2));
fmap = build_feature_map(params, patchGrid);

rankCol = (1:topK)';

Timp = table(rankCol, idxImp(1:topK), impSorted(1:topK), string(fmap(idxImp(1:topK))), ...
    'VariableNames', {'rank','featureIndex','RF_permImportance','featureDescription'});
Tf = table(rankCol, idxFis(1:topK), fisSorted(1:topK), string(fmap(idxFis(1:topK))), ...
    'VariableNames', {'rank','featureIndex','FisherScore','featureDescription'});

writetable(Timp, fullfile(outDir, "FEATURE_IMPORTANCE_RF_" + tag + "_P" + params.P + ".csv"));
writetable(Tf,   fullfile(outDir, "FEATURE_IMPORTANCE_FISHER_" + tag + "_P" + params.P + ".csv"));

Tmap = table((1:numel(fmap))', string(fmap(:)), 'VariableNames', {'featureIndex','description'});
writetable(Tmap, fullfile(outDir, "FEATURE_MAP_" + tag + "_P" + params.P + ".csv"));

% plots
fig = figure("Visible","off","Units","pixels","Position",[30 30 figW figH],"Color","w");
ax = axes(fig); %#ok<LAXES>
bar(ax, impSorted(1:topK)); grid(ax,"on"); apply_dense_grid(ax, 1, topK, 20);
xlabel(ax,"rank"); ylabel(ax,"RF perm importance (OOB Δerror)");
title(ax, sprintf("RF importance (cataract vs rest) | %s | %s | P=%d", baseName, tag, params.P), "Interpreter","none");
exportgraphics(fig, fullfile(outDir, "FEATURE_IMPORTANCE_RF_" + tag + "_P" + params.P + ".png"), ...
    "Resolution", exportRes, "BackgroundColor","white");
close(fig);

fig = figure("Visible","off","Units","pixels","Position",[30 30 figW figH],"Color","w");
ax = axes(fig); %#ok<LAXES>
bar(ax, fisSorted(1:topK)); grid(ax,"on"); apply_dense_grid(ax, 1, topK, 20);
xlabel(ax,"rank"); ylabel(ax,"Fisher score");
title(ax, sprintf("Fisher score (cataract vs rest) | %s | %s | P=%d", baseName, tag, params.P), "Interpreter","none");
exportgraphics(fig, fullfile(outDir, "FEATURE_IMPORTANCE_FISHER_" + tag + "_P" + params.P + ".png"), ...
    "Resolution", exportRes, "BackgroundColor","white");
close(fig);
end

function desc = build_feature_map(params, patchGrid)
nbins = get_nbins(params);
Gh = patchGrid(1); Gw = patchGrid(2);
nPatches = Gh*Gw;
channels = ["R","G","B","Gray"];

desc = strings(nPatches * numel(channels) * nbins, 1);
k = 0;
for p = 1:nPatches
    [rr,cc] = ind2sub([Gh,Gw], p);
    for ch = 1:numel(channels)
        for b = 0:(nbins-1)
            k = k + 1;
            desc(k) = sprintf("patch(%d,%d) | CH=%s | bin=%d | LBP(P=%d,R=%d,riu2)", ...
                rr, cc, channels(ch), b, params.P, params.R);
        end
    end
end
end

%% ============================================================
% Patch preview (4 sample patches + their LBP hist)
%% ============================================================
function save_patch_preview(Irgb, Ygray, mask, meta, params, outPath, showFigs, figW, figH, exportRes)
if isfield(meta,"empty") && meta.empty, return; end
PI = meta.patchInfo;
if isempty(PI), return; end

[~,ord] = sort(PI(:,7),"descend");
topK = min(4, numel(ord));
sel = ord(1:topK);

fig = figure("Visible", ternary(showFigs,"on","off"), ...
    "Units","pixels","Position",[30 30 figW figH], "Color","w");

t = tiledlayout(3,4,"Padding","compact","TileSpacing","compact");

nexttile(t,[1 4]);
imshow(Irgb);
title(sprintf("Fundus + patch grid | LBP(P=%d,R=%d,riu2)", params.P, params.R), "Interpreter","none");
hold on;

bw = bwperim(mask);
[yy,xx] = find(bw);
plot(xx,yy,'.','MarkerSize',1);

bb = meta.bbox; % [x0 y0 x1 y1]
rectangle('Position',[bb(1), bb(2), bb(3)-bb(1)+1, bb(4)-bb(2)+1], 'LineWidth',2);

Gh = meta.patchGrid(1); Gw = meta.patchGrid(2);
x0 = bb(1); y0 = bb(2); x1 = bb(3); y1 = bb(4);
for c = 1:(Gw-1)
    x = x0 + c * (x1-x0+1)/Gw;
    line([x x],[y0 y1],'LineWidth',1);
end
for r = 1:(Gh-1)
    y = y0 + r * (y1-y0+1)/Gh;
    line([x0 x1],[y y],'LineWidth',1);
end

for k = 1:topK
    row = PI(sel(k),:);
    xs=row(3); ys=row(4); xe=row(5); ye=row(6);
    rectangle('Position',[xs, ys, xe-xs+1, ye-ys+1], 'LineWidth',3);
    text(xs, ys-5, sprintf("#%d", k), "Color","w", "FontWeight","bold");
end
hold off;

for k = 1:topK
    row = PI(sel(k),:);
    rr=row(1); cc=row(2);
    xs=row(3); ys=row(4); xe=row(5); ye=row(6);

    patchGray = Ygray(ys:ye, xs:xe);
    patchMask = mask(ys:ye, xs:xe);

    h = lbp_hist_norm(patchGray, patchMask, params);
    nbins = numel(h);

    nexttile(t);
    imshow(patchGray);
    title(sprintf("Patch (%d,%d)", rr, cc), "Interpreter","none");

    nexttile(t);
    bar(0:(nbins-1), h);
    grid on; ax = gca; apply_dense_grid(ax, 0, nbins-1, 25);
    xlabel("LBP bin"); ylabel("L1");
    title("LBP hist (Gray)", "Interpreter","none");
end

exportgraphics(fig, outPath, "Resolution", exportRes, "BackgroundColor","white");
if ~showFigs, close(fig); end
end

function out = ternary(cond,a,b)
if cond, out=a; else, out=b; end
end
