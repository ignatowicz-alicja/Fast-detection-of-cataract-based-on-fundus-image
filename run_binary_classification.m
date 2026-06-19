function run_binary_classification(cfg)
%RUN_BINARY_CLASSIFICATION Run binary cataract-vs-normal classification.
%
% Supported variants:
%   - LBP features,
%   - histogram features,
%   - merged LBP + histogram features,
%   - raw images with transfer learning.
%
% Supported models:
%   - SVM with RBF kernel,
%   - Random Forest,
%   - ResNet-50 transfer learning,
%   - MobileNetV2 transfer learning.
%
% The function saves metrics, predictions, timing breakdowns and summary
% plots to cfg.binary.outDir.

if nargin < 1
    error('Configuration structure cfg is required. Run main.m or call config_default first.');
end

rng(cfg.randomSeed);

%% ===================== DEVICE INFO =====================
GPU_AVAILABLE = true;
USE_GPU_IF_AVAILABLE = true;
try
    try
        GPU_AVAILABLE = (gpuDeviceCount('available') > 0);
    catch
        GPU_AVAILABLE = (gpuDeviceCount > 0);
    end
catch
    GPU_AVAILABLE = false;
end
if GPU_AVAILABLE && USE_GPU_IF_AVAILABLE
    try, gpuDevice; catch, end
end
fprintf('\n=== DEVICE INFO ===\n');
fprintf('GPU available: %s\n', string(GPU_AVAILABLE));
if GPU_AVAILABLE && USE_GPU_IF_AVAILABLE
    fprintf('Using GPU when supported.\n');
else
    fprintf('Using CPU.\n');
end

%% ===================== SETTINGS FROM CONFIG =====================
LBP_ROOT   = cfg.binary.lbpRoot;
LBP_CONFIG = cfg.binary.lbpConfig;
LBP_MAT    = cfg.binary.lbpMat;

HIST_FROM_IMAGES = false; %#ok<NASGU>
HIST_CSV_1 = cfg.binary.histCsvFiles(1);
if numel(cfg.binary.histCsvFiles) >= 2
    HIST_CSV_2 = cfg.binary.histCsvFiles(2);
else
    HIST_CSV_2 = "";
end

CAT_LABEL    = cfg.labels.cataract;
NORMAL_LABEL = cfg.labels.normal;

IMAGE_ROOT      = cfg.binary.imageRoot;
PATH_REMAP_FROM = cfg.binary.pathRemapFrom;
PATH_REMAP_TO   = cfg.binary.pathRemapTo;

OUT_DIR = cfg.binary.outDir;
if ~exist(OUT_DIR,"dir"), mkdir(OUT_DIR); end

K       = cfg.binary.kFolds;
HOLDOUT = cfg.binary.holdout;
BOXC    = cfg.binary.boxConstraint;
DL_DO_CV = cfg.binary.dlDoCV;

RUN_LBP      = cfg.binary.runLBP;
RUN_HIST     = cfg.binary.runHIST;
RUN_LBP_HIST = cfg.binary.runLBPHIST;
RUN_RAW_DL   = cfg.binary.runRawDL;

CLASSIC_MODELS = string(cfg.binary.classicModels);
DL_MODELS      = string(cfg.binary.dlModels);

CNN_INPUT_SIZE   = cfg.binary.cnnInputSize;
CNN_MINIBATCH    = cfg.binary.cnnMiniBatch;
CNN_MAX_EPOCHS   = cfg.binary.cnnMaxEpochs;
CNN_LEARN_RATE   = cfg.binary.cnnLearnRate;
CNN_FREEZE_BACKBONE = cfg.binary.cnnFreezeBackbone;

FUNDUS_ROI = cfg.roi;
PUB = cfg.pub;
set_matplotlib_defaults(PUB);

Master = table();

%% ===================== 1) LOAD LBP =====================
AllLbp = table();
if RUN_LBP || RUN_LBP_HIST || RUN_RAW_DL
    AllLbp = load_lbp_two_classes(LBP_ROOT, LBP_CONFIG, LBP_MAT, CAT_LABEL, NORMAL_LABEL);
    if ~isempty(AllLbp)
        writetable(AllLbp, fullfile(OUT_DIR, "LBP_TWO_CLASSES_CLEAN.csv"));
    end
end

%% ===================== 2) LOAD HIST =====================
AllHist = table();
if RUN_HIST || RUN_LBP_HIST
    fprintf("Wczytywanie cech HIST z dostarczonego pliku CSV...\n");
    AllHist = load_hist_two_classes([HIST_CSV_1 HIST_CSV_2], CAT_LABEL, NORMAL_LABEL);
    if ~isempty(AllHist)
        writetable(AllHist, fullfile(OUT_DIR, "HIST_TWO_CLASSES.csv"));
    else
        fprintf("NOTE: No HIST loaded -> HIST and LBP+HIST will be skipped.\n");
        RUN_HIST = false; RUN_LBP_HIST = false;
    end
end

%% ===================== 3) BUILD VARIANTS =====================
Variants = {};
if RUN_LBP
    Variants{end+1} = make_variant_binary("LBP_BALANCED", AllLbp, [], "lbp", CAT_LABEL, NORMAL_LABEL, true);
end
if RUN_HIST && ~isempty(AllHist)
    Variants{end+1} = make_variant_binary("HIST_BALANCED", [], AllHist, "hist", CAT_LABEL, NORMAL_LABEL, true);
end
if RUN_LBP_HIST && ~isempty(AllLbp) && ~isempty(AllHist)
    Variants{end+1} = make_variant_binary("LBP_HIST_BALANCED", AllLbp, AllHist, "merge", CAT_LABEL, NORMAL_LABEL, true);
end
if RUN_RAW_DL
    if ~isempty(AllLbp) && all(ismember({'Filename','Label'}, AllLbp.Properties.VariableNames))
        Traw = AllLbp(:, {'Filename','Label'});
    elseif ~isempty(AllHist) && all(ismember({'Filename','Label'}, AllHist.Properties.VariableNames))
        Traw = AllHist(:, {'Filename','Label'});
    else
        Traw = table();
    end
    Vraw = make_variant_binary("RAW_IMAGES_BALANCED", Traw, [], "raw", CAT_LABEL, NORMAL_LABEL, true);
    Variants{end+1} = Vraw;
end

%% ===================== 4) RUN ALL VARIANTS =====================
for vi = 1:numel(Variants)
    V = Variants{vi};
    if isempty(V) || isempty(V.Y), continue; end

    tag = V.tag;
    outV = fullfile(OUT_DIR, tag);
    if ~exist(outV,"dir"), mkdir(outV); end

    if any(strcmp(V.T.Properties.VariableNames,"Filename"))
        if any(strcmp(V.T.Properties.VariableNames,"Label"))
            V.T.Filename = resolve_image_paths2(string(V.T.Filename), string(V.T.Label), IMAGE_ROOT, PATH_REMAP_FROM, PATH_REMAP_TO);
        else
            V.T.Filename = resolve_image_paths2(string(V.T.Filename), strings(height(V.T),1), IMAGE_ROOT, PATH_REMAP_FROM, PATH_REMAP_TO);
        end
    end
    writetable(V.T, fullfile(outV, "dataset_used.csv"));

    fprintf("\n=== VARIANT: %s ===\n", tag);
    fprintf("Samples: %d | cataract=%d normal=%d | features=%d\n", numel(V.Y), sum(V.Y=="cataract"), sum(V.Y=="normal"), size(V.X,2));

    cvH = cvpartition(V.Y, "HoldOut", HOLDOUT);
    tr = training(cvH); te = test(cvH);

    isRaw = strcmpi(V.mode, "raw");
    if isRaw
        modelsToRun = DL_MODELS;
    else
        modelsToRun = CLASSIC_MODELS;
    end

    Res = table();
    for mi = 1:numel(modelsToRun)
        modelName = modelsToRun(mi);

        if isRaw && ~DL_DO_CV
            accCV = NaN; f1CV = NaN; aucCV = NaN; prAucCV = NaN; tpsCV = NaN;
        else
            [accCV, f1CV, aucCV, prAucCV, tpsCV] = eval_model_cv_binary(modelName, V, K, BOXC, ...
                CNN_INPUT_SIZE, CNN_MINIBATCH, CNN_MAX_EPOCHS, CNN_LEARN_RATE, CNN_FREEZE_BACKBONE, ...
                "none", [224 224], 16, 10, 1e-4, ...
                FUNDUS_ROI, PUB);
        end

        timingOpts = struct();
        timingOpts.GPU_AVAILABLE = GPU_AVAILABLE;
        timingOpts.USE_GPU_IF_AVAILABLE = USE_GPU_IF_AVAILABLE;
        timingOpts.UseLBP  = ~isRaw && (strcmpi(V.mode,'lbp') || strcmpi(V.mode,'merge'));
        timingOpts.UseHIST = ~isRaw && (strcmpi(V.mode,'hist') || strcmpi(V.mode,'merge'));
        timingOpts.LbpSpec = parse_lbp_config(LBP_CONFIG);
        timingOpts.HistSpec = infer_hist_spec_from_table(V.T);
        timingOpts.HistNormalizeL1 = false;

        [Mtest, predTable] = eval_model_holdout_and_save_binary_timed(modelName, V, tr, te, BOXC, outV, PUB, ...
            CNN_INPUT_SIZE, CNN_MINIBATCH, CNN_MAX_EPOCHS, CNN_LEARN_RATE, CNN_FREEZE_BACKBONE, ...
            "none", [224 224], 16, 10, 1e-4, ...
            FUNDUS_ROI, timingOpts);

        writetable(predTable, fullfile(outV, modelName + "_predictions.csv"));

        row = table(modelName, ...
            mean(accCV,"omitnan"), std(accCV,[],'omitnan'), ...
            mean(f1CV,"omitnan"),  std(f1CV,[],'omitnan'), ...
            mean(aucCV,"omitnan"), std(aucCV,[],'omitnan'), ...
            mean(prAucCV,"omitnan"), std(prAucCV,[],'omitnan'), ...
            mean(tpsCV,"omitnan"), std(tpsCV,[],'omitnan'), ...
            Mtest.Accuracy, Mtest.Precision, Mtest.Recall, Mtest.Specificity, Mtest.F1, Mtest.AUC_ROC, Mtest.AUC_PR, ...
            Mtest.TimeTotalMs, Mtest.TimePerSampleMs, ...
            'VariableNames', ["Model",...
            "AccCV_Mean","AccCV_Std","F1CV_Mean","F1CV_Std","AUCCV_Mean","AUCCV_Std","PRAUCCV_Mean","PRAUCCV_Std",...
            "TimePerSampleCV_MeanMs","TimePerSampleCV_StdMs",...
            "AccTest","PrecTest","RecTest","SpecTest","F1Test","AUCTest","PRAUCTest","TimeTotalTestMs","TimePerSampleTestMs"]);

        Res = [Res; row]; %#ok<AGROW>

        row2 = row;
        row2 = addvars(row2, string(tag), 'Before', 1, 'NewVariableNames',"Variant");
        row2 = addvars(row2, height(V.T), sum(V.Y=="cataract"), sum(V.Y=="normal"), 'After',"Variant", ...
            'NewVariableNames', ["N_Total","N_Cataract","N_Normal"]);
        Master = [Master; row2]; %#ok<AGROW>

        fprintf("%s | Test AUC=%.4f | Test Acc=%.4f | Test F1=%.4f | %.3f ms/sample\n", ...
            modelName, Mtest.AUC_ROC, Mtest.Accuracy, Mtest.F1, Mtest.TimePerSampleMs);
    end

    Res = sortrows(Res, ["AUCTest","F1Test"], "descend");
    writetable(Res, fullfile(outV, "summary_metrics.csv"));
    make_variant_summary_plot_python_like(Res, outV, tag, PUB);
end

%% ===================== 5) EXPORT MASTER TO EXCEL =====================
if isempty(Master)
    warning("No results were produced. Check input paths and run flags in config_default.m.");
    return;
end
Master = sortrows(Master, ["Variant","AUCTest"], "ascend");
excelPath = fullfile(OUT_DIR, "MASTER_METRICS.xlsx");
writetable(Master, excelPath, "Sheet","Master");
writetable(Master, fullfile(OUT_DIR, "MASTER_METRICS.csv"));
fprintf("\nDONE. All results saved in:\n  %s\n", OUT_DIR);

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================


end

function set_matplotlib_defaults(PUB)
    set(groot,'defaultFigureColor','w');
    set(groot,'defaultAxesFontName', char(PUB.FontName));
    set(groot,'defaultTextFontName', char(PUB.FontName));
    set(groot,'defaultAxesFontSize', PUB.FontSizeAxes);
    set(groot,'defaultTextFontSize', PUB.FontSizeAxes);
    set(groot,'defaultAxesLineWidth', 1.2);
    set(groot,'defaultLineLineWidth', PUB.LineWidth);
    set(groot,'defaultAxesBox','on');
    set(groot,'defaultAxesXGrid','on');
    set(groot,'defaultAxesYGrid','on');
    set(groot,'defaultAxesGridAlpha', 0.20);
end

function T = load_lbp_two_classes(LBP_ROOT, LBP_CONFIG, MAT_NAME, CAT_LABEL, NORMAL_LABEL)
    wanted = lower(string([CAT_LABEL NORMAL_LABEL]));
    T = table();
    for w = 1:numel(wanted)
        cls = wanted(w); matPath = fullfile(LBP_ROOT, cls, LBP_CONFIG, MAT_NAME);
        if ~exist(matPath,"file"), continue; end
        S = load(matPath, "X", "pathsTxt");
        if ~isfield(S,"X") || ~isfield(S,"pathsTxt"), continue; end
        X = S.X; P = string(S.pathsTxt);
        if isempty(X) || size(X,1) ~= numel(P), continue; end
        valid = any(X ~= 0, 2) & all(isfinite(X), 2);
        Xc = X(valid,:); Pc = P(valid);
        featNames = "lbp_" + string(1:size(Xc,2));
        Tfeat = array2table(Xc, "VariableNames", cellstr(featNames));
        Ttmp = table(); Ttmp.Filename = Pc; Ttmp.Label = repmat(string(cls), numel(Pc), 1);
        Ttmp = [Ttmp Tfeat]; T = [T; Ttmp]; %#ok<AGROW>
    end
end

function Th = load_hist_two_classes(csvList, CAT_LABEL, NORMAL_LABEL)
    Th = table(); csvList = string(csvList);
    for i = 1:numel(csvList)
        p = csvList(i); if strlength(p)==0, continue; end
        if ~exist(p,"file"), continue; end
        T = readtable(p); Th = [Th; T]; %#ok<AGROW>
    end
    if isempty(Th), return; end

    if any(strcmp(Th.Properties.VariableNames, "Path")) && ~any(strcmp(Th.Properties.VariableNames, "Filename"))
        Th.Properties.VariableNames{strcmp(Th.Properties.VariableNames, 'Path')} = 'Filename';
    end

    if ~any(strcmp(Th.Properties.VariableNames,"Label"))
        vars = string(Th.Properties.VariableNames);
        cand = vars(ismember(lower(vars), ["label","class","classname","category","y","target"]));
        if ~isempty(cand)
            Th.Properties.VariableNames{vars==cand(1)} = 'Label';
        end
    end

    if ~any(strcmp(Th.Properties.VariableNames,"Label")), error("HIST table missing Label"); end
    if ~any(strcmp(Th.Properties.VariableNames,"Filename")), error("HIST table missing Filename"); end

    Th.Label = lower(strtrim(string(Th.Label))); Th.Filename = string(Th.Filename);
    wanted = lower(string([CAT_LABEL NORMAL_LABEL]));
    keep = ismember(Th.Label, wanted); Th = Th(keep,:);
end

function V = make_variant_binary(tag, Tlbp, Thist, mode, CAT_LABEL, NORMAL_LABEL, matchByBasename)
    V = struct(); V.tag = tag; V.mode = char(mode);
    V.X = []; V.Y = categorical(); V.W = []; V.T = table();

    mode = lower(string(mode));
    switch mode
        case "lbp",  T = Tlbp;
        case "hist", T = Thist;
        case "merge"
            [T, ok] = merge_lbp_hist_tables(Tlbp, Thist, matchByBasename);
            if ~ok, V = []; return; end
        case "raw"
            if ~isempty(Tlbp) && istable(Tlbp), T = Tlbp;
            elseif ~isempty(Thist) && istable(Thist), T = Thist;
            else, V = []; return;
            end
        otherwise
            error("Unknown mode: %s", mode);
    end

    if isempty(T) || ~istable(T) || width(T)==0, V = []; return; end
    T = normalize_label_filename(T);

    if ~any(strcmp(T.Properties.VariableNames,'Label')) || ~any(strcmp(T.Properties.VariableNames,'Filename'))
        V = []; return;
    end

    T = filter_two_classes(T, CAT_LABEL, NORMAL_LABEL);
    if isempty(T), V=[]; return; end
    T = balance_binary_equal(T, CAT_LABEL, NORMAL_LABEL);
    if isempty(T), V=[]; return; end

    Y = categorical(lower(strtrim(string(T.Label))), [lower(CAT_LABEL) lower(NORMAL_LABEL)], ["cataract","normal"]);

    if mode == "raw"
        V.T = T(:, intersect(["Filename","Label"], string(T.Properties.VariableNames), 'stable'));
        V.Y = Y;
        W = ones(size(Y)); nPos = sum(Y=="cataract"); nNeg = sum(Y=="normal");
        if nPos>0 && nNeg>0, W(Y=="cataract")=0.5/nPos; W(Y=="normal")=0.5/nNeg; end
        V.W = W;
        return;
    end

    vars = string(T.Properties.VariableNames); drop = intersect(vars, ["Label","Filename"]);
    Tf = T; Tf(:, drop) = [];
    isNum = varfun(@isnumeric, Tf, "OutputFormat","uniform");
    X = table2array(Tf(:,isNum));

    ok = all(isfinite(X),2);
    X = X(ok,:); Y = Y(ok); T = T(ok,:);

    W = ones(size(Y)); nPos = sum(Y=="cataract"); nNeg = sum(Y=="normal");
    if nPos>0 && nNeg>0, W(Y=="cataract")=0.5/nPos; W(Y=="normal")=0.5/nNeg; end

    V.X = X; V.Y = Y; V.W = W; V.T = T;
end

function T = normalize_label_filename(T)
    if isempty(T) || ~istable(T) || width(T)==0, return; end
    vars = string(T.Properties.VariableNames);

    if ~any(vars=="Label")
        cand = vars(ismember(lower(vars), ["label","class","classname","category","y","target"]));
        if ~isempty(cand)
            T.Properties.VariableNames{vars==cand(1)} = 'Label';
            vars = string(T.Properties.VariableNames);
        end
    end
    if ~any(vars=="Filename")
        if any(vars=="Path")
            T.Properties.VariableNames{vars=="Path"} = 'Filename';
            vars = string(T.Properties.VariableNames);
        else
            cand = vars(ismember(lower(vars), ["file","filepath","image","imgpath","fullpath"]));
            if ~isempty(cand)
                T.Properties.VariableNames{vars==cand(1)} = 'Filename';
                vars = string(T.Properties.VariableNames);
            end
        end
    end

    if any(vars=="Label"), T.Label = lower(strtrim(string(T.Label))); end
    if any(vars=="Filename"), T.Filename = string(T.Filename); end
end

function T = filter_two_classes(T, CAT_LABEL, NORMAL_LABEL)
    keep = ismember(lower(strtrim(string(T.Label))), lower(string([CAT_LABEL NORMAL_LABEL])));
    T = T(keep,:);
end

function Tout = balance_binary_equal(T, CAT_LABEL, NORMAL_LABEL)
    labels = lower(strtrim(string(T.Label)));
    isCat  = labels == lower(CAT_LABEL); isNorm = labels == lower(NORMAL_LABEL);
    Tcat  = T(isCat,:); Tnorm = T(isNorm,:);
    n = min(height(Tcat), height(Tnorm));
    Tout = [Tcat(randperm(height(Tcat), n),:); Tnorm(randperm(height(Tnorm), n),:)];
    Tout = Tout(randperm(height(Tout)), :);
end

function [T, ok] = merge_lbp_hist_tables(Tlbp, Thist, byBaseName)
    ok = false; T = table(); if isempty(Tlbp) || isempty(Thist), return; end
    A = normalize_label_filename(Tlbp); B = normalize_label_filename(Thist);
    A.JoinKey = make_join_key(string(A.Filename), byBaseName);
    B.JoinKey = make_join_key(string(B.Filename), byBaseName);

    varsB = string(B.Properties.VariableNames); dropB = intersect(varsB, ["Label","Filename"]);
    Bfeat = B; Bfeat(:, dropB) = [];
    numVars = Bfeat.Properties.VariableNames(varfun(@isnumeric, Bfeat, "OutputFormat","uniform"));
    if isempty(numVars), return; end

    G = groupsummary(B, "JoinKey", "mean", numVars);
    for i = 1:numel(G.Properties.VariableNames)
        vn = string(G.Properties.VariableNames{i});
        if startsWith(vn, "mean_"), G.Properties.VariableNames{i} = char("hist_" + extractAfter(vn, "mean_")); end
    end
    [~, ia] = unique(B.JoinKey, "stable"); G.Label = B.Label(ia);
    J = outerjoin(A, G, "Keys","JoinKey", "MergeKeys", true);
    J = unify_join_columns(J);

    vars = string(J.Properties.VariableNames); drop = intersect(vars, ["JoinKey","Label","Filename"]);
    Jtmp = J; Jtmp(:, drop) = [];
    numCols = Jtmp.Properties.VariableNames(varfun(@isnumeric, Jtmp, "OutputFormat","uniform"));
    for i = 1:numel(numCols)
        v = J.(numCols{i}); if any(isnan(v)), v(isnan(v)) = 0; J.(numCols{i}) = v; end
    end
    featVars = setdiff(string(J.Properties.VariableNames), ["Filename","Label","JoinKey"], "stable");
    T = J(:, ["Filename","Label", featVars]); ok = true;
end

function J = unify_join_columns(J)
    vars = string(J.Properties.VariableNames);
    if ~any(vars=="Filename")
        cands = vars(startsWith(vars,"Filename"));
        if ~isempty(cands)
            f = string(J.(cands(1)));
            for k = 2:numel(cands)
                g = string(J.(cands(k))); miss = strlength(f)==0; f(miss) = g(miss);
            end
            J.Filename = f;
        end
    end
    if ~any(vars=="Label")
        cands = vars(startsWith(vars,"Label"));
        if ~isempty(cands)
            lab = string(J.(cands(1)));
            for k = 2:numel(cands)
                g = string(J.(cands(k))); miss = strlength(lab)==0; lab(miss) = g(miss);
            end
            J.Label = lower(strtrim(lab));
        end
    else
        J.Label = lower(strtrim(string(J.Label)));
    end
end

function key = make_join_key(paths, byBaseName)
    paths = string(paths); key = strings(numel(paths),1);
    for i = 1:numel(paths)
        p = paths(i);
        if byBaseName, [~, name, ext] = fileparts(char(p)); s = string(name) + string(ext); else, s = p; end
        [~, nameNoExt, ~] = fileparts(char(s)); s = lower(string(nameNoExt)); s = regexprep(s, "\s+", "");
        s = regexprep(s, "(_patch.*)$", ""); s = regexprep(s, "(_roi.*)$", "");
        s = regexprep(s, "(_crop.*)$", "");  s = regexprep(s, "(_aug.*)$", "");
        s = regexprep(s, "(_lbp.*)$", "");   s = regexprep(s, "(_hist.*)$", "");
        s = regexprep(s, "(_x\d+_y\d+.*)$", ""); key(i) = s;
    end
end

function filesOut = resolve_image_paths2(filesIn, labelsIn, imageRoot, remapFrom, remapTo)
    filesOut = string(filesIn); labelsIn = string(labelsIn);
    for i = 1:numel(filesOut)
        p = filesOut(i); if exist(p,"file")==2, continue; end
        if strlength(remapFrom) > 0 && strlength(remapTo) > 0
            if startsWith(p, remapFrom, "IgnoreCase", true)
                cand = remapTo + extractAfter(p, strlength(remapFrom));
                if exist(cand,"file")==2, filesOut(i) = cand; continue; end
            end
        end
        if strlength(imageRoot) == 0, continue; end
        [~, name, ext] = fileparts(char(p)); base = string([name ext]);
        lab = ""; if i <= numel(labelsIn), lab = lower(strtrim(labelsIn(i))); end
        if strlength(lab) > 0
            cand1 = fullfile(imageRoot, lab, base);
            if exist(cand1,"file")==2, filesOut(i) = string(cand1); continue; end
        end
        cand2 = fullfile(imageRoot, base);
        if exist(cand2,"file")==2, filesOut(i) = string(cand2); continue; end
    end
end

function [accCV, f1CV, aucCV, prAucCV, tPerSampleMsCV] = eval_model_cv_binary(modelName, V, K, BOXC, ...
    cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, vitModelName, vitInput, vitMb, vitEpochs, vitLR, fundusROI, PUB)
    cv = cvpartition(V.Y, "KFold", K);
    accCV = zeros(K,1); f1CV = zeros(K,1); aucCV = zeros(K,1); prAucCV = zeros(K,1); tPerSampleMsCV = nan(K,1);
    for fold = 1:K
        tr = training(cv, fold); te = test(cv, fold); Wtr = V.W(tr);
        [M, ~] = train_predict_once_binary(modelName, V, tr, te, Wtr, BOXC, ...
            cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, vitModelName, vitInput, vitMb, vitEpochs, vitLR, ...
            fundusROI, PUB, 'SaveArtifacts', false);
        accCV(fold) = M.Accuracy; f1CV(fold) = M.F1; aucCV(fold) = M.AUC_ROC;
        prAucCV(fold) = M.AUC_PR; tPerSampleMsCV(fold) = M.TimePerSampleMs;
    end
end

function [Mtest, predTable] = eval_model_holdout_and_save_binary_timed(modelName, V, tr, te, BOXC, outDir, PUB, ...
    cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, vitModelName, vitInput, vitMb, vitEpochs, vitLR, fundusROI, timingOpts)

    Wtr = V.W(tr);
    [Mtest, artifacts, timingTbl] = train_predict_once_binary_timed(modelName, V, tr, te, Wtr, BOXC, ...
        cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, vitModelName, vitInput, vitMb, vitEpochs, vitLR, ...
        fundusROI, PUB, timingOpts);

    predTable = table();
    predTable.Filename = artifacts.FilesTe;
    predTable.TrueLabel = artifacts.Ytrue;
    predTable.PredLabel = artifacts.Ypred;
    predTable.ScoreCataract = artifacts.PosScore;

    if ~isempty(timingTbl)
        writetable(timingTbl, fullfile(outDir, modelName + "_timing_breakdown.csv"));
    end
end

function [M, artifacts, timingTbl] = train_predict_once_binary_timed(modelName, V, tr, te, Wtr, BOXC, ...
    cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, vitModelName, vitInput, vitMb, vitEpochs, vitLR, fundusROI, PUB, timingOpts)

    X = V.X; Y = V.Y;
    Ytr = Y(tr); Yte = Y(te);

    artifacts = struct();
    artifacts.FilesTe = strings(nnz(te),1);
    if any(strcmp(V.T.Properties.VariableNames,'Filename'))
        artifacts.FilesTe = string(V.T.Filename(te));
    end
    artifacts.Ytrue = Yte;
    artifacts.Ypred = Yte;
    artifacts.PosScore = zeros(nnz(te),1);

    timingTbl = table();

    switch string(modelName)
        case "SVM_RBF"
            Xtr = X(tr,:); Xte = X(te,:);

            [tRoiMs, tLbpMs, tHistMs, ~] = time_feature_stages_on_test(artifacts.FilesTe, fundusROI, timingOpts);

            t0 = tic; XtrH = sqrt(abs(Xtr)); XteH = sqrt(abs(Xte)); tHell = toc(t0);
            t0 = tic;
            mu = mean(XtrH,1,'omitnan'); sd = std(XtrH,0,1,'omitnan'); sd(sd<1e-12)=1;
            XtrZ = (XtrH-mu)./sd; XteZ = (XteH-mu)./sd;
            tStd = toc(t0);

            tPca = 0;
            if size(XtrZ,2) > 100
                t0 = tic;
                [coeff, scoreTr, ~, ~, explained] = pca(XtrZ);
                numComp = find(cumsum(explained) >= 95, 1);
                if isempty(numComp), numComp = size(scoreTr,2); end
                XtrZ = scoreTr(:,1:numComp);
                XteZ = XteZ * coeff(:,1:numComp);
                tPca = toc(t0);
            end

            opts_svm = struct('Optimizer','bayesopt','ShowPlots',false, ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'MaxObjectiveEvaluations',30);

            mdl = fitcsvm(XtrZ, Ytr, 'KernelFunction','rbf', ...
                'OptimizeHyperparameters','auto', ...
                'HyperparameterOptimizationOptions', opts_svm, ...
                'Standardize', false, ...
                'ClassNames', categorical({'cataract','normal'}), ...
                'Weights', Wtr);

            t0 = tic; [Yhat, sc] = predict(mdl, XteZ); dt = toc(t0);
            posScore = sc(:,1);

            nTe = max(1,numel(Yte));
            timingTbl = make_timing_table(modelName, nTe, tRoiMs, tLbpMs, tHistMs, (tHell+tStd+tPca)*1000/nTe, dt*1000/nTe);

        case "RF"
            Xtr = X(tr,:); Xte = X(te,:);
            [tRoiMs, tLbpMs, tHistMs, ~] = time_feature_stages_on_test(artifacts.FilesTe, fundusROI, timingOpts);

            mdl = TreeBagger(500, Xtr, Ytr, 'Method','classification', 'MinLeafSize',5, 'OOBPrediction','off', 'Weights', Wtr);
            t0 = tic; [YhatRaw, scoreRaw] = predict(mdl, Xte); dt = toc(t0);
            Yhat = categorical(string(YhatRaw));
            idxPos = find(lower(string(mdl.ClassNames)) == 'cataract', 1);
            if isempty(idxPos), idxPos = 1; end
            posScore = scoreRaw(:,idxPos);

            nTe = max(1,numel(Yte));
            timingTbl = make_timing_table(modelName, nTe, tRoiMs, tLbpMs, tHistMs, 0, dt*1000/nTe);

        case "RESNET50_TL"
            [Yhat, posScore, dtFwd, info, trainedNet, filesTe] = predict_cnn_tl('resnet50', V, tr, te, cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, fundusROI, timingOpts);
            [tRoiMs, tPreMs, tFwdMs] = time_cnn_stages(trainedNet, filesTe, cnnInput, fundusROI, timingOpts);
            timingTbl = make_timing_table_dl(modelName, numel(filesTe), tRoiMs, tPreMs, tFwdMs);
            dt = dtFwd;

        case "MOBILENETV2_TL"
            [Yhat, posScore, dtFwd, info, trainedNet, filesTe] = predict_cnn_tl('mobilenetv2', V, tr, te, cnnInput, cnnMb, cnnEpochs, cnnLR, cnnFreeze, fundusROI, timingOpts);
            [tRoiMs, tPreMs, tFwdMs] = time_cnn_stages(trainedNet, filesTe, cnnInput, fundusROI, timingOpts);
            timingTbl = make_timing_table_dl(modelName, numel(filesTe), tRoiMs, tPreMs, tFwdMs);
            dt = dtFwd;

        otherwise
            error("Unknown model: %s", modelName);
    end

    % Metrics
    cm = confusionmat(Yte, Yhat, 'Order', categorical({'cataract','normal'}));
    TP = cm(1,1); FN = cm(1,2); FP = cm(2,1); TN = cm(2,2);
    acc  = (TP+TN)/max(1,sum(cm(:)));
    prec = TP/max(1,(TP+FP));
    rec  = TP/max(1,(TP+FN));
    spec = TN/max(1,(TN+FP));
    f1   = 2*prec*rec/max(1e-12,(prec+rec));

    [~, ~, ~, auc] = perfcurve(Yte, posScore, 'cataract');
    [~, ~, ~, prAuc] = perfcurve(Yte, posScore, 'cataract', 'xCrit','reca','yCrit','prec');

    artifacts.Ypred = Yhat;
    artifacts.PosScore = posScore;

    timePerSampleMs = 0;
    timeTotalMs = 0;
    if ~isempty(timingTbl) && any(strcmp(timingTbl.Properties.VariableNames,'TotalMsPerSample'))
        timePerSampleMs = timingTbl.TotalMsPerSample(1);
        timeTotalMs = timePerSampleMs * max(1,numel(Yte));
    end

    M = struct();
    M.Accuracy = acc; M.Precision = prec; M.Recall = rec; M.Specificity = spec;
    M.F1 = f1; M.AUC_ROC = auc; M.AUC_PR = prAuc;
    M.TimeTotalMs = timeTotalMs; M.TimePerSampleMs = timePerSampleMs;
end

function [tRoiMs, tLbpMs, tHistMs, nOk] = time_feature_stages_on_test(filesTe, fundusROI, timingOpts)
    filesTe = string(filesTe);
    n = numel(filesTe);
    tRoi = 0; tLbp = 0; tHist = 0; nOk = 0;
    for i = 1:n
        fn = filesTe(i);
        if exist(fn,'file') ~= 2, continue; end
        nOk = nOk + 1;

        t0 = tic; [Iroi, mask] = read_fundus_roi_with_mask(fn, fundusROI); tRoi = tRoi + toc(t0); %#ok<NASGU>
        Iu8 = im2uint8(normalize01(Iroi));

        if isfield(timingOpts,'UseLBP') && timingOpts.UseLBP
            t0 = tic; extract_lbp_local_riu2(Iu8, mask, timingOpts.LbpSpec); tLbp = tLbp + toc(t0);
        end
        if isfield(timingOpts,'UseHIST') && timingOpts.UseHIST
            t0 = tic; extract_hist_ranges(Iu8, mask, timingOpts.HistSpec, timingOpts.HistNormalizeL1); tHist = tHist + toc(t0);
        end
    end
    if nOk == 0, tRoiMs=0; tLbpMs=0; tHistMs=0; return; end
    tRoiMs  = (tRoi  / nOk) * 1000;
    tLbpMs  = (tLbp  / nOk) * 1000;
    tHistMs = (tHist / nOk) * 1000;
end

function T = make_timing_table(modelName, nTe, tRoiMs, tLbpMs, tHistMs, tPreprocMs, tPredictMs)
    total = tRoiMs + tLbpMs + tHistMs + tPreprocMs + tPredictMs;
    T = table(string(modelName), nTe, tRoiMs, tLbpMs, tHistMs, tPreprocMs, tPredictMs, total, ...
        'VariableNames', {'Model','N_Test','ROI_ms','LBP_ms','HIST_ms','Preproc_ms','Predict_ms','TotalMsPerSample'});
end

function T = make_timing_table_dl(modelName, nTe, tRoiMs, tPreMs, tFwdMs)
    total = tRoiMs + tPreMs + tFwdMs;
    T = table(string(modelName), nTe, tRoiMs, tPreMs, tFwdMs, total, ...
        'VariableNames', {'Model','N_Test','ROI_ms','DL_Preproc_ms','Forward_ms','TotalMsPerSample'});
end

function [tRoiMs, tPreMs, tFwdMs] = time_cnn_stages(net, filesTe, inputSize, fundusROI, timingOpts)
    filesTe = string(filesTe);
    filesTe = filesTe(arrayfun(@(p) exist(p,'file')==2, filesTe));
    n = numel(filesTe);
    if n == 0, tRoiMs=0; tPreMs=0; tFwdMs=0; return; end

    tRoi = 0; tPre = 0;
    X = zeros([inputSize 3 n], 'single');
    for i = 1:n
        t0 = tic; Iroi = read_fundus_roi(filesTe(i), fundusROI); tRoi = tRoi + toc(t0);
        t0 = tic;
        Irs = imresize(Iroi, inputSize);
        if size(Irs,3) == 1, Irs = repmat(Irs,[1 1 3]); end
        X(:,:,:,i) = im2single(Irs);
        tPre = tPre + toc(t0);
    end

    useGpu = isfield(timingOpts,'GPU_AVAILABLE') && timingOpts.GPU_AVAILABLE && isfield(timingOpts,'USE_GPU_IF_AVAILABLE') && timingOpts.USE_GPU_IF_AVAILABLE;
    t0 = tic;
    if useGpu
        try
            Xg = gpuArray(X);
            scores = predict(net, Xg); %#ok<NASGU>
            wait(gpuDevice);
        catch
            scores = predict(net, X); %#ok<NASGU>
        end
    else
        scores = predict(net, X); %#ok<NASGU>
    end
    tFwd = toc(t0);

    tRoiMs = (tRoi/n)*1000;
    tPreMs = (tPre/n)*1000;
    tFwdMs = (tFwd/n)*1000;
end

function spec = parse_lbp_config(cfg)
    cfg = string(cfg);
    spec = struct('grid',[4 4],'P',8,'R',1,'useRiu2',true);
    m = regexp(cfg, 'grid(\d+)x(\d+)', 'tokens', 'once');
    if ~isempty(m), spec.grid = [str2double(m{1}), str2double(m{2})]; end
    m = regexp(cfg, '_P(\d+)', 'tokens', 'once');
    if ~isempty(m), spec.P = str2double(m{1}); end
    m = regexp(cfg, '_R(\d+)', 'tokens', 'once');
    if ~isempty(m), spec.R = str2double(m{1}); end
    spec.useRiu2 = contains(lower(cfg), 'riu2');
end

function HistSpec = infer_hist_spec_from_table(T)
    HistSpec = struct();
    vars = string(T.Properties.VariableNames);
    hvars = vars(startsWith(vars,'hist_', 'IgnoreCase', true));
    HistSpec.FeatureNames = cellstr(hvars);
end

function feat = extract_hist_ranges(Iu8, mask, HistSpec, doL1)
    if isempty(HistSpec) || ~isfield(HistSpec,'FeatureNames'), feat=[]; return; end
    fnames = string(HistSpec.FeatureNames);
    feat = zeros(1, numel(fnames), 'double');
    R = Iu8(:,:,1); G = Iu8(:,:,2); B = Iu8(:,:,3); Y = rgb2gray(Iu8);
    for i = 1:numel(fnames)
        name = extractAfter(fnames(i), 'hist_');
        tok = regexp(char(name), '^(R|G|B|Gray|Y).*?(\d+).*?(\d+)', 'tokens', 'once');
        if isempty(tok), continue; end
        ch = string(tok{1}); lo = str2double(tok{2}); hi = str2double(tok{3});
        if ch=='R', vals = double(R(mask));
        elseif ch=='G', vals = double(G(mask));
        elseif ch=='B', vals = double(B(mask));
        else, vals = double(Y(mask));
        end
        feat(i) = sum(vals >= lo & vals <= hi);
    end
    if doL1
        s = sum(feat); if s > 0, feat = feat ./ s; end
    end
end

function featLocal = extract_lbp_local_riu2(Iu8, mask, LbpSpec)
    params = struct('P',LbpSpec.P,'R',LbpSpec.R,'useRiu2',true);
    Rch = Iu8(:,:,1); Gch = Iu8(:,:,2); Bch = Iu8(:,:,3); Ych = rgb2gray(Iu8);
    [featLocal, ~] = lbp_local_features(Rch,Gch,Bch,Ych, mask, params, LbpSpec.grid);
end

function nb = get_nbins_lbp(params)
    if isfield(params,'useRiu2') && params.useRiu2, nb = params.P + 2; else, nb = 2^params.P; end
end

function h = lbp_hist_norm(Ich, mask, params)
    counts = lbp_hist_counts(Ich, mask, params);
    s = sum(counts);
    if s > 0, h = counts / s; else, h = counts; end
end

function [featVec, meta] = lbp_local_features(Rch,Gch,Bch,Ych, mask, params, patchGrid)
    nbins = get_nbins_lbp(params);
    [H,W] = size(mask);
    stats = regionprops(mask,'BoundingBox');
    if isempty(stats)
        featVec = zeros(1, patchGrid(1)*patchGrid(2)*4*nbins);
        meta = struct('empty',true);
        return;
    end
    bb = stats(1).BoundingBox;
    x0 = max(1, floor(bb(1))); y0 = max(1, floor(bb(2)));
    x1 = min(W, ceil(bb(1)+bb(3)-1)); y1 = min(H, ceil(bb(2)+bb(4)-1));

    Gh = patchGrid(1); Gw = patchGrid(2);
    patchW = floor((x1-x0+1)/Gw);
    patchH = floor((y1-y0+1)/Gh);

    featList = zeros(Gh*Gw, 4*nbins);
    patchInfo = [];
    idx = 0;
    for rr = 1:Gh
        for cc = 1:Gw
            idx = idx + 1;
            xs = x0 + (cc-1)*patchW; ys = y0 + (rr-1)*patchH;
            if cc < Gw, xe = xs + patchW - 1; else, xe = x1; end
            if rr < Gh, ye = ys + patchH - 1; else, ye = y1; end

            pmask = mask(ys:ye, xs:xe);
            if nnz(pmask) < 200
                featList(idx,:) = 0;
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
    meta = struct(); meta.patchGrid = patchGrid; meta.patchInfo = patchInfo;
end

function h = lbp_hist_counts(Ich, mask, params)
    I = im2double(Ich);
    P = params.P; R = params.R;
    codes = lbp_codes_circular(I, P, R);
    border = ceil(R);
    if size(mask,1) <= 2*border || size(mask,2) <= 2*border
        h = zeros(1, get_nbins_lbp(params));
        return;
    end
    mask2 = mask(1+border:end-border, 1+border:end-border);
    vals  = codes(mask2);
    nbins = get_nbins_lbp(params);
    if params.useRiu2
        labels = lbp_riu2_map(uint16(vals), P);
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
    codes = zeros(size(gc), 'uint16');
    for p = 0:(P-1)
        theta = 2*pi*p/P;
        dx = R*cos(theta);
        dy = -R*sin(theta);
        Xn = Xc + dx;
        Yn = Yc + dy;
        gp = interp2(I, Xn, Yn, 'linear');
        bit = gp >= gc;
        codes = bitor(codes, uint16(bit) * uint16(2^p));
    end
end

function labels = lbp_riu2_map(codes, P)
    persistent lutP lut;
    if isempty(lut) || isempty(lutP) || lutP ~= P
        lutP = P;
        lut = zeros(2^P,1,'uint16');
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

function [Iout, mask] = read_fundus_roi_with_mask(fn, p)
    I = imread(fn);
    I = force_rgb(I);
    Id = im2double(I);
    mask = build_fundus_mask(Id, p);
    Iout = Id;
    Iout(repmat(~mask,[1 1 3])) = 0;
    if isfield(p,'cropToMaskBBox') && p.cropToMaskBBox
        stats = regionprops(mask, 'BoundingBox');
        if ~isempty(stats)
            bb = stats(1).BoundingBox;
            Iout = imcrop(Iout, bb);
            mask = imcrop(mask, bb) > 0;
        end
    end
    Iout = im2single(Iout);
end

function x = normalize01(x)
    x = double(x);
    x = x - min(x(:));
    d = max(x(:));
    if d < 1e-12, d = 1e-12; end
    x = x / d;
end

function spec = dummy(~), spec=[]; end %#ok<DEFNU>

%% ----------- DL helper (trainNetwork compatible, no trainnet/dlnetwork) -----------
function [Yhat, posScore, dt, info, trainedNet, filesTe] = predict_cnn_tl(whichNet, V, tr, te, inputSize, miniBatch, maxEpochs, learnRate, freezeBackbone, fundusROI, timingOpts)

    filesAll = string(V.T.Filename);
    yAll     = V.Y;

    filesTr0 = filesAll(tr);
    filesTe0 = filesAll(te);

    okTr = arrayfun(@(p) exist(p,"file")==2, filesTr0);
    okTe = arrayfun(@(p) exist(p,"file")==2, filesTe0);

    filesTr = filesTr0(okTr);
    filesTe = filesTe0(okTe);

    yTr = yAll(tr);  yTr = yTr(okTr);
    yTe = yAll(te);  yTe = yTe(okTe);

    if numel(filesTr) == 0, error("DL: TRAIN empty after path filtering."); end
    if numel(filesTe) == 0, error("DL: TEST empty after path filtering."); end
    if numel(unique(yTr)) < 2, error("DL: TRAIN has only one class after filtering."); end

    yTr = categorical(yTr, {'cataract','normal'});
    yTe = categorical(yTe, {'cataract','normal'});

    imdsTr = imageDatastore(cellstr(filesTr)); imdsTr.Labels = yTr;
    imdsTe = imageDatastore(cellstr(filesTe)); imdsTe.Labels = yTe;

    imdsTr.ReadFcn = @(fn) read_fundus_roi(fn, fundusROI);
    imdsTe.ReadFcn = @(fn) read_fundus_roi(fn, fundusROI);

    [imdsTrain, imdsVal] = splitEachLabel(imdsTr, 0.8, "randomized");

    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandRotation', [-10 10], ...
        'RandScale', [0.95 1.05]);

    augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
    augVal   = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'gray2rgb');
    augTe    = augmentedImageDatastore(inputSize, imdsTe, 'ColorPreprocessing', 'gray2rgb');

    if whichNet == "resnet50"
        net0 = resnet50;
    else
        net0 = mobilenetv2;
    end

    lgraph = layerGraph(net0);
    if whichNet == "mobilenetv2"
        lgraph = replace_mobilenet_head(lgraph, 2);
    else
        lgraph = replace_last_fc(lgraph, 2);
    end

    if freezeBackbone
        % Simple freezing: set LR factors to 0 for all layers; head gets overwritten by replace_* (LR=10)
        layers = lgraph.Layers;
        for i = 1:numel(layers)
            if isprop(layers(i),'WeightLearnRateFactor'), layers(i).WeightLearnRateFactor = 0; end
            if isprop(layers(i),'BiasLearnRateFactor'),   layers(i).BiasLearnRateFactor   = 0; end
        end
        lgraph = layerGraph(layers);
        learnRate = learnRate * 0.25;
    end

    % Ensure classification head exists
    lgraph = ensure_softmax_and_classification(lgraph, {'cataract','normal'});

    execEnv = 'cpu';
    if isfield(timingOpts,'GPU_AVAILABLE') && timingOpts.GPU_AVAILABLE && isfield(timingOpts,'USE_GPU_IF_AVAILABLE') && timingOpts.USE_GPU_IF_AVAILABLE
        execEnv = 'gpu';
    end

    numIter = max(1, floor(numel(augTrain.Files)/miniBatch));

    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', learnRate, ...
        'MiniBatchSize', miniBatch, ...
        'ValidationData', augVal, ...
        'ValidationFrequency', numIter, ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', execEnv);

    trainedNet = trainNetwork(augTrain, lgraph, options);

    t0 = tic;
    scores = predict(trainedNet, augTe);
    dt = toc(t0);

    if size(scores,2) ~= 2 && size(scores,1) == 2, scores = scores'; end

    classNames = {'cataract','normal'};
    [~, idx] = max(scores, [], 2);
    Yhat = categorical(classNames(idx), classNames);

    posScore = scores(:,1); % cataract assumed first
    filesTe = string(augTe.Files);
    info = [];
end

function lgraph = ensure_softmax_and_classification(lgraph, classNamesCell)
    isOut = arrayfun(@(L) contains(class(L),"Classification","IgnoreCase",true) || contains(class(L),"OutputLayer","IgnoreCase",true), lgraph.Layers);
    idx = find(isOut, 1, "last");
    if ~isempty(idx)
        lgraph = removeLayers(lgraph, lgraph.Layers(idx).Name);
    end

    lastName = lgraph.Layers(end).Name;

    if ~any(strcmp({lgraph.Layers.Name}, 'softmax'))
        lgraph = addLayers(lgraph, softmaxLayer('Name','softmax'));
        lgraph = connectLayers(lgraph, lastName, 'softmax');
        lastName = 'softmax';
    end

    if ~any(strcmp({lgraph.Layers.Name}, 'classoutput'))
        lgraph = addLayers(lgraph, classificationLayer('Name','classoutput','Classes', categorical(classNamesCell)));
        lgraph = connectLayers(lgraph, lastName, 'classoutput');
    end
end

function lgraph = replace_last_fc(lgraph, numClasses)
    idxFC = find(arrayfun(@(L) isa(L, 'nnet.cnn.layer.FullyConnectedLayer'), lgraph.Layers), 1, 'last');
    oldName = lgraph.Layers(idxFC).Name;
    newFC = fullyConnectedLayer(numClasses, 'Name', oldName, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    lgraph = replaceLayer(lgraph, oldName, newFC);
end

function lgraph = replace_mobilenet_head(lgraph, numClasses)
    idxLogits = find(strcmp({lgraph.Layers.Name}, 'Logits'));
    if ~isempty(idxLogits)
        oldName = 'Logits';
        newFC = fullyConnectedLayer(numClasses, 'Name', oldName, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
        lgraph = replaceLayer(lgraph, oldName, newFC);
    else
        idxFC = find(arrayfun(@(L) isa(L, 'nnet.cnn.layer.FullyConnectedLayer'), lgraph.Layers));
        if ~isempty(idxFC)
            oldName = lgraph.Layers(idxFC(end)).Name;
            newFC = fullyConnectedLayer(numClasses, 'Name', oldName, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
            lgraph = replaceLayer(lgraph, oldName, newFC);
        else
            error("No 'Logits' or FullyConnectedLayer found to replace.");
        end
    end
end

%% ----------- ROI (UNCHANGED) -----------
function Iout = read_fundus_roi(fn, p)
    I = imread(fn); I = force_rgb(I); Id = im2double(I); mask = build_fundus_mask(Id, p);
    Iout = Id; Iout(repmat(~mask, [1 1 3])) = 0;
    if isfield(p, "cropToMaskBBox") && p.cropToMaskBBox
        stats = regionprops(mask, "BoundingBox"); if ~isempty(stats), Iout = imcrop(Iout, stats(1).BoundingBox); end
    end
    Iout = im2single(Iout);
end

function mask = build_fundus_mask(Id, p)
    V = max(Id, [], 3); Vsm = imgaussfilt(V, p.gaussSigma); nonBlack = Vsm > p.nonBlackThr;
    nonBlack = imclose(nonBlack, strel("disk", p.closeRadius, 0));
    if p.fillHoles, nonBlack = imfill(nonBlack, "holes"); end
    if ~any(nonBlack(:)), mask = true(size(nonBlack)); return; end
    nPix = numel(nonBlack); minArea = max(1, round(p.minAreaFrac * nPix));
    nonBlack = bwareaopen(nonBlack, minArea); nonBlack = bwareafilt(nonBlack, 1);

    if p.useEdges
        try
            E = edge(Vsm, p.edgeMethod, [], p.edgeSigma);
            if p.edgeDilate > 0, E = imdilate(E, strel("disk", p.edgeDilate, 0)); end
            if p.edgeClose > 0, E = imclose(E, strel("disk", p.edgeClose, 0)); end
            E = E & nonBlack; edgeFilled = bwareaopen(imfill(E, "holes"), minArea);
            if nnz(edgeFilled & nonBlack)/max(1, nnz(nonBlack)) > 0.60, mask0 = nonBlack | edgeFilled; else, mask0 = nonBlack; end
        catch
            mask0 = nonBlack;
        end
    else
        mask0 = nonBlack;
    end

    mask = imfill(mask0, "holes"); mask = bwareafilt(mask, 1);
    mask = imerode(mask, strel("disk", max(0, p.innerMarginPx), 0));
    if p.erodePixels > 0, mask = imerode(mask, strel("disk", p.erodePixels, 0)); end
    mask = bwareafilt(imfill(mask, "holes"), 1);
end

function I = force_rgb(I)
    if ismatrix(I), I = repmat(I, [1 1 3]); elseif size(I,3) > 3, I = I(:,:,1:3); end
end

function make_variant_summary_plot_python_like(Res, outV, tag, PUB)
    fig = figure("Visible","off","Color","w","Position",[100 100 PUB.FigW PUB.FigH]);
    mdl = string(Res.Model); x = 1:numel(mdl);
    yyaxis left; plot(x, Res.AUCTest, "o-"); hold on; plot(x, Res.F1Test, "s-"); ylim([0 1]); ylabel("Score (holdout)");
    yyaxis right; plot(x, Res.TimePerSampleTestMs, "d-"); ylabel("Inference time (ms/sample)");
    xticks(x); xticklabels(mdl); xtickangle(25);
    title("Variant Summary: " + string(tag), "Interpreter","none", "FontSize", PUB.FontSizeTitle);
    legend({"AUC (ROC)","F1","Time"}, "Location","best");
    exportgraphics(fig, fullfile(outV, "_variant_summary.png"), "Resolution", PUB.DPI);
    close(fig);
end
