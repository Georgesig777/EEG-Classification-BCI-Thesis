clc;
clear;

% Φόρτωση των EEG δεδομένων και των ετικετών
file_paths = {
    'BCICIV_1_mat/BCICIV_calib_ds1a.mat'
};

[segments, labels] = swallow_create_data(file_paths);

% Στάδιο προεπεξεργασία: Φιλτράρισμα για κάθε segment ξεχωριστά
fs = 100; % Συχνότητα δειγματοληψίας
for i = 1:length(segments)
    X = segments{i};
    X = preprocessEEG(X, fs);   
    segments{i} = X;            
end

% Splitting των δεδομένων σε σύνολα train και test 
cv = cvpartition(length(segments), 'HoldOut', 0.2);
train_idx = find(training(cv));
test_idx = find(test(cv));

% Παράμετροι για το cropping των σημάτων
cropSize = 250;   % Μέγεθος παραθύρου (samples)
stride = 30;      % Βήμα μετατόπισης του παραθύρου
inputSize = [59 cropSize 1];  % Διαστάσεις εισόδου του CNN

% Παραμετροποίηση των πειραμάτων 
conv_filters = [20];   % Αριθμός φίλτρων convolution
dropouts = [0.3];      % Ποσοστό dropout
batch_sizes = [64];    % Μέγεθος batch για εκπαίδευση
max_epochs = [30];     % Μέγιστος αριθμός εποχών

results = [];  % Αποθήκευση αποτελεσμάτων κάθε πειράματος
timestamp = datestr(now, 'mm-dd_HH-MM');
output_folder = fullfile('results', ['CNN_' timestamp]); % Δημιουργία φακέλου για αποθήκευση
mkdir(output_folder);
result_file = fullfile(output_folder, 'cnn_results.xlsx');

exp_id = 1;  % Μετρητής πειραμάτων

for f = 1:length(conv_filters)
    for d = 1:length(dropouts)
        for b = 1:length(batch_sizes)
            for e = 1:length(max_epochs)

                % Προετοιμασία δεδομένων εκπαίδευσης με cropping
                [XTrain, YTrain] = deal([]);
                for idx = 1:numel(train_idx)
                    trialIndex = train_idx(idx);
                    x = segments{trialIndex};
                    label = labels(trialIndex);
                    for t = 1:stride:(size(x,2) - cropSize + 1)
                        crop = x(:, t:t+cropSize-1);
                        XTrain = cat(4, XTrain, reshape(crop, [size(crop), 1]));
                        YTrain = [YTrain; label];
                    end
                end
                YTrain = categorical(YTrain); % Ετικέτες για κάθε crop

                % Δημιουργία του CNN μοντέλου                
                layers = [
                    imageInputLayer(inputSize, 'Name', 'input')
                    convolution2dLayer([1 25], conv_filters(f), 'Stride', [1 1], 'Name', 'conv_temporal')
                    batchNormalizationLayer('Name','bn1')
                    convolution2dLayer([59 1], conv_filters(f), 'Stride', [1 1], 'Name','conv_spatial', 'BiasLearnRateFactor', 0)
                    batchNormalizationLayer('Name','bn2')
                    functionLayer(@(X) X.^2, 'Name','square')  % Τετραγωνισμός του σήματος
                    averagePooling2dLayer([1 150], 'Stride', [1 30], 'Name','mean_pool')
                    functionLayer(@(X) log(max(X, 1e-6)), 'Name','log')  % Εφαρμογή λογαρίθμου
                    dropoutLayer(dropouts(d), 'Name','dropout')
                    fullyConnectedLayer(2, 'Name','fc')
                    softmaxLayer('Name','softmax')
                    classificationLayer('Name','output')
                ];

                % Ρυθμίσεις εκπαίδευσης
                options = trainingOptions('adam', ...
                    'MaxEpochs', max_epochs(e), ...
                    'MiniBatchSize', batch_sizes(b), ...
                    'Shuffle', 'never', ...
                    'Plots','none', ...
                    'Verbose', false);

                % Εκπαίδευση του μοντέλου
                tic;
                net = trainNetwork(XTrain, YTrain, layers, options);
                training_time = toc;

                % Αξιολόγηση στο σύνολο εκπαίδευσης
                YPredTrain = classify(net, XTrain);
                trainAcc = mean(YPredTrain == YTrain);
                
                f1 = figure('visible','off');
                plotconfusion(YTrain, YPredTrain);
                trainConfFile = sprintf('conf_train_%d.png', exp_id);
                saveas(f1, fullfile(output_folder, trainConfFile));
                close(f1);

                % Αξιολόγηση στο test set ανά trial (softmax average)
                YPredTrials = repmat(categorical(missing), numel(test_idx), 1);
                trueLabels = labels(test_idx);

                for i = 1:numel(test_idx)
                    x = segments{test_idx(i)};
                    if size(x,2) < cropSize
                        continue;
                    end
                    crops = [];
                    for t = 1:stride:(size(x,2) - cropSize + 1)
                        crop = x(:, t:t+cropSize-1);
                        crops = cat(4, crops, reshape(crop, [size(crop), 1]));
                    end
                    if isempty(crops)
                        continue;
                    end
                    scores = predict(net, crops);
                    avgScore = mean(scores, 1);
                    [~, idx_class] = max(avgScore);
                    YPredTrials(i) = net.Layers(end).Classes(idx_class);
                end

                valid = ~ismissing(YPredTrials);
                testAcc = mean(YPredTrials(valid) == trueLabels(valid));

                % Αποθήκευση πίνακα σύγχυσης για τα test δεδομένα (per trial)
                f2 = figure('visible','off');
                plotconfusion(trueLabels(valid), YPredTrials(valid)');
                testConfFile = sprintf('conf_test_%d.png', exp_id);
                saveas(f2, fullfile(output_folder, testConfFile));
                close(f2);

                % Αξιολόγηση στο test set ανά crop
                XTest = [];
                YTest = [];
                
                for idx = 1:numel(test_idx)
                    trialIndex = test_idx(idx);
                    x = segments{trialIndex};
                    label = labels(trialIndex);
                    for t = 1:stride:(size(x,2) - cropSize + 1)
                        crop = x(:, t:t+cropSize-1);
                        XTest = cat(4, XTest, reshape(crop, [size(crop), 1]));
                        YTest = [YTest; label];
                    end
                end
                YTest = categorical(YTest);
                
                % Προβλέψεις ανά crop
                YPredTest = classify(net, XTest);
                testAcc_crop = mean(YPredTest == YTest);

                % Αποθήκευση πίνακα σύγχυσης για το test set (per crop)
                f2 = figure; 
                plotconfusion(YTest(:), YPredTest(:));
                title(sprintf('Test Confusion Matrix (Per Crop) – Exp %d', exp_id));
                testConfFile = sprintf('conf_test_crop_%d.png', exp_id);
                saveas(f2, fullfile(output_folder, testConfFile));

                % Καταγραφή παραμέτρων και αποτελεσμάτων του πειράματος   
                results = [results; {
                    exp_id, conv_filters(f), dropouts(d), batch_sizes(b), max_epochs(e), ...
                    trainAcc*100, testAcc_crop*100, training_time, trainConfFile, testConfFile
                }];

                exp_id = exp_id + 1;
            end
        end
    end
end

% Αποθήκευση όλων των αποτελεσμάτων σε αρχείο Excel
results_table = cell2table(results, ...
    'VariableNames', {'ID','Filters','Dropout','BatchSize','MaxEpochs','TrainAcc','TestAcc','TrainTime','TrainConfMat','TestConfMat'});
writetable(results_table, result_file);
fprintf("Όλα τα πειράματα CNN ολοκληρώθηκαν. Τα αποτελέσματα αποθηκεύτηκαν στο %s\n", result_file);