clc;
clear;

% Φόρτωση των δεδομένων με την κατάλληλη μορφη 
[data, labels, trials] = create_data_stacked("BCICIV_1_mat/BCICIV_calib_ds1a.mat");

% Train-test split 80/20 (Χωρίς να διαχωρίζονται τα δεδομενα απο τα trials)
unique_trials = unique(trials);

cv = cvpartition(length(unique_trials), 'HoldOut', 0.2);
train_trials = unique_trials(training(cv));
test_trials  = unique_trials(test(cv));

% Επιλογή samples που ανήκουν στα train και test trials
train_mask = ismember(trials, train_trials);
test_mask  = ismember(trials, test_trials);

train_data   = data(:, train_mask);
train_labels = labels(:, train_mask);
train_trials = trials(:, train_mask);

test_data    = data(:, test_mask);
test_labels  = labels(:, test_mask);
test_trials = trials(:, test_mask);

% Shuffling των δεδομένων
rand_idx_train = randperm(size(train_data, 2));
train_data = train_data(:, rand_idx_train);
train_labels = train_labels(:, rand_idx_train);
train_trials = train_trials(:, rand_idx_train);

rand_idx_test = randperm(size(test_data, 2));
test_data = test_data(:, rand_idx_test);
test_labels = test_labels(:, rand_idx_test);
test_trials = test_trials(:, rand_idx_test);

%  Μετατροπή labels και one-hot encoding
train_labels_vec = double(train_labels);
test_labels_vec  = double(test_labels);

train_targets = full(ind2vec(train_labels_vec));
test_targets  = full(ind2vec(test_labels_vec));

%  Παραμετροποίηση των πειραμάτων 
hid_layers = {[10]};       % Αρχιτεκτονική δικτύου
lrates = [0.0001];       % Ρυθμός μάθησης
regs = 0;                   % Regularization
max_epochs = [12];         % Μέγιστος αριθμός εποχών
goals = [1e-3];            % Στόχος σφάλματος

% Δημιουργία φακέλου με ημερομηνία για αποθήκευση αποτελεσμάτων
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM');
output_folder = fullfile('results', ['FFN_' timestamp]);
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Εκτέλεση πειραμάτων
exp_id = 1;
experiment_data = {};

for h = 1:length(hid_layers)
    for lr = 1:length(lrates)
        for reg = 1:length(regs)
            for ep = 1:length(max_epochs)
                for g = 1:length(goals)

                    % Εμφάνιση ρυθμίσεων πειράματος
                    fprintf("Πείραμα %d: Layers=%s | LR=%.5f | Reg=%.3f | Epochs=%d | Goal=%.1e\n", ...
                        exp_id, strjoin(string(hid_layers{h}), '-'), ...
                        lrates(lr), regs(reg), max_epochs(ep), goals(g));

                    % Δημιουργία και εκπαίδευση του δικτύου
                    net = feedforwardnet(hid_layers{h}, 'trainlm');
                    net.trainParam.lr = lrates(lr);
                    net.trainParam.epochs = max_epochs(ep);
                    net.trainParam.goal = goals(g);
                    net.performParam.regularization = regs(reg);

                    [net, tr] = train(net, train_data, train_targets);

                    % Αξιολόγηση στο training set
                    outputs_train = net(train_data);
                    [~, predicted_train_channels] = max(outputs_train);
                    acc_train_channels = mean(predicted_train_channels == train_labels_vec);

                    % Αξιολόγηση στο test set
                    outputs_test = net(test_data);
                    [~, predicted_test_channels] = max(outputs_test);
                    acc_test_channels = mean(predicted_test_channels == test_labels_vec);

                    % Αξιολόγηση σε επίπεδο trial
                    trials_in_test = unique(test_trials);
                    trial_preds = arrayfun(@(t) mode(predicted_test_channels(test_trials==t)), trials_in_test);
                    trial_truth = arrayfun(@(t) mode(test_labels_vec(test_trials==t)), trials_in_test);
                    acc_trial = mean(trial_preds == trial_truth);

                    % Αποθήκευση confusion matrices
                    f_train = figure('visible','off');
                    plotconfusion(train_targets, outputs_train);
                    train_conf_file = sprintf('conf_train_channels_%d.png', exp_id);
                    saveas(f_train, fullfile(output_folder, train_conf_file));
                    close(f_train);

                    f_test = figure('visible','off');
                    plotconfusion(test_targets, outputs_test);
                    test_conf_file = sprintf('conf_test_channels_%d.png', exp_id);
                    saveas(f_test, fullfile(output_folder, test_conf_file));
                    close(f_test);

                    f_trial = figure('visible','off');
                    targets_onehot = full(ind2vec(trial_truth, 2));
                    preds_onehot   = full(ind2vec(trial_preds, 2));
                    plotconfusion(targets_onehot, preds_onehot);
                    test_trial_conf_file = sprintf('conf_test_trials_%d.png', exp_id);
                    saveas(f_trial, fullfile(output_folder, test_trial_conf_file));
                    close(f_trial);

                    % Καταγραφή αποτελεσμάτων
                    experiment_data = [experiment_data; {
                        exp_id, ...
                        strjoin(string(hid_layers{h}), '-'), ...
                        lrates(lr), ...
                        regs(reg), ...
                        max_epochs(ep), ...
                        goals(g), ...
                        acc_train_channels * 100, ...
                        acc_test_channels * 100, ...
                        acc_trial * 100, ...
                        train_conf_file, ...
                        test_conf_file, ...
                        test_trial_conf_file
                    }];

                    fprintf("Train: %.2f%% | Test: %.2f%% | Trial: %.2f%%\n", ...
                        acc_train_channels * 100, acc_test_channels * 100, acc_trial * 100);

                    exp_id = exp_id + 1;
                end
            end
        end
    end
end

% Αποθήκευση αποτελεσμάτων ανα πείραμα με τις παραμέτρους σε μορφή Excel 
results_file = fullfile(output_folder, 'ffn_results.xlsx');
results_table = cell2table(experiment_data, ...
    'VariableNames', {'ID','Layers','LearningRate','Regularization', ...
                      'MaxEpochs','Goal',...
                      'TrainAcc_Channels','TestAcc_Channels','TestAcc_Trials', ...
                      'ConfMat_Train_Channels','ConfMat_Test_Channels','ConfMat_Test_Trials'});
writetable(results_table, results_file);

fprintf("Όλα τα πειράματα ολοκληρώθηκαν. Τα αποτελέσματα αποθηκεύτηκαν στο %s\n", results_file);