clc;
clear;

% Φόρτωση των δεδομένων με την κατάλληλη μορφη 
[segments, labels] = swallow_create_data("BCICIV_1_mat/BCICIV_calib_ds1a.mat");

% Στάδιο προεπεξεργασίας σε κάθε segment ξεχωριστά
fs = 100;
for i = 1:length(segments)
    X = preprocessEEG(segments{i}, fs);   
    segments{i} = X;
end

% Split train/test
cv = cvpartition(length(segments), 'HoldOut', 0.2);
train_data = training(cv);
test_data  = test(cv);

trainSequences = segments(train_data);
testSequences  = segments(test_data);
train_labels = labels(train_data);
test_labels  = labels(test_data);

% Παραμετροποίηση των πειραμάτων 
inputSize = size(trainSequences{1}, 1);
numClasses = 2;
learning_rates = [0.0001,0.001];

modelTypes = {
    [64 16],...
    64
};
dropoutRates = [0];
fcLayerSizes = [2];
batch_sizes = [16];
l2_reg = [0.01];
max_epochs = 70;

% Δημιουργία φακέλου με ημερομηνία για αποθήκευση αποτελεσμάτων και κώδικα
timestamp = datestr(now, 'mm-dd_HH-MM');
output_folder = fullfile('results', ['Swallow_LSTM_' timestamp]);
mkdir(output_folder);
results_filename = sprintf('lstm_results_.xlsx');
script_name = 'Swallow_LSTM.m';
copyfile(script_name, fullfile(output_folder, script_name));

experiment_data = [];
exp_id = 1;

for m = 1:length(modelTypes)
    hiddenUnits = modelTypes{m};
    nLayers = length(hiddenUnits);

    for d = 1:length(dropoutRates)
        for f = 1:length(fcLayerSizes)
            for b = 1:length(batch_sizes)
                for lreg = 1:length(l2_reg)
                    for lr = 1:length(learning_rates)

                        fprintf("Experiment %d: ModelType=%s | Dropout=%.2f | FC=%d | BatchSize=%d\n | L2Reg=%.2f | lr=%.5f\n" , ...
                            exp_id, mat2str(hiddenUnits), dropoutRates(d), fcLayerSizes(f), batch_sizes(b), l2_reg(lreg), learning_rates(lr));
        
                        % Δημιουργία του LSTM δικτύου
                        layers = [sequenceInputLayer(inputSize)];
                        for l = 1:nLayers
                            outputMode = 'last';
                            if l < nLayers
                                outputMode = 'sequence';
                            end
                            layers = [layers;
                                lstmLayer(hiddenUnits(l), 'OutputMode', outputMode)];                                                              
                        end
        
                        layers = [layers;
                            fullyConnectedLayer(fcLayerSizes(f))
                            dropoutLayer(dropoutRates(d))
                            fullyConnectedLayer(numClasses)
                            softmaxLayer
                            classificationLayer];
                        
                        options = trainingOptions('adam', ...
                            'InitialLearnRate', learning_rates(lr), ...
                            'MaxEpochs', max_epochs, ...
                            'MiniBatchSize', batch_sizes(b), ...                                                       
                            'Plots', 'training-progress', ...  
                            'L2Regularization', l2_reg(lreg));
                        
                        % Εκπαίδευση
                        tic;
                        [net, trainingInfo] = trainNetwork(trainSequences, train_labels, layers, options);
                        training_time = toc;

                        net_filename = sprintf('net_exp_%d.mat', exp_id);
                        save(fullfile(output_folder, net_filename), 'net');
                                              

                        predsTrain = classify(net, trainSequences);
                        acc_train = sum(predsTrain == train_labels) / numel(train_labels);
        
                        predsTest = classify(net, testSequences);
                        acc_test = sum(predsTest == test_labels) / numel(test_labels);
        
                        % confusion matrices
                        f1 = figure('visible','off');
                        plotconfusion(train_labels, predsTrain');
                        saveas(f1, fullfile(output_folder, sprintf('conf_train_%d.png', exp_id)));
                        close(f1);
        
                        f2 = figure('visible','off');
                        plotconfusion(test_labels, predsTest');
                        saveas(f2, fullfile(output_folder, sprintf('conf_test_%d.png', exp_id)));
                        close(f2);
                                              
                        % Καταγραφή παραμέτρων και αποτελεσμάτων του πειράματος                        
                        experiment_data = [experiment_data; {
                            exp_id, ...
                            mat2str(hiddenUnits), ...
                            dropoutRates(d), ...
                            fcLayerSizes(f), ...
                            batch_sizes(b), ... 
                            l2_reg(lreg), ...
                            learning_rates(lr), ...
                            acc_train * 100, ...
                            acc_test * 100, ...
                            training_time, ...
                            sprintf('conf_train_%d.png', exp_id), ...
                            sprintf('conf_test_%d.png', exp_id),...
                            net_filename
                            
                        }];
        
                        exp_id = exp_id + 1;
                    end
                end
            end
        end
    end
end

% Αποθήκευση αποτελεσμάτων ανα πείραμα με τις παραμέτρους σε μορφή Excel 
results_table = cell2table(experiment_data, ...
    'VariableNames', {'ID','HiddenUnits','Dropout','FCLayerSize','BatchSize', ...
                      'L2Regularization', ...
                      'LearningRate','TrainAcc','TestAcc','TrainTime','TrainConfMat', ...
                      'TestConfMat','NetFile'});

writetable(results_table, fullfile(output_folder, results_filename));

fprintf("All experiments finished. Results saved to %s\n", fullfile(output_folder, results_filename));