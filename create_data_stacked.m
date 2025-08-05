function [data, labels, trials] = create_data_stacked(file_path)

    load(file_path);  % Φόρτωση δεδομένων απο το dataset (cnt, mrk, nfo)
    fs = 100; % Συχνότητα δειγματοληψίας 
    cnt = 0.1 * double(cnt);  % Μετετροπή σε µV
    events = mrk.pos;
    labels = mrk.y;

    channels = size(cnt, 2); % Αριθμός καναλιών 
    pre_samples = round(0.5 * fs);   % 50 samples
    post_samples = round(4.0 * fs);  % 400 samples
    segment_len = pre_samples + post_samples;  % 450 samples

    data = []; % Ο τελίκος πίνακας δεδομένων
    all_labels = []; 
    trials = [];
    trial_cnt = 1;

    for i = 1:length(events)
        % Χωρισμός των δεδομένων σε Segments  
        firstSample = events(i) - pre_samples;
        lastSample = events(i) + post_samples - 1;

        if firstSample < 1 || lastSample > size(cnt, 1)
            continue;
        end
        seg = cnt(firstSample:lastSample, :)';% [59 × 450]

        filt_seg = preprocessEEG(seg, fs); % Φιλτράρισμα σε καθε segment

        % Προσθήκη του αντίστροφου segment [450 × 59] στον πίνακα data 
        data = [data, filt_seg'];       % [450 × (59 * trial_cnt)]

        % One label per channel per trial
        all_labels = [all_labels, repmat((labels(i)+3)/2, 1, channels)];

        % Αντιστοίχηση του trial_cnt σε όλα τα κανάλια σε ποιο trial ανήκουν
        trials = [trials, repmat(trial_cnt, 1, channels)]; % [1 × (59 * trial_cnt)]
        trial_cnt = trial_cnt + 1;
    end

    labels = categorical(all_labels); % [1 × (59 * trials_cnt-1)]
end