function visualizeEEGSegment(raw_segment, fs, trial_num, channel_num)
% Visualise ενός EEG segment πριν και μετά την προεπεξεργασία
% raw_segment : [channels x samples]
% fs          : συχνότητα δειγματοληψίας (Hz)
% trial_num   : αριθμός trial για τον τίτλο
% channel_num : αριθμός καναλιού για εμφάνιση

    % Εφαρμογή προεπεξεργασίας στο segment
    preprocessed_segment = preprocessEEG(raw_segment, fs);

    % Δημιουργία χρονικού άξονα (σε sec)
    time_axis = (0:size(raw_segment,2)-1) / fs;
   
    figure;

    % Ακατέργαστο segment
    subplot(2,1,1);
    plot(time_axis, raw_segment(channel_num,:), 'b');
    title(sprintf('Ακατέργαστο EEG - Κανάλι %d, Trial %d', channel_num, trial_num), 'FontSize', 12);
    xlabel('Χρόνος (s)');
    ylabel('Πλάτος (\muV)');
    grid on;

    % Φιλτραρισμένο segment
    subplot(2,1,2);
    plot(time_axis, preprocessed_segment(channel_num,:), 'r');
    title(sprintf('Φιλτραρισμένο EEG - Κανάλι %d, Trial %d', channel_num, trial_num), 'FontSize', 12);
    xlabel('Χρόνος (s)');
    ylabel('Πλάτος (\muV)');
    grid on;

end