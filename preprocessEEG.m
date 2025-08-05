function X_filtered = preprocessEEG(X, fs)
% X: [channels × time]=[59x450] EEG segment
% fs: Συχνότητα δειγματοληψίας 

% Bandpass Φίλτρο 4–35Hz
order = 4; % Ταξη του φιλτρου 
[b, a] = butter(order, [4 35] / (fs/2), 'bandpass');
% Εφαρμογή σε καθε κανάλι απο το trial
for ch = 1:size(X,1)
    X(ch,:) = filtfilt(b, a, X(ch,:));
end

X_filtered = X;
end
