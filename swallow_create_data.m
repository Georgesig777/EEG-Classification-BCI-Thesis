function [segments, labels] = swallow_create_data(file_paths)
   
    if ischar(file_paths)
        file_paths = {file_paths};  % Μετατροπή του κάθε path σε κελί(cell array)
    end
    
    segments = {};  
    labels = [];    
    preTime = 0.5;           % Χρόνος σε δευτερόλεπτα πριν το ερέθισμα του event
    postTime = 4.0;          % Χρόνος σε δευτερόλεπτα μετά το ερέθισμα του event

    for f = 1:length(file_paths)
        load(file_paths{f});  % Φόρτωση δεδομένων από το dataset (cnt, mrk, nfo)
        
        cnt = 0.1 * double(cnt);  % Μετατροπή των μονάδων σε µV
        fs = nfo.fs;              % Συχνότητα δειγματοληψίας
        
        segment_len = round((preTime + postTime) * fs); % Συνολικό μήκος segment σε δείγματα (450 samples)

        n_channels = size(cnt, 2);  % Συνολικός αριθμός καναλιών
        events = mrk.pos;          % Χρονικές στιγμές των evenets (γεγονότα)
        labels_raw = mrk.y;        % Κλάσεις που ανήκουν το καθε event 

        for i = 1:length(events)
            % Υπολογισμός ορίων του segment γύρω από το event
            startTime = events(i) - round(preTime * fs);
            endTime = startTime + segment_len - 1;

            % Εξαγωγή του segment και μετατροπή του σε [channels × samples]=[59x450]           
            seg = cnt(startTime:endTime, :)';
            segments{end+1} = seg;          
            labels(end+1) = labels_raw(i);  % Προσθήκη της αντίστοιχης κλάσης
        end
    end

    % Μετατοπή των κλάσεων σε τιμες [1, 2] και στην κατάλληλη μορφη 
    labels = (labels + 3) / 2;
    labels = categorical(labels);
end