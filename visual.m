% Τα μοντέλα και τα accuracies 
models = {'FFN', 'LSTM', 'CNN'};
acc = [60.4, 70, 90];

% Δημιουργία figure
figure('Color', 'w'); % Λευκό φόντο
b = bar(acc, 'FaceColor', 'flat', 'EdgeColor', 'none'); 

% Ανοιχτά χρώματα για κάθε μπάρα (όχι μαύρο εσωτερικά)
b.CData = [0.4 0.7 1; 0.5 0.9 0.5; 1 0.7 0.4]; % μπλε, πράσινο, πορτοκαλί

% Προσθήκη τιμών πάνω από κάθε μπάρα
text(1:length(acc), acc + 2, ...
    string(acc) + "%", ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2]);

% Άξονες και τίτλοι
set(gca, 'XTickLabel', models, 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Model', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Accuracy (%)', 'FontSize', 16, 'FontWeight', 'bold');
title('Comparison of Model Accuracies on Test Data', 'FontSize', 18, 'FontWeight', 'bold');

ylim([0 100]);
grid on;
box off;