

function [data] = normalizeData(data)

classe = data(:, end);
data(:, end) = [];

[tam_base, qt_features] = size(data);

for i=1:qt_features
    mymax = max(data(:, i));
    mymin = min(data(:, i)); 
    for j=1:tam_base
        data(j, i) = (data(j, i) - mymin)/(mymax-mymin);
    end
end

data(:, end+1) = classe;

end