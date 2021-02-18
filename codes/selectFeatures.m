

function [data, escolhidos] = selectFeatures(data, thresh)

    classe = data(:, end);
    data(:, end) = [];
    [idx1,scores1] = fscchi2(data, classe);

    mymin = min(scores1);
    mymax = max(scores1);

    if (mymin < 0)
        scores1 = scores1 + ( -1*mymin );
    end

    scores1 = scores1/mymax;

    escolhidos = scores1 >= thresh;
    data = data(:, escolhidos'==1);

    data = cat(2, data, classe);


end