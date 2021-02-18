

function [results] = classify(inputTable, predictors, isCategoricalPredictor, option)


response = inputTable.Y;

switch option
    case 'ensemble'
        disp('Ensemble Classification')
        tEnsemble = templateTree(...
            'MaxNumSplits', 3, ...
            'NumVariablesToSample', 'all');
        cEnsemble = fitcensemble(...
            predictors, ...
            response, ...
            'Method', 'LogitBoost', ...
            'NumLearningCycles', 186, ...
            'Learners', tEnsemble, ...
            'LearnRate', 0.9010484321649903, ...
            'ClassNames', [1; 2]);
        
        
        trainedClassifier.ClassificationEnsemble = cEnsemble;
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        ensemblePredictFcn = @(x) predict(cEnsemble, x);
        trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
        
        
        partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 10, 'nprint', 1);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'knn'
        disp('KNN Classification')
        classificationKNN = fitcknn(...
            predictors, ...
            response, ...
            'Distance', 'Cityblock', ...
            'Exponent', [], ...
            'NumNeighbors', 1, ...
            'DistanceWeight', 'Inverse', ...
            'Standardize', true, ...
            'ClassNames', [1; 2]);
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        knnPredictFcn = @(x) predict(classificationKNN, x);
        trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationKNN = classificationKNN;
        partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'svm'
        disp('Support Vector Machine Classification')
        classificationSVM = fitcsvm(...
            predictors, ...
            response, ...
            'KernelFunction', 'gaussian', ...
            'PolynomialOrder', [], ...
            'KernelScale', 3.923215001471216, ...
            'BoxConstraint', 971.7357302464671, ...
            'Standardize', false, ...
            'ClassNames', [1; 2]);
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        svmPredictFcn = @(x) predict(classificationSVM, x);
        trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationSVM = classificationSVM;
        
        partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'trees'
        disp('Decision Tree Classification')
        classificationTree = fitctree(...
            predictors, ...
            response, ...
            'SplitCriterion', 'gdi', ...
            'MaxNumSplits', 8, ...
            'Surrogate', 'off', ...
            'ClassNames', [1; 2]);
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        treePredictFcn = @(x) predict(classificationTree, x);
        trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationTree = classificationTree;
        partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'discriminant'
        disp('Discriminant Analysis Classification')
        classificationDiscriminant = fitcdiscr(...
            predictors, ...
            response, ...
            'DiscrimType', 'diagLinear', ...
            'FillCoeffs', 'off', ...
            'ClassNames', [1; 2]);
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
        trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
        
        partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'bayes'
        disp('Naive Bayes Classification')
        distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
        distributionNames(isCategoricalPredictor) = {'mvmn'};
        
        if any(strcmp(distributionNames,'Kernel'))
            classificationNaiveBayes = fitcnb(...
                predictors, ...
                response, ...
                'Kernel', 'Normal', ...
                'Support', 'Unbounded', ...
                'DistributionNames', distributionNames, ...
                'ClassNames', [1; 2]);
        else
            classificationNaiveBayes = fitcnb(...
                predictors, ...
                response, ...
                'DistributionNames', distributionNames, ...
                'ClassNames', [1; 2]);
        end
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
        trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
        partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2), partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    otherwise
        disp('Invalid option selected')
        
end

results = {val1, val2, fpr, tpr};

end