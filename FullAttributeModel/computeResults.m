% -------------------------------------------------------------------------
% Function to compute total and average precision (top-k dynamic)
% Author: Sukrit Shankar 
% -------------------------------------------------------------------------
function [averagePrecision, averageAttPrecision, truePositives, falsePositives] = ...
        computeResults (results,testAnnotations)

% -------------------------------------------
annotationCertaintyThreshold = 0.4;
% For SUN, make it 0.4 to see 10% improvement
% This shows how much the attriute groundtruth annotations can be ambiguous
% -------------------------------------------
totAnnotations = 0; 
truePositives = zeros(size(testAnnotations)); 
falsePositives = zeros(size(testAnnotations)); 
for i = 1:1:size(testAnnotations,1)  % Number of Test Images 
    noOfAttributesToDetect = length(find(testAnnotations(i,:) > annotationCertaintyThreshold)); % For this image
    totAnnotations = totAnnotations + noOfAttributesToDetect; 
    [~,sortedIndices] = sort(results(i,:),'descend'); 

    count = 1; r = 1; 
    for j = 1:1:size(testAnnotations,2) 
        if (testAnnotations(i,sortedIndices(1,r)) > 0)
            truePositives(i,sortedIndices(1,r)) = 1; 
        else
            falsePositives(i,sortedIndices(1,r)) = 1; 
        end
        count = count + 1; r = r + 1; 
        if (count > noOfAttributesToDetect)
            break;
        end
    end

    clear noOfAttributesToDetect results_sorted sortedIndices;
end

% Compare
tp = length(find (truePositives == 1)); 
fp = length(find (falsePositives == 1)); 
averagePrecision = tp / (tp + fp); 

for i = 1:1:size(testAnnotations,2)
    tp = length(find (truePositives(:,i) == 1)); 
    fp = length(find (falsePositives(:,i) == 1)); 
    averageAttPrecision(1,i) =  tp / (tp + fp) * 100; 
end

% -------------------------------------------------------------------------
