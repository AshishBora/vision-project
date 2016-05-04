load ../data/SUN/SUN_WS/SUN_WS_TEST_ANNOTATIONS.mat;
load probs_test.mat;
[averagePrecision, averageAttPrecision, truePositives, falsePositives] = computeResults(x, testSetAnnotations);

disp(averagePrecision)