Report for the Assigment1
==========

Dingcheng Yue(dyue2)

Overall accuracy:
For naive bayes without NA treatment: 73.2(6797)%
For naive bayes with NA treatment: 73.8(5621)%
For naive bayes provided by caret: 75.8(1699)%
For svm using svmlight: 75.4(902)%

For all the test, the accuracy is calculated with equal weight on false positive and false negative.
The cases are performed by 80/20 partition on training/test, and 10 times for each method.

As we can see that four cases does not differ significantly and if count the factor that of parition, I think they are equally good. However, I believe the overall performance is not good and I think there might be some reason for that. First, for svm, the feature space is small (only 8), and for naive bayes, there is highly probable that the feature space is also too small and each feature are not independent as the naive bayes's assumption claims.

