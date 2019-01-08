# project4-SentimentAnalysis

## Problem Statement / Objective
The problem assigned was to predict the sentiment of three splits of IMDB movie reviews provided to an
AUC level of more than 0.96 for the lowest split, while not exceeding a vocabulary count of 3000.

The problem of predicting the sentiment was split into the following steps in a Jupyter notebook (Appendix
B). The movie reviews dataset and the splits dataset were read as input and the text was pre-processed to
remove spurious data, this clean text was then converted to word count vectors with some transformation,
and finally a vocabulary was selected from these results and saved to a text file. A logistic regression model
was then applied to the three data splits provided which were limited by the vocabulary selected, and a result
file was generated for each split as output, and the AUC was calculated.

The Python code submitted is effectively a subset of the notebook just described. It reads the movie reviews
dataset, the splits dataset, and the vocabulary text file as input, applies the same preprocessing, tokenization,
transformations and logistic regression model on the split specified by the ‘s’ variable in the code (as
required), and a submission text file is created as the output.

The technology used for this project was Jupyter Notebooks, Python, with Pandas, Numpy, Scipy, and
Scikit-learn libraries. The modeling was run on an iMac quad-core i7 4.2GHz, 40GB RAM, and the runtime
of the Python code is approximately 20 seconds. The metric used for this report is ROC AUC as required.

## Method
The method is described in sequence in the following sections, although the actual process was highly
iterative with changes in one area having impacts in other areas. For example initially the stop words list had
a significant impact, but once the vocabulary limit was in place, stop words made only a minor improvement
and the list had to be refined.

### Text Pre-processing
A visual inspection of the movie reviews dataset revealed some <br> html tags that needed to be removed,
and identified options to remove numbers, punctuation etc. However a baseline was set by initially just
removing the html tags. Later attempts to improve performance by removing numbers, stemming,
lemmatization etc. actually reduced model accuracy so were dropped, leaving only the very simple html tag
removal preprocessing.

### Tokenization, Transformation and Filtering
Initially words were simply tokenized as-is, with unigrams and simple word count vectors used to establish a
baseline. TFIDF transformation was found to significantly improve performance, and using a basic set of
stop words also provided a minor improvement so both were included in the model. Further experimentation
found that using up to bigrams improved performance, as did establishing a minimum document frequency
of 20 documents, and a maximum document frequency of 30%. However filtering tokens by length actually
reduced performance so this was dropped. Also using n-grams greater than 2 did have good performance but
at the expense of much longer run-times and memory usage so the benefit was not worthwhile.

### Vocabulary Selection
The method used to select the optimal vocabulary was to take the whole movie review dataset and apply the
pre-processing, filtering and transformations as described above, and then to apply a statistical method (ttest)
to select the most ‘influential’ words. The t-test applied (as recommended) compares a dataset of
positive sentiment reviews with a dataset of negative sentiment reviews and generates t-statistics measuring
how significantly different the sentiment is for each token. Tokens were then sorted by t-statistic magnitude
and the top 2900 most significant tokens were used to create the vocabulary.

Scaled t-statistics were written to a CSV file for the word importance data visualization discussed below.
Prior using the t-test approach, L1 regularized logistic regression was used and tokens with non-zero
coefficients were used to determine the vocabulary, however while the results met the 0.96 AUC, they were
not as good as the t-test method results, and this method was not as statistically justified as using the t-test.

### Modelling
Logistic Regression was selected from a number of other classifiers (such as multinomial naïve Bayes,
XGBoost, SVM RBF kernel, SVC) for its excellent performance, and very fast run times for the volume of
data provided. All other models performed worse and their run-times were many times slower. A baseline
model using L1 regularization with C=1 gave reasonably good results, however tuning the model to use L2
regularization and setting C = 20 made a substantial improvement in AUC.

The model configuration and pre-processing settings are listed in Appendix A1.

## Results
The minimum AUC across all splits is greater than the 0.96 required, and while the vocabulary is 2900
words (unigrams and bigrams), slightly less than this is used for the different splits due to words not
appearing in all splits.

Model Performance Summary
                        AUC
Performance for Split_1 0.96685
Performance for Split_2 0.96580
Performance for Split_3 0.96612
Vocabulary Size 2900

Experiments with vocabulary size showed that the best minimum AUC is obtained using a vocabulary size of 2999
(given the constraint on this of 3000), however the baseline minimum AUC of 0.96 across all three splits can be
achieved with as few as 1200 words. The 2900 limit was chosen as a balance between these (see Appendix A3 below).

## Conclusion
The performance table above shows a minimum AUC over the 3 splits of 0.96580, well over the 0.96
required. The model and pre-processing required to achieve this are both fairly simple, but it was a challenge
to both select the right model/methods, and to tune the various parameters to achieve the target.

It was possible to achieve higher AUC values than shown, but only by sacrificing performance on one split
to benefits the others. For example a different model had a performance of 0.961 on split one, and around
0.97 on splits 2 and 3. However the final model chosen had the highest minimum AUC based on the
vocabulary size. It was also possible to increase the performance by using a vocabulary size of 2999 instead
of 2900, but the improvement was very small, and so it seemed better to go with the lower vocabulary size
which is well clear of the maximum allowed.

Future Steps: a broad range of models, methods and settings were evaluated to get to the current setup, so
further adjustments on this setup could easily result in over-tuning. For example the stop words list could
adjusted to contain only those words which improve model performance, but that would be adjusting the
model to be highly specific to the current dataset. Potentially evaluating more complex models such as
neural nets could result in improvements, also if additional training data was available that may also allow
the model to be tuned to be a little more general.

In summary, the simplicity of the overall approach and model seems to indicate that is has not been too overtuned
to the dataset provided, but still provides a high AUC score.
