# Language Modeling

#### Open in directory containing the following files:
* main.py
* train-Spring2022.txt
* test.txt

#### Compile using python main.py
#### outputs files:
* test-p.txt (preprocessed test corpus)
* train-p.txt (preprocessed training corpus)

###### Pre-processing files
<ol>
  <li>Pad each sentence in the training and test corpora with start and end symbols(&lt;s&gt; and &lt;s\&gt; respectively.).</li>
  <li>Lowercase all words in the training and test corpora. Note that the data already has
been tokenized (i.e. the punctuation has been split off words)</li>
  <li>Replace all words occurring in the training data once with the token &lt;unk&gt;. Every word
in the test data not seen in training should be treated as &lt;unk\&gt;.</li>
  <li>Fourth item</li>
</ol>

###### Training Models
<ol>
    <li>A unigram maximum likelihood model.</li>
    <li>A bigram maximum likelihood model.</li>
    <li>A bigram model with Add-One smoothing.</li>
</ol>

