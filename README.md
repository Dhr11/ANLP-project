# ANLP-project

Google Docs:
https://docs.google.com/document/d/103rxbocFTvP155Tz1k2u0NcnKwBF1_YY-zOrYCe-55Y/edit

## Introduction

With large amount of data being uploaded daily on web through news articles, social
media, blogs etc, it’ll only get harder for one to skim through the textual information. One can
imagine how text summarization can be vital for people to reduce time spent reading and to companies to sort and select from gigabytes of text information. Our main objectives from this project
were,
• To practice and experiment with NLP skills learnt in class
• Learn about automatic and extractive summarization

• Successfully Build an Abstractive Text summarizer
• Learn about different evaluation metrics for text summarization
To fulfill these objectives, our first step was to read and research about this task. Finally,
we chose a recent work don by Nallapati et al in 2016, as our main inspiration. The reason for
selecting this paper, was due to its impact in the field as it was one of the first to build a reasonable abstractive summarizer.

## Data

### Amazon Review data

The data was provided by Stanford Network Analysis Project (SNAP) and was made available on
Kaggle This dataset consists of reviews of fine foods from amazon. The data span a period of more
than 10 years from Oct 1999 to Oct 2012, including all 568,454 reviews up to October 2012. It
consists reviews of 256,059 user and of 74,258 products. 260 users have given more than 5 reviews.
Reviews include product and user information, ratings, and a plain text review. It also includes
reviews from all other Amazon categories.
The following is one of the example from the data-set
Text:
I have bought several of the Vitality canned dog food products and have found them all to be
of good quality. The product looks more like a stew than a processed meat and it smells better.
My Labrador is finicky and she appreciates this product better than most
Summary:
Good Quality Dog Food

### Daily Mail data

The CNN/Daily Mail dataset as processed by Nallapati et al. (2016) was used for evaluating
abstractive text summarization. The dataset has set of online news articles (781 tokens average)
and corresponding multi-sentence summaries (3.75 sentences average). Common evaluation metrics
in different papers are full-length F1-scores of ROUGE-1, ROUGE-2, ROUGE-L, and METEOR
(optional).
After some exploration we selected Daily Mail, as the sole dataset for our purpose. This was because, Daily Mail had more than 200k samples and CNN had many instances with noise in the
text. We split the dataset as 80/20 ratio of Train and Test dataset.
The paper by Rush et al, was one of the first ones to tackle abstractive summarization. There they
took the first word of each sentence and headline as text-summary pairs for the dataset. This technique eventually has been applied in different fashions over time. One of the common approaches
is to use the first few sentences as text and corresponding summary as target.
Here in Daily Mail dataset, we have set of articles with multiple sentences and corresponding we
have set of 3-4 highlights. For the purpose of our task, we have chosen the first two sentences as
the text and the first highlight as the target summary. This is a relevant tactic , especially for
this dataset because journalists try to convey important information at the start of the articles.
Naturally, this intuition would fail for some instances, but for the sake of reducing computational
complexity, it was a necessary step in our pipeline.
The following is one of the example from the data-set
Text:
The family of Stephen Collins fear the 7th Heaven star is on the brink of suicide over shocking
allegations he molested several underage girls.The actor , who played a beloved pastor on the hit
family friendly show , has been left distraught after his taped confessions made during a marriage
therapy session were made public Now those closest to the 67-year-old are deeply concerned the
star has hit rock bottom and fear he may harm himself .
Summary:
Actor Stephen Collins has hit rock bottom since the child molestation claims have become
public

## Pre processing

Tokenization is the process by which big quantity of text is divided into smaller parts called tokens.
These tokens are very useful for finding patterns as well as is considered as a base step for stemming
and lemmatization.
It can also be provided as input for further text cleaning steps such as punctuation removal,
numeric character removal or stemming. Machine learning models need numeric data to be trained
and make a prediction. Word tokenization becomes a crucial part of the text (string) to numeric
data conversion.
For eg: \Mr.Smith is a good writer" => [\Mr.Smith", "is", \a", \good", \writer"]
In our project, we initially tokenized the words using NLTK’s \word tokenize" method. This
ensured that all the punctuations and salutations were handled unlike the keras tokenizer which
only seperates a token based on a space.The tokenised words from NLTK are then contracted with
a space in between them such that the Keras tokenizer can be used for tokenizing words before
they were fed to our model.
For eg: \Mr.Smith is a good writer" => [\Mr.Smith", \good", \writer"]
We also improvised the tokenisation as we came across examples which the NLTK did not
handle well for tokenising. We used Spacy module’s nlp function to both remove punctuation and
remove stop words. The text was cleaner than the output from NLTK.
A stop word is a commonly used word (such as \the", \a", \an", \in") that a search engine has
been programmed to ignore, both when indexing entries for searching and when retrieving them as
the result of a search query. We would not want these words taking up valuable processing time.
For this, we can remove them easily, by using a list of words that you consider to be stop words.
NLTK has a list of stop-words and we used it to remove them from our tokenized sentences.
We expanded contractions to make the corpus more consistent using a library called pycontractions. Contractions are shortened version of words or syllables. They often exist in either written
or spoken forms in the English language. These shortened versions or contractions of words are
created by removing specific letters and sounds. In case of English contractions, they are often created by removing one of the vowels from the word. Examples would be, do not to don’t and I would
to I’d. Converting each contraction to its expanded, original form helps with text standardization.
Eg you’ll was expanded to ’you all’.
To reduce training time, we only kept words with frequency above a threshold (Text => 2;
Summary => 2). Our model (with embeddings) will not be as good at capturing the context
of words that occur less frequently. For this reason, its better to keep them out and save some
training time. We also set a threshold for the sentence length. The training texts should be 400
and summary should be 20.

## Findings and Observations

• Amazon more short and keyword extraction
The distribution of Amazon review summaries is shown in the diagram below. As the diagram
indicates, we have very short summaries with most of them being around 4-5 letter summary.
At this length, as one would suspect, we might not always have proper sentences as the
target summary. This can also be looked at, by considering the domain of the dataset, as
its Amazon review summary, it would ideally have keywords present in summary to grab the
user/customer’s attention.
• Daily mail, has set of articles which can be exclusive in their domain
Daily Mail comprises of a bit over 200,000 text articles-summary pairs. However, as these
are news articles written by journalists, there’s good chance for many words to be rare in the
vocabulary, especially Proper Nouns. Due to this, its possible for validation set to have many
words not present in the train dataset, and thus we can loose out on evaluation scores. To
handle this, in future we can look to generalize the concepts in the sentence.
Example:
Text: Police evidence about the links between murdered Russian spy Alexander Litvinenko
and British intelligence will be heard in secret , a court was told yesterday . Mr Litvinenko ,
43 , was poisoned with radioactive polonium-210 allegedly while having tea with former KGB
agents Andrei Lugovoy and Dmitry Kovturn at a central London hotel in 2006 .
Target Summary: Alexander Litvinenko , 43 , was poisoned drinking tea in London in 2006
Here the full name of the person is expected in the final summary. This can be an issue in
our model, and can be handled by generalization or pointer generator networks.
• More length of summary, leads to more abstractive nature than extractive
There’s a stark difference between the distributions of lengths of summaries in Daily Mail
dataset and Amazon reviews. On further inspection, as one expects that when the sentence
length is longer, say more than 10 words/tokens, we expect more Part of Speech to be part
of the text, than just the vital nouns in the main text. Below we have an example from the
Daily Mail dataset. Here, if we run a Named Entity Tagger, we would have only the exact
location as the tagged entities, and rest would be non specific words which can be flexible for
sumarization. Also if we check the output of dependency graph from Stanford NLP tool, we
can see we have a lot of text information like subjectr, object and modifiers, and thus we can
expect grammatically better constructed sentences, when the length is increased.
Example Target Summary: The girls went to a high school football game in Oscoda ,
Michigan but never showed up at a party they were supposed to attend
• Evaluation metric in this field is still prone to bias for extractive summarization
The usual metrics used for summarization, such as ROUGE and METEOR, have some limitations:
{ They emphasize on content selection and but fail to value for other possible key aspects
like coherence and fluency.
{ Datasets such as CNN/DailyMail and Gigaword provide only a single reference. But as
summarization is subjective, we have correspondingly low agreement between annotators, thus the metrics were designed to be used with multiple reference summaries per
input.
{ To quantify content selection, they are based on lexical overlap, which resembles more
to extractive summarization, as abstractive summarization can also have same meaning
with no lexical overlap.
Therefore, many new papers claiming state of the art scores, based only on these metrics are questionable.

## Results
We initially tried getting the most accurate model architecture . For this we used short sentences
of Amazon reviews. Since the training process itself takes huge time we purposefully considered a
Amazon dataset which had short phrases as summary.
We had the following approach,
1. Tried model without embedding initialisation to see the accuracy when the embedding matrix
is trained from scratch
2. Added pretrained glove embeddings to the model and saw the accuracy improve
3. Added bidirectional LSTM to capture the sequence in forward and backward
4. Once we got the best performing architecture we moved to CNN- Daily Mail dateset which
had longer sentences and text input. This took us 26 Hours to train and had about 20000
data points.


Model               ROUGE-1 ROUGE-2 ROUGE-L
Tan et al.          38.1    13.9    34.0
Nallapti et al.     35.46   13.3    32.65
Chen et al.         12.12   3.6     11.34
Our Model           24.1    9.2     20.9
