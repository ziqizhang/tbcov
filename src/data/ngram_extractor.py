from nltk import WordNetLemmatizer, PorterStemmer
import re, functools,nltk, datetime, logging, sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd, numpy as np
import csv

def pre_process(s):
    s = re.sub('[^0-9a-zA-Z]+', '*', s)


    text= " ".join(re.split("[^a-zA-Z]*", s)).strip()
    return text
def tokenize(tweet, stem_or_lemma=0):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and normalizes tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens=[]
    if stem_or_lemma==0:
        for t in tweet.split():
            if len(t)<4:
                tokens.append(t)
            else:
                tokens.append(stemmer.stem(t))
    elif stem_or_lemma==1:
        for t in tweet.split():
            if len(t)<4:
                tokens.append(t)
            else:
                tokens.append(lemmatizer.lemmatize(t))
    else:
        tokens = [str(t) for t in tweet.split()] #this is basic_tokenize in TD's original code
    return tokens

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("english")
logger = logging.getLogger(__name__)

ngram_vectorizer = TfidfVectorizer(
    # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    #tokenizer=functools.partial(tokenize, stem_or_lemma=1),
    analyzer="word",
    preprocessor=None,
    ngram_range=(2, 3),
    stop_words=stopwords,  # We do better when we keep stopwords
    use_idf=True,
    smooth_idf=False,
    norm=None,  # Applies l2 norm smoothing
    decode_error='replace',
    token_pattern=r"(?u)\b\w[\w-]*\w\b",
    max_features=500000,
    min_df=1,
    max_df=0.99
)


def get_ngram_tfidf(texts):
    logger.info("\tgenerating n-gram vectors, {}".format(datetime.datetime.now()))
    tfidf = ngram_vectorizer.fit_transform(texts).toarray()
    logger.info("\t\t complete, dim={}, {}".format(tfidf.shape, datetime.datetime.now()))
    vocab = {i: v for i, v in enumerate(ngram_vectorizer.get_feature_names())}
    return tfidf, vocab


if __name__ == "__main__":
    infile=sys.argv[1]
    col_text=sys.argv[2]
    col_topic=sys.argv[3]
    outfolder=sys.argv[4]

    df = pd.read_csv(infile, header=0, delimiter=',', quoting=0, encoding="utf-8")

    current_topic=None
    current_text=None
    texts={}

    for index, row in df.iterrows():
        text=row[col_text]
        topic=row[col_topic]

        if current_topic is None:
            current_text= str(text)
            current_topic=topic
        elif current_topic!=topic:
            texts[current_topic]=current_text
            current_text=str(text)
            current_topic=topic
        else:
            text+="\n\n"+str(text)

        if index%5000 ==0:
            print(index)
    if current_text is not None:
        texts[current_topic]=current_text

    topics={}
    text_values=[]
    topic_id=0
    for k, v in texts.items():
        topics[topic_id]=k
        topic_id+=1
        text_values.append(v)
    tfidf, vocab=get_ngram_tfidf(text_values)

    #go through each document, output
    index=0
    for entry in tfidf:
        indexes = np.nonzero(entry)[0]
        if len(indexes)>0:
            topic = topics[index]
            ngrams={}
            for i in indexes:
                tfidfscore=entry[i]
                ngram = vocab[i]
                ngrams[ngram]=tfidfscore
            ngrams = {k: v for k, v in sorted(ngrams.items(), key=lambda item: item[1], reverse=True)}

            # output
            print("For {} total={}".format(topic, len(ngrams)))
            for k, v in ngrams.items():
                print("\t{} \t\t {}".format(k, v))

            with open(outfolder + "/{}.csv".format(topic), 'w', newline='\n', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file, delimiter=',',quotechar='"', quoting=csv.QUOTE_ALL)
                for k, v in ngrams.items():
                    writer.writerow([k, v])

        index+=1

    #save output