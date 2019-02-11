import re   # Regular expression library
import nltk
from nltk.tokenize import PunktSentenceTokenizer   # already train sentence tokensier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
import random


def cleantext(raw_text):
    """clean raw text:
           1. remove markups
           2. remove some special characters
           3. lower all"""

    #1. remove markups
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_text)

    #2. remove special characters
    element2remove = [r"\.\s", r'"', r"'", r"\!", r"\?", r"\,",
                      r"\:", r"\(", r"\)", r"\n", r"\*", r"\]", r"\[", "&nbsp;",
                      "&lt;", "&amp", r"\//", "=", r"\{", r"\}", r"\&gt;", ";",
                      r"\%"]
    for e in element2remove:
        cleantext = re.sub(e, ' ', cleantext)

    #3. lower all (remove Capital letters)
    cleantext = cleantext.lower()

    return cleantext


def cleantags(tags):
    """clean raw tags: remove markups and lower all"""

    tags = tags.replace("><", ' ')
    tags = tags.replace("<", '')
    tags = tags.replace(">", '')
    tags = tags.lower()

    return tags


def process_content(text_body, stopwords):
    """
    Tokenize a text ('text_body') by words and keep the nouns(NN) and adjective
    or numeral, ordinal (JJ) then remove stop words ('stopwords').
    Inputs:
        - text_body: string
        - stopwords: list of string
    Outputs:
        - return a list of words selection
        """

    tokenizer = PunktSentenceTokenizer()
    tokenized = tokenizer.tokenize(text_body)
    all_nouns = []
    for sentence in tokenized:
        words = sentence.split()
        # get the information of the word (noune, verb,etc..)
        tagged = nltk.pos_tag(words)
        for w in tagged:
            if ((w[1] == "NN")or(w[1] == "JJ")) and (w[0] not in stopwords):
                all_nouns.append(w[0])
    return all_nouns


def vocabulary(all_words, all_tags, Nw, Nt):
    """ return a dictionary of vocabulary. It merges the top Nw words from
    all_words with the top Nt from all_tags. This format is needed as input of
    nltk vectoriser"""

    top_w = list(zip(*all_words.most_common(Nw)))[0]
    top_t = list(zip(*all_tags.most_common(Nt)))[0]
    # add top tags to this list
    top_voc = set(list(top_w) + list(top_t))

    # formating: list of for for vectorized input
    return dict(zip(top_voc, range(0, len(top_voc))))


def display_topics2(model, feature_names, n_top_words=25):
    """create a panda data frame of the top words present in each topics.
    Inputs:
        - model: fitted model, from where we will extract the results (numeric)
        - feature_names: list of string (words used in the model)
        - no_top_words: int, number of top words per topics we want to extract
    Outputs:
        - DataFrame
        """
    word_dict = {};
    for topic_idx, topic in enumerate(model.components_):
        word_dict["Topic%d" % (topic_idx)] = [feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
    return pd.DataFrame(word_dict).T


def tag_pred(model, vectorized_input, feature_names, top_t, nb_tag_pred=10,
             threshold=0.15):
    """
    Extract and format output of LDA or NMF models. Rather than returning
    topics, it return a list of words related to those topics.
    Inputs:
        - model: fitted sklearn model, from where we will extract the results
        (numeric).
        - vectorized_input: tf or tf-idf vectorized input
        - feature_names: list of string (words used in the model for
        vectorized_input)
        - top_t: list of str, all existing allowed tags.
        - threshold: topic frequency threshold to be considered as one of
        "mains" topics
        - nb_tag_pred: Number of proposed tag per document (len of tag_pred)
    Outputs:
        tag_pred: list of list, tags list per documents
    """

    # Topic df -----------------------------------------------------------------
    topic_df = display_topics2(model, feature_names)
    # associate each topic with a list of tags
    topics_kwords_df = topic_df.T #(topic_df.isin(top_t)*topic_df).T
    topic2tags_d = {}
    # tags_per_topic = []
    for topic in topics_kwords_df:
        tag_list = []
        for e in topics_kwords_df.loc[:, topic]:
            if e is not "":
                tag_list.append(e)
        topic2tags_d[topic] = tag_list

    # Create Document Vs Topic df ----------------------------------------------
    import numpy as npy
    model_output = model.transform(vectorized_input)
    topicnames = ["Topic" + str(i) for i in range(model.components_.shape[0])]
    docnames = ["Post" + str(i) for i in range(vectorized_input.shape[0])]
    df_document_topic = pd.DataFrame(npy.round(model_output, 2),
                                     columns=topicnames,
                                     index=docnames)

    # Tag predictions ----------------------------------------------------------
    tag_pred_l = []
    for post in df_document_topic.index:
        tags_post = []
        topics_proba = df_document_topic.loc[post, :]
        mask = topics_proba >= threshold
        topic_pred = list(df_document_topic.loc[post, mask].index)
        tot_proba = topics_proba[topic_pred].sum()

            # if no major topic in this post, propose just top 10 tags
        if len(topic_pred) == 0:
            tags_post = tags_post + top_t[0:nb_tag_pred].copy()
        else:
            for topic in topic_pred:
                # pic number of top elements ~ to proba of the topic
                nb_elements = int(round(topics_proba[topic]*10/tot_proba,0))
                tags_post = tags_post + topic2tags_d[topic][0:nb_elements].copy()
        tag_pred_l.append(tags_post)

    return tag_pred_l


def topic_score(model, vectorized_input, feature_names, top_t, document_tags, limited=True):
    """The scoring function is design for the purpose off assessing the quality of the predicted tag
    in the context of tags suggestion. For tis we will compute the ratio of real tags words present
    in the predicted list of tags.
    Inputs:
        - model: fitted sklearn model, from where we will extract the results (numeric).
        - vectorized_input: tf or tf-idf vectorized input
        - feature_names: list of string (words used in the model)
        - document_tags: list of list of str, each document has a list of real tags
        - top_t: top_t: list of str, all existing allowed tags.
        - vocab_list: list of str, used here as the allowed tags pool (if limited=True).
        - limited: indicate if the scoring is done on tags present in the vocabulary list (False: or all tags).
        True by default since we cannot predic tags that are not in the input vocabulary list.
    Output:
        - tag_pred: list of list, proposed tags.
        """

    prediction = tag_pred(model, vectorized_input, feature_names, top_t)

    tag_score = []   # list of score for each document (score = ratio of real tags present in the prediction)
    for i in range(len(document_tags)):
        score = 0
        count_e = 0
        for e in document_tags[i]:
            if ((limited==True) and (e in top_t)) or (limited==False):
                if e in prediction[i]:
                    score = score + 1
                count_e = count_e + 1
                tag_score.append(round(score/count_e,2))

    return sum(tag_score)/len(tag_score)


def raw2XY(text_df, tags_list, vocab_list, tfidf=False):
    """format text_df to be ready to be use in sklearn models(X, Y).
    X and Y would be construct as Term frequency tables (tf). However Y,
    by its nature, can only have one occurence of each tag for each topics.
    So it can be more comparable as a one hot encoded table.
    Inputs:
        - text_df, data frame with two colums:"Tags" and "Body".
        each row represent a document
        - tags_list: list of tags that will be used in the tf table of tags (Y)
        - vocab_list: list of words that will be used in the tf table of the text(X)
    Outputs:
        - X, Y: Term Frequency tables """

    # Y
    tag_vectorizer = CountVectorizer(vocabulary=tags_list,
                                token_pattern=r"(?<=<).+?(?=>)")  # keep c#, .net, ...
    Y = tag_vectorizer.fit_transform(text_df.loc[:, "Tags"])
    Y = pd.DataFrame(Y.A,
                     columns=tag_vectorizer.get_feature_names(),
                     index=text_df.index
                    )

    # X
    corpora = text_df.loc[:, "Title"] + text_df.loc[:, "Body"]
    
    if tfidf == True:
        corpus_vectorizer = TfidfVectorizer(vocabulary=vocab_list,
                                            token_pattern=r"[a-zA-Z.0-9+#-_/]*[^.\s]")    
    else:
        corpus_vectorizer = CountVectorizer(vocabulary=vocab_list,
                                            token_pattern=r"[a-zA-Z.0-9+#-_/]*[^.\s]")  # keep c#, .net, ...
        
    X = corpus_vectorizer.fit_transform(corpora.apply(cleantext))
    X = pd.DataFrame(X.A,
                     columns=corpus_vectorizer.get_feature_names(),
                     index=text_df.index
                    )

    return X, Y


def strat_binary_multilabels(Y, min_occurences=200):
    """stratification sampling on one-hot encoded features
    Inputs:
        - Y: One hot encoded DataFrame
        - min_occurence: minimum occurence of each features
    Ouputs:
        - a list of index that can be used to sample
        """
    # df initialisation
    Y_strat = Y.iloc[0:100,:]
    Y_temp = Y.iloc[100:,:]

    # iteration through each features
    for var_name in reversed(Y_temp.columns):  # reversed to start from less present tags
        # print(var_name)
        mask = Y_temp.loc[:,var_name] == 1
        idx = Y_temp.loc[mask, :].index

        n_in = Y_strat.loc[:,var_name].sum()
        idx_sel = random.sample(list(idx), max(0,(min_occurences-n_in)))

        Y_strat = pd.concat([Y_strat, Y_temp.loc[idx_sel,:]])
        Y_temp = Y_temp.drop(idx_sel)
    return Y_strat.index


def supervised_predict(model, X_test, labels, n_words=10, output_format="df"):
    """based on prediction probability return a prediction.
    rather than returning (standard model):
        - 1 if probability is superior to 50%
        - 0 otherwise
    in this supervised_predict (this function) the predicion is:
        - 1 if it belong to the n_words with highest probability
        - 0 otherwise 
    Libraries:
        - pandas as pd
    Inputs:
        - model: already fitted classification model. Model should have a predict_proba() method
        - X_test: array, vectorised inputs
        - labels: list, labels names on the Y array (prediction of the model)
        - n_words: number of words return by the predition
        - output_format: DataFrame if set to "df", list of list if set to "list".
    Outputs:
        - Y_pred: pandas data frame, one hot encoded prediction
        if output format set to list it return a list of list of tags"""
    
    proba = model.predict_proba(X_test)
    proba_df = pd.DataFrame(0, 
                           index=X_test.index, 
                           columns=labels)
    for i, c in enumerate(proba_df.columns):
        proba_df.loc[:,c] = list(zip(*proba[i]))[1]

    Y_pred = pd.DataFrame(0, 
                           index=X_test.index, 
                           columns=labels)
    if output_format == "df":
        for idx in proba_df.index:  # quite long time loops to optimize if possible
            col = proba_df.loc[idx, :].nlargest(n_words).index
            Y_pred.loc[idx, col] = 1
        return Y_pred
    
    elif output_format == "list":
        pred_l = []
        for idx in proba_df.index:  # quite long time loops to optimize if possible
            col = proba_df.loc[idx, :].nlargest(n_words).index
            pred_l.append(Y_pred.loc[idx, col].columns)
        return pred_l



