#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
******************* AUTHORSHIP *************************
Created Date:        January 7, 2017
File name:           topic_modeling.py
Operating System:    Mac OS X v10.9.5 (2 core processor)
Language:            Python 2.7.3
Author:              Jeevith Bopaiah
Email:				 jeevith.bopaiah@uky.edu

******************** DESCRIPTION ************************
This was built as a part of VPR Office of Research's initiative to understand
the current research capabilities of the university and expand their research
expertise in other promising avenues of research.

In this project, we have built a tool that identifies the research areas based
on the ongoing research activity and ranks these areas based on their popularity
among the university's research community.

It uses the grant proposal abstracts submitted to OSPA to build an unsupervised
learning model that uses the features present in the language to cluster the proposals
based on its research area. These identified research areas are ranked based on the
number of proposals related to a particular domain.

"""

from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pymongo import MongoClient
from collections import Counter, OrderedDict
import operator
import pickle
import pandas as pd
import re
import nltk
import pymssql
import os
import sys
import string
import datetime
import threading

HOSTNAME = ''
USER = ''
PASSWORD = ''
DATABASE_NAME = ''
# sets up an interface for accessing the mongoDB database
client = MongoClient('localhost', 27017)
db = client.TopicModelDB
db_collection = db.topic_model_collection
db_ranked_topic_ids_collection = db.ranked_topics

# sets up an interface for accessing sql server database
conn_abstract_search = pymssql.connect(host=HOSTNAME, user=USER, password=PASSWORD, database=DATABASE_NAME)
cur_abstract_search = conn_abstract_search.cursor()

# invokes a stemmer module to perform the stemming task
stemmer = SnowballStemmer("english")

# retrieves the abstracts from the OSPA database between the start and end date
def getdata(start, end, h_name, u_name, pword, db_name):
    if start > end:
        start, end = end, start
    conn = pymssql.connect(host=h_name, user=u_name, password=pword, database=db_name)
    cur = conn.cursor()
    cur.execute("<SQL query to fetch the abstracts for a given period(start-date and end-date) from the database>")
    grants = cur.fetchall()
    cur.close()
    dataset = []
    for line in grants:
        if len(line[4]) > 49:   # to ensure the abstracts used for building the model have atleast 50 words.
            ukrf = line[0].encode('utf-8').strip()
            title = line[1].encode('utf-8').strip()
            keywords = line[2].encode('utf-8').strip()
            topics = line[3].encode('utf-8').strip()
            abstract = line[4].encode('utf-8').replace('\n',' ').replace('\r', ' ').strip()

            dataset.append([ukrf, title, keywords, topics, abstract])
    return dataset

# tokenizes the text into individual words and stems these words to transform it into its root word.
def tokenize_and_stem(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # stemming each word into its root word
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# tokenizes the text into individual words
def tokenize_only(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# inserts the final results(topic ranks, ukrf, topic words) into SQL database
def insert_result_sql(topic_dictionary, ukrf_dictionary, start, end):

	# the time the result was generated
    date_now = datetime.datetime.now()
    cur_abstract_search.execute("SELECT * FROM TopicIterations WHERE StartDate = CAST('" + str(start) + "-01-01' AS Date) and EndDate =  CAST('" + str(end) +"-12-31' AS Date)")
    results = cur_abstract_search.fetchall()

    # checks if the data for the given (start-end) period already exists in the database.
    if len(results) == 0:

    	# inserts start-date, end-date and run-date(result generation date) into the table "TopicIterations"
        cur_abstract_search.execute("INSERT INTO TopicIterations VALUES (CAST('" + str(start) +"-01-01' AS Date), CAST('" + str(end) +"-12-31' AS Date), CAST('" + str(date_now) + "' AS Date)) ")
        conn_abstract_search.commit()
        cur_abstract_search.execute("Select scope_identity()")
        rec_id = cur_abstract_search.fetchall()
        rec_id = str(rec_id).replace("[(Decimal('", "").replace("'),)]", "").strip()

        # inserts the 'rank' and all 'topical words' associated with the rank into the table "TopicKeywords"
        for each_topic_rank, topic_list in topic_dictionary.items():
            for each_topic in topic_list:
                cur_abstract_search.execute("INSERT INTO TopicKeywords VALUES (%d, %d, %s)", (int(rec_id), int(each_topic_rank), each_topic))
                conn_abstract_search.commit()

        # inserts the 'rank' and all 'ukrf' classified as belonging to a particular rank into the table "TopicAwards"
        for each_rank, list_of_ukrf in ukrf_dictionary.items():
            for each_ukrf in list_of_ukrf:
                cur_abstract_search.execute("INSERT INTO TopicAwards VALUES (%d, %d, %s)", (int(rec_id), each_rank, each_ukrf))
                conn_abstract_search.commit()

# extracts the date when the last K-means clustering model was created in the mongoDB database
def last_record_updated():

	# checks if the model is created in the database
    if db_collection.count() > 0:
        record_data = db_collection.find({})
        date_of_last_insert = ''
        for recent_record in record_data:
            date_of_last_insert = recent_record['_id'].generation_time
            break
        return str(date_of_last_insert)
    else:
        return "Building Topic Model..."

# builds a tf-idf matrix from the tokenized words.
# builds an unsupervised learning model using the tf-idf as the features.
# predicts the new unseen text into one of the clusters using the trained model.
def extract_topics(start_date, end_date, num_topics, num_words, create_model_Flag, num_clusters, h_name, u_name, pword, db_name):
    list_of_abstracts = []
    list_of_abstracts_ids = []
    all_abstracts_stemmed = []
    all_abstracts_tokenized = []
    all_abstracts = []
    kmeans_model_filename = "Topic_modeling_K_means\\"+str(start_date)+"_"+str(end_date)+"_allTopicModel_all_abstracts.pickle"
    temp_kmeans_model_filename = "Topic_modeling_K_means\\"+str(start_date)+"_"+str(end_date)+"_allTopicModel_all_abstracts_temp.pickle"
    tfidf_model_filename = "Topic_modeling_K_means\\"+str(start_date)+"_"+str(end_date)+"_allTopicModel_tfidf.pickle"
    temp_tfidf_model_filename = "Topic_modeling_K_means\\"+str(start_date)+"_"+str(end_date)+"_allTopicModel_tfidf_temp.pickle"
    db_model_name = str(start_date)+"_"+str(end_date)+"_allTopicModel_all_abstracts"

    # checks if the ranking of the research areas for a given period already exists in the mongoDB database.
    # if it exists, the results are directly served from the database without running the prediction part again.
    if create_model_Flag == 0 and (db_ranked_topic_ids_collection.find({'start': start_date, 'end': end_date}).count() > 0):
        rec_obj = db_ranked_topic_ids_collection.find({'start': start_date, 'end': end_date})
        record = db_collection.find_one({"ModelName": db_model_name})
        all_topics = record['terms']
        for each_record in rec_obj:
            rank_clusters = each_record['rank_order']
            break
        rank_topics = {}
        for r, each_rank_id in enumerate(rank_clusters):
            r = r+1
            rank_topics[r] = [ str(each_word) for each_word in all_topics[each_rank_id][:num_words] ]
            if r == num_topics:
                break
        return OrderedDict(sorted(rank_topics.items()))

    all_abstracts = getdata(start_date, end_date, h_name, u_name, pword, db_name)

    # reads a list of stop-words and stores it in a dictionary data structure
    fp_stop_words = open("stop_words.txt", "r")
    list_of_stop_words = []
    for each_line in fp_stop_words:
        each_stop_word = each_line.strip()
        list_of_stop_words.append(each_stop_word)

    fp_stop_words.close()

    i = 1
    for each_abstract in all_abstracts:
        if each_abstract[1] == '':
            each_abstract_str = each_abstract[2] + " " + each_abstract[3] + " " + each_abstract[4]
        elif each_abstract[2] == '':
            each_abstract_str = each_abstract[1] + " " + each_abstract[3] + " " + each_abstract[4]
        elif each_abstract[3] == '':
            each_abstract_str = each_abstract[1] + " " + each_abstract[2] + " " + each_abstract[4]
        else:
            each_abstract_str = each_abstract[1] + " " + each_abstract[2] + " " + each_abstract[3] + " " + each_abstract[4]

        # converts the text into ascii characters
        text_in_ascii = set(string.printable)
        each_abstract_before_stopwords_removal = filter(lambda x: x in text_in_ascii, each_abstract_str)

        # removes all the stopwords present in the abstracts
        each_abstract_after_stopwords_removal = ''
        for each_word in each_abstract_before_stopwords_removal.split(" "):
            if not (each_word in list_of_stop_words) and each_abstract_after_stopwords_removal == '':
                each_abstract_after_stopwords_removal = each_word
                continue
            if not (each_word in list_of_stop_words):
                each_abstract_after_stopwords_removal = each_abstract_after_stopwords_removal + " " + each_word

        try:
            all_words_tokenized = tokenize_only(each_abstract_after_stopwords_removal)
            if len(all_words_tokenized) < 50:
                continue
            all_abstracts_tokenized.extend(all_words_tokenized)
            all_words_stemmed = tokenize_and_stem(each_abstract_after_stopwords_removal)
            all_abstracts_stemmed.extend(all_words_stemmed)
        except Exception as e:
            print ('Error Encountered: ' + str(e))
            sys.exit()
        i = i + 1
        list_of_abstracts.append(each_abstract_after_stopwords_removal)
        list_of_abstracts_ids.append(each_abstract[0].encode('ascii', 'ignore'))

	# If create_model_Flag = 0, it perform the prediction task, where it predicts the cluster for unseen data using the
	# previously built model. If create_model_Flag = 1, it performs the k-means model building process.
    if create_model_Flag == 0:
        fp_read_model = open(kmeans_model_filename, "rb")
        fp_read_tfidf_model = open(tfidf_model_filename, "rb")

        # reads the tfidf model from the file. While creating the model, we save the tfidf vector in a pickled file
        tfidf_model = pickle.load(fp_read_tfidf_model)
        tfidf_matrix = tfidf_model.transform(list_of_abstracts)

        # reads the k-means model from the file. While creating the model, we save the k-means model in a pickled file
        kmeans_model_predict = pickle.load(fp_read_model)

        # predicts the cluster for new unseen abstracts.
        result_clusters = kmeans_model_predict.predict(tfidf_matrix)

        # determines the frequency of each cluster for the purpose of ranking them.
        cluster_frequency = Counter(result_clusters)

        # fetches the words that describe a topic from mongoDB database.
        record = db_collection.find_one({"ModelName": db_model_name})
        all_topics = record['terms']

        # maps the abstracts and ukrf to their respective cluster as predicted by k-means
        mapping_abstract_ids_with_clusters = {'Abstracts': list_of_abstracts, 'Id': list_of_abstracts_ids, 'cluster': result_clusters}
        frame = pd.DataFrame(mapping_abstract_ids_with_clusters, index = [result_clusters] , columns = ['Id', 'cluster'])

        rank_ukrf_dictionary = {}
        rank_topics_dictionary = {}
        rank_topic_clusters = {}
        rank_list = []
        l=0

        # sort the prediction results in descending order based on the cluster frequency.
        # iterate over the result for num_topics times to get the top n(num_topics) researched areas
        for cluster_num, cluster_freq in sorted(cluster_frequency.items(), key=operator.itemgetter(1), reverse=True):
            rank_list.append(cluster_num)
            ukrf_list = []
            l = l + 1
            rank = l
            if l <= num_topics:
                if cluster_freq == 1:
                    ukrf_list.append(frame.ix[cluster_num]['Id'])
                else:
                    for ids in frame.ix[cluster_num]['Id'].values.tolist():
                        ukrf_list.append(ids)
                rank_ukrf_dictionary[rank] = ukrf_list

                # converting unicode to string
                rank_topics_dictionary[rank] = [ str(each_word) for each_word in all_topics[cluster_num][:num_words] ]

        rank_topic_clusters['start'] = start_date
        rank_topic_clusters['end'] = end_date
        rank_topic_clusters['rank_order'] = rank_list

        fp_read_model.close()
        fp_read_tfidf_model.close()

        try:
            insert_result_sql(rank_topics_dictionary, rank_ukrf_dictionary, start_date, end_date)

            # check if the results for the given period already exists in mongoDB.
            # If it doesn't exists, write these result into the database.
            # The next time when the request with the same period(start-date and end-date) is initiated
            # by the user, the results are fetched directly from database without having to predict again.
            if (db_ranked_topic_ids_collection.find({'start': start_date, 'end': end_date}).count() == 0) and (datetime.datetime.now().year != end_date):
                db_ranked_topic_ids_collection.insert_one(rank_topic_clusters)
        except Exception as e:
            print ("Error Message:" + str(e))

        return OrderedDict(sorted(rank_topics_dictionary.items()))

    # It builds an unsupervised learning model using k-means clustering algorithm.
    # It also builds a tf-idf vector that is required by the k-means algorithm
    else:

        if os.path.exists(kmeans_model_filename):
            os.rename(kmeans_model_filename, temp_kmeans_model_filename)
        if os.path.exists(tfidf_model_filename):
            os.rename(tfidf_model_filename, temp_tfidf_model_filename)
        try:
            fp_write_kmeans_model = open(kmeans_model_filename, "wb")
            fp_write_tfidf_model = open(tfidf_model_filename, "wb")
            vocab_frame = pd.DataFrame({'words': all_abstracts_tokenized}, index = all_abstracts_stemmed)

            # builds a tf-idf vector with specified parameters, particularly tuned for this type of data.
            tfidf_vectorizer = TfidfVectorizer(max_df=0.2, max_features=200000,
                                         min_df=0.01, stop_words='english',
                                         use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

            tfidf_matrix_fit = tfidf_vectorizer.fit(list_of_abstracts) #fit the vectorizer to synopses
            tfidf_matrix = tfidf_matrix_fit.transform(list_of_abstracts)

            # retrieves all the features generated by tf-idf vectorizer
            terms = tfidf_vectorizer.get_feature_names()

            # initializes the K-means model with certain parameters tuned for optimal clustering
            kmeans_model_create = KMeans(n_clusters=num_clusters, n_init=75, n_jobs=-1)
            kmeans_model_create.fit(tfidf_matrix)

            # saves the k-means and tf-idf model as a pickle file
            pickle.dump(kmeans_model_create, fp_write_kmeans_model)
            pickle.dump(tfidf_matrix_fit, fp_write_tfidf_model)

            fp_write_kmeans_model.close()
            fp_write_tfidf_model.close()

            # generates labels for the clusters created while training the model
            clusters = kmeans_model_create.labels_.tolist()

            # maps the abstracts and the ukrf with its respective cluster labels
            mapping_abstract_ids_with_clusters = {'Abstracts': list_of_abstracts, 'Id': list_of_abstracts_ids, 'cluster': clusters}
            frame = pd.DataFrame(mapping_abstract_ids_with_clusters, index = [clusters] , columns = ['Id', 'cluster'])

            # sorts the topical words within the clusters in the descending order of tf-idf scores
            order_centroids = kmeans_model_create.cluster_centers_.argsort()[:, ::-1]

            key_terms_dictionary = {}
            list_of_topics_all_clusters = []

            # for each cluster we store the top 100 words in mongoDB.
            # This allows us to train the model once and use it for prediction tasks as and when needed.
            for i in range(num_clusters):
                topic_list_per_cluster = []
                j = 0
                for ind in order_centroids[i, :200]:
                    K_means_terms = vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')

                    # we ignore any topical word less than 4
                    if len(K_means_terms) < 4:
                        continue

                    if K_means_terms not in topic_list_per_cluster:
                        topic_list_per_cluster.append(K_means_terms)
                        j = j + 1
                    if j == 100:
                        break
                list_of_topics_all_clusters.append(topic_list_per_cluster)

            key_terms_dictionary['ModelName'] = db_model_name
            key_terms_dictionary['terms'] = list_of_topics_all_clusters
        except Exception as er:
            if os.path.exists(kmeans_model_filename):
                os.remove(kmeans_model_filename)
                os.rename(temp_kmeans_model_filename, kmeans_model_filename)
            elif os.path.exists(temp_kmeans_model_filename):
                os.rename(temp_kmeans_model_filename, kmeans_model_filename)
            if os.path.exists(tfidf_model_filename):
                os.remove(tfidf_model_filename)
                os.rename(temp_tfidf_model_filename, tfidf_model_filename)
            elif os.path.exists(temp_tfidf_model_filename):
                os.rename(temp_tfidf_model_filename, tfidf_model_filename)
            sys.exit()

        try:
        	# when we create a new model, we discard the older ones from the database.
        	# At any point in time, only one and the most recent model can exist in a database.
            if db_collection.count() > 0:
                db_collection.drop()
            db_collection.insert_one(key_terms_dictionary)
            if os.path.exists(temp_tfidf_model_filename):
                os.remove(temp_tfidf_model_filename)
            if os.path.exists(temp_kmeans_model_filename):
                os.remove(temp_kmeans_model_filename)

            # each time we create a new model, we also purge all the
            # data in other databases as well.
            # This ensures that there is no stale data in any of the databases.
            cur_abstract_search.execute("DELETE FROM TopicIterations")
            conn_abstract_search.commit()
            cur_abstract_search.execute("DELETE FROM TopicKeywords")
            conn_abstract_search.commit()
            cur_abstract_search.execute("DELETE FROM TopicAwards")
            conn_abstract_search.commit()
            db_ranked_topic_ids_collection.drop()
        except Exception as e:
            print ('Error Encountered: ' + str(e))
            sys.exit()

if __name__ == '__main__':
    arg_len = len(sys.argv)
    if arg_len == 9:
        create_model_Flag = 0
        num_clusters = 75
    elif arg_len == 10:
        create_model_Flag = sys.argv[9]
        num_clusters = 75
    elif arg_len == 11:
        create_model_Flag = sys.argv[9]
        num_clusters = sys.argv[10]
    else:
        print "Invalid number of arguments"
        sys.exit(0)

    start_date = sys.argv[1]
    end_date = sys.srgv[2]
    num_topics = sys.argv[3]
    num_words = sys.argv[4]
    h_name = sys.argv[5]
    u_name = sys.argv[6]
    pword = sys.argv[7]
    db_name = sys.argv[8]

    if create_model_Flag == 0:
        return extract_topics(start_date, end_date, num_topics, num_words, 0, num_clusters, h_name, u_name, pword, db_name)
    else:
        # creating a model is a time-consuming process. Hence, we process it as an asynchronous call.
        # We create a new thread that starts the model creation process and the main thread does not wait for the
        # model creation task to complete. In this way, the model creation task runs as a background process.
        create_model_thread = threading.Thread(target=extract_topics, args=(start_date, end_date, 0, 0, 1, num_clusters, h_name, u_name, pword, db_name,))
        create_model_thread.start()
