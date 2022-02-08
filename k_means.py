# -*- coding: utf-8 -*-
"""
###
"""

import random as rd
import re
import math
import string
import pandas as pd
import numpy as np
from tabulate import tabulate

class K_Means_Class:

    def __init__(self, strURL, k = 5, tolerance = 0.001, maxIter = 10, data_table = []):
        self.k = k
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.strURL = strURL
        #self.data_table = data_table


    def pre_process_raw_data(self):
        #pd.read_csv('https://drive.google.com/uc?export=download&id=' +strUrl.split('/')[-2], encoding = 'unicode_escape', delimiter = strDelim)
        dataFile = open(self.strURL, "r", encoding="utf8")
        tweetsData = list(dataFile)
        tweets_array = []

        for i in range(len(tweetsData)):

            # remove \n from the end after every sentence
            tweetsData[i] = tweetsData[i].strip('\n')

            # Remove the tweet id and timestamp
            tweetsData[i] = tweetsData[i][50:]

            # Remove any word that starts with the symbol @
            tweetsData[i] = " ".join(filter(lambda x: x[0] != '@', tweetsData[i].split()))

            # Remove any URL
            tweetsData[i] = re.sub(r"http\S+", "", tweetsData[i])
            tweetsData[i] = re.sub(r"www\S+", "", tweetsData[i])

            # remove colons from the end of the sentences (if any) after removing url
            tweetsData[i] = tweetsData[i].strip()
            tweet_len = len(tweetsData[i])
            if tweet_len > 0:
                if tweetsData[i][len(tweetsData[i]) - 1] == ':':
                    tweetsData[i] = tweetsData[i][:len(tweetsData[i]) - 1]

            # Remove any hash-tags symbols
            tweetsData[i] = tweetsData[i].replace('#', '')
            tweetsData[i] = tweetsData[i].replace('-', '')


            # Convert every word to lowercase
            tweetsData[i] = tweetsData[i].lower()

            # remove punctuations
            tweetsData[i] = tweetsData[i].translate(str.maketrans('', '', string.punctuation))

            # trim extra spaces
            tweetsData[i] = " ".join(tweetsData[i].split())

            # convert each tweet from string type to as list<string> using " " as a delimiter
            tweets_array.append(tweetsData[i].split(' '))

        dataFile.close()

        return tweets_array

    def pre_process_raw_data_using_dataFrame(self, strUrl, strDelim):
        #Read the CSV file from Google Drive and create the data frame
        dataFrame = pd.read_csv('https://drive.google.com/uc?export=download&id=' +strUrl.split('/')[-2], encoding = 'unicode_escape', delimiter = strDelim)
        tweets_array = []

        # To drop any blank rows
        dataFrame.dropna(axis = 0, how = 'any', thresh = None, inplace = True)
        
        # remove \n from the end after every sentence
        dataFrame = dataFrame.replace('\n','', regex=True)

        # Remove the tweet id and timestamp
        dataFrame = dataFrame.iloc[: , 2:]
        
        # Remove any word that starts with the symbol @
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].str.replace('(\@\w+.*?)',"")

        #  Remove any hashtag symbols e.g. convert #depression to depression
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].str.replace('#',"")
        
        # Remove any URL
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].apply(lambda x: re.split('www:\/\/.*', str(x))[0])
        
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].map(lambda x: x.lower().strip())
        # remove punctuations
        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].str.replace('[^\w\s]','')

        dataFrame.iloc[:, 0] = dataFrame.iloc[:, 0].str.replace('-',"")
        
        dataFrame.columns =['tweets']
        return dataFrame.to_numpy()


    def main_kmeans(self, tweetsData):

        #tweetsData = self.pre_process_raw_data()
        prev_centroids = []
        curr_centroids = []
        # initialization, assign random tweets as centroids
        count = 0
        hash_map = dict()
        while count < self.k:
            random_tweet_idx = rd.randint(0, len(tweetsData) - 1)
            if random_tweet_idx not in hash_map:
                count += 1
                hash_map[random_tweet_idx] = True
                curr_centroids.append(tweetsData[random_tweet_idx])
        
        loop_count = 0
        
        # run the iterations until not converged or loop is not reached to the max iteration
        while (self.is_converged(prev_centroids, curr_centroids)) == False and (loop_count < self.maxIter):

            # assignment of each tweets to the closest centroids
            clusters = self.assign_cluster_to_nearest_centroid(tweetsData, curr_centroids)

            # we need to keep track of previous centroid for convergence check
            prev_centroids = curr_centroids

            # update centroid based on clusters formed
            #curr_centroids = update_centroids(clusters)
            loop_count = loop_count + 1

        s2_error = self.squared_error(clusters)
        #Please do not play with the below few lines of code
        count = 0
        strInfo = ''
        while count < self.k:
            if count == 0:
                strInfo = strInfo + str(count)+': '+str(len(clusters[count]))+' tweets \n'
            else:
                strInfo = strInfo + "                    " + str(count)+': '+str(len(clusters[count]))+' tweets \n'
            
            count = count + 1
        print("   "+str(self.k)+"        "+str(s2_error)+"    "+strInfo)
        return clusters, s2_error


    def is_converged(self, prev_centroid, new_centroids):
        # if lengths are not equal then funtion will return false
        if len(prev_centroid) != len(new_centroids):
            return False

        # iterate over each entry of clusters and check if they are same
        for i in range(len(new_centroids)):
            if " ".join(new_centroids[i]) != " ".join(prev_centroid[i]):
                return False

        return True


    # Jaccard Distance calculator
    def jaccard_distance_calculation(self, tweet1, tweet2):
        tweet1 = tweet1.split(' ')
        tweet12 = tweet2.split(' ')
        print(tweet1)
        print(tweet2)
        # Jaccard distance is: 1 - length(tweet1 intersection tweet2)/length(tweet1 union tweets2)
        # get the intersection
        #intersectionLength = len(set(tweet1).intersection(tweet2))
        intersectionLength = len(set(tweet1) & set(tweet2))

        # get the union
        #unionLength = len(set().union(tweet1, tweet2))
        unionLength = len(set(tweet1) | set(tweet2))

        # return the jaccard distance
        return round(1 - (float(intersectionLength) / unionLength), 5)
		

    def assign_cluster_to_nearest_centroid(self, tweetsData, centroids):
        clusters = dict()

        # for every tweet iterate each centroid and assign closest centroid to a it
        for i in range(len(tweetsData)):
            min_dis = math.inf
            cluster_idx = -1;
            for j in range(len(centroids)):
                dis = self.jaccard_distance_calculation(centroids[j], tweetsData[i])
                
                if centroids[j] == tweetsData[i]:
                    cluster_idx = j
                    min_dis = 0
                    break

                if dis < min_dis:
                    cluster_idx = j
                    min_dis = dis

            # if minimum distance = 1 i.e. the there is no word common between centroid and the current tweet, assign any random cluster
            if min_dis == 1:
                cluster_idx = rd.randint(0, len(centroids) - 1)

            # centroid allocation to a tweet
            clusters.setdefault(cluster_idx, []).append([tweetsData[i]])

            # Distance of each tweet form closest centroid, this min_dis will be used for squared error calculation
            last_tweet_idx = len(clusters.setdefault(cluster_idx, [])) - 1
            clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(min_dis)
        #print(str(clusters.setdefault(cluster_idx, []))+': '+str(len(clusters.setdefault(cluster_idx, [])))+' tweets')
        return clusters

    def squared_error(self, clusters):
        squared_error = 0
        # For every cluster squared_error is calculated as the sum of square of distances of the tweet from the associated centroid
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                squared_error = squared_error + int(pow(clusters[i][j][1], 2))
            
        return squared_error
