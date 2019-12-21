#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:31:15 2019

@author: subeydengur
"""
from bs4 import BeautifulSoup
import requests
import os
import pandas as pd

os.chdir('To folder with all (and ONLY) movie scripts.txt')
movies=os.listdir()

movies=[i.replace('.txt','') for i in movies ]


box_office=[]
for movie in movies:
    try:
        page_link = 'https://www.boxofficemojo.com/search/?q='+movie
        page_response = requests.get(page_link)
        page_content = BeautifulSoup(page_response.content, "lxml")

        link = page_content.findAll('a',{'class':"a-size-medium a-link-normal a-text-bold"})[0].get('href')
        movie_link = 'https://www.boxofficemojo.com' + link
        movie_response = requests.get(movie_link)
        movie_content = BeautifulSoup(movie_response.content, "lxml")
#         print(i)
        box_office.append(movie_content.findAll('span',{'class':'money'})[2].get_text())
#         i=i+1
    except:
#         print("a problem with entry", i)
        box_office.append(-100)
        
   
     
b_off=[]
for box in box_office:
    try:
#     b_off.append(box.get_text())
        b_off.append(box.replace('$','').replace(',',''))
    except:
        b_off.append(-100)

df = pd.DataFrame({'Title':movies, 'Box Office':b_off})


#This is a rerun of for the missing titles.
#e.g. it cant find avengersthe2012 but with this we search for the avengers which works

missing = list(df['Title'][df['Box Office']==-100])
search_missing = [i.replace('the','***') for i in missing]
search_missing = [' the ' + i if '***' in i else i for i in search_missing]

search_missing = [i[:14] for i in search_missing]


#now re run the code that finds box_office
missing_bo=[]
for movie in search_missing:
    try:
        page_link = 'https://www.boxofficemojo.com/search/?q='+movie
        page_response = requests.get(page_link)
        page_content = BeautifulSoup(page_response.content, "lxml")

        link = page_content.findAll('a',{'class':"a-size-medium a-link-normal a-text-bold"})[0].get('href')
        movie_link = 'https://www.boxofficemojo.com' + link
        movie_response = requests.get(movie_link)
        movie_content = BeautifulSoup(movie_response.content, "lxml")
#         print(i)
        missing_bo.append(movie_content.findAll('span',{'class':'money'})[2].get_text())
#         i=i+1
    except:
#         print("a problem with entry", i)
        missing_bo.append(-100)

missing_df = pd.DataFrame({'Title':missing,'Fixed Title':search_missing, 'Box Office':missing_bo})

missing_df=missing_df.drop(columns=['Fixed Title'])

missing_df=missing_df.set_index('Title')
df=df.set_index('Title')
df.update(missing_df)

def clean_currency(x):
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)
    
#gets rid of all commas and $ for Box-Office column
    
df['Box Office'] = df['Box Office'].apply(clean_currency).astype('int')


#this is to allocate tiers for each box office result 
#The ranges are found so that each tier ends up with equal/similar sample size
def tier_finder(val):
    if (val<4700000):
        tier = 1
    elif ((val>4700000) & (val<55000000)):
        tier = 2
    elif ((val>55000000) & (val<195000000)):
        tier = 3
    elif ((val>195000000)):
        tier = 4
    
    
    return tier


df['Tier'] = df['Box Office'].apply(tier_finder)






#We had issues with these titles which is why they are omitted
#wrote them one by one so that in case we fix it, we can just comment out

df = df.drop("starshiptroopers", axis=0)
df = df.drop("apocalypsenow", axis=0)
df = df.drop("youvegotmail", axis=0)
df = df.drop("stingthe", axis=0)
df = df.drop("peggysuegotmarried", axis=0)





#df.to_pickle('title_Bo.pickle')
#df.to_excel("title_Bo.xlsx") 

#This part is to create a txt file with all titles and all tiers 

file1 = open("all_titles.txt","w") 
all_titles=df.index.tolist()
for i in all_titles:
    file1.write(str(i)+',')
file1.close()

file1 = open("all_tiers.txt","w") 
tierlist=df['Tier'].tolist()
for i in tierlist:
    file1.write(str(i)+',')
file1.close()





    


    
