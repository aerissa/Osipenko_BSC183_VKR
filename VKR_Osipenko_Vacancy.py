#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fake_useragent')
get_ipython().system('pip install requests')
get_ipython().system('pip install bs4')
get_ipython().system('pip install pyopenssl ndg-httpsclient pyasn1')


# In[ ]:


import requests
import fake_useragent
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from os import name
from collections import Counter

#Для обработки естественного языка
import re
import string
import nltk
from pymystem3 import Mystem
from nltk.corpus import stopwords
from string import punctuation
from nltk import word_tokenize
from nltk.probability import FreqDist


# In[ ]:


number_of_pages = 100

job_title = ["Журналист"]
for job in job_title:
	data=[]
	for i in range(number_of_pages):
	    url = 'https://api.hh.ru/vacancies'
	    par = {'text': job, 'area':'113','per_page':'10', 'page':i}
	    r = requests.get(url, params=par)
	    e=r.json()
	    data.append(e)
	    vacancy_details = data[0]['items'][0].keys()
	    df = pd.DataFrame(columns= list(vacancy_details))
	    ind = 0
	    for i in range(len(data)):
	    	for j in range(len(data[i]['items'])):
	    		df.loc[ind] = data[i]['items'][j]
	    		ind+=1
	df.to_csv("Journalist.csv")


# In[ ]:


df['og_link'] = df['alternate_url'] + '?from=vacancy_search_list&hhtmFrom=vacancy_search_list'


# In[ ]:


vac_url = df['og_link'].tolist()


# In[ ]:


def get_vacansy(link):
    ua = fake_useragent.UserAgent()
    data = requests.get(
        url=link,
        headers={"user-agent":ua.random}
    )
    if data.status_code != 200:
        return
    soup = BeautifulSoup(data.content, "lxml")
    try:
        name = soup.find(attrs={"class":"vacancy-title"}).find("h1").text
    except:
        name = ""
    try:
        tags = [tag.text for tag in soup.find(attrs={"class":"bloko-tag-list"}).find_all("span",attrs={"class":"bloko-tag__section_text"})]
    except:
        tags = []
    resume = {
        "name":name,
        "tags":tags,
    }
    return resume


# In[ ]:


if __name__ == "__main__":
    dat1 = []
    for a in vac_url:
        dat1.append(get_vacansy(a))
        time.sleep(1)
        with open("data.json","w",encoding="utf-8")as f:
            json.dump(dat1,f,indent = 4, ensure_ascii=False)


# In[ ]:


df_vac = pd.json_normalize(dat1)


# In[ ]:


df_vac.to_csv('vacancy_data_VKR.csv')


# In[ ]:


vac_vac = df_vac.copy()


# In[ ]:


vac_vac.tags = vac_vac.tags.apply(lambda x: np.nan if len(x)==0 else x)


# In[ ]:


vac_vac.dropna()


# In[ ]:


name_list = df_vac['name'].tolist()


# In[ ]:


texx = ' '.join(name_list)
texx = texx.replace('-', ' ')
texx = texx.replace('/', ' ')
texx


# In[ ]:


mystem = Mystem() 
russian_stopwords = stopwords.words("russian") + ['это', 'язык', 'умение', 'специалист', 'менеджер','отдел', 'навык', 'хотеть', 'делать', 'начинать', 'хотеться', 'мочь', 'который', 'весь', 'свой']
def preprocess_text(text):
    text = re.sub(r"\n", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords              and token != " "               and token.strip() not in punctuation              and token not in russian_stopwords]
    
    text = " ".join(tokens)
    
    return text


# In[ ]:


texx = preprocess_text(texx)


# In[ ]:


texx_tokens = word_tokenize(texx)


# In[ ]:


text1 = nltk.Text(texx_tokens)


# In[ ]:


fdist = FreqDist(text1)


# In[ ]:


fdist.most_common(20)


# In[ ]:


vac_names = pd.DataFrame.from_dict(fdist, orient='index').reset_index()
vac_names = vac_names.rename(columns={'index':'name', 0:'count'})


# In[ ]:


vac_names['%'] = vac_names['count'].apply(lambda x: x / 577 * 100)
vac_names = vac_names.sort_values('count', ascending=False)
vac_names.index = range(1, len(vac_names)+1)


# In[ ]:


vac_names[vac_names['%']>3]


# In[ ]:


vac_names.to_excel('vac_names.xlsx')


# In[ ]:


list_tags = df_vac['tags'].tolist()


# In[ ]:


flat_list1 = [item for sublist in list_tags for item in sublist]


# In[ ]:


counts2 = Counter(flat_list1)
print(counts2)


# In[ ]:


vac_tags = pd.DataFrame.from_dict(counts2, orient='index').reset_index()
vac_tags = vac_tags.rename(columns={'index':'tag', 0:'count'})


# In[ ]:


vac_tags['%'] = vac_tags['count'].apply(lambda x: x / 577 * 100)
vac_tags = vac_tags.sort_values('count', ascending=False)
vac_tags.index = range(1, len(vac_tags)+1)


# In[ ]:


vac_tags['% full'] = vac_tags['count'].apply(lambda x: x / 498 * 100)


# In[ ]:


new_vac = vac_tags[vac_tags['% full']>3]


# In[ ]:


new_vac


# In[ ]:


new_vac.to_excel('new_vac.xlsx')


# In[ ]:


exp_vac_tags = vac_tags[vac_tags['% full']>3.5]


# In[ ]:


exp_vac_tags['expertise'] = ""


# In[ ]:


exp_vac_tags.loc[1,'expertise'] = 'TE'
exp_vac_tags.loc[2,'expertise'] = 'TE'
exp_vac_tags.loc[3,'expertise'] = 'TE'
exp_vac_tags.loc[4,'expertise'] = 'IntE'
exp_vac_tags.loc[5,'expertise'] = 'TE'
exp_vac_tags.loc[6,'expertise'] = 'IntE'
exp_vac_tags.loc[7,'expertise'] = 'AE'
exp_vac_tags.loc[8,'expertise'] = 'IntE'
exp_vac_tags.loc[9,'expertise'] = 'InnE'
exp_vac_tags.loc[10,'expertise'] = 'TE'
exp_vac_tags.loc[11,'expertise'] = 'IntE'
exp_vac_tags.loc[12,'expertise'] = 'InnE'
exp_vac_tags.loc[13,'expertise'] = 'AE'
exp_vac_tags.loc[14,'expertise'] = 'TE'
exp_vac_tags.loc[15,'expertise'] = 'TE'
exp_vac_tags.loc[16,'expertise'] = 'IntE'
exp_vac_tags.loc[17,'expertise'] = 'InnE'
exp_vac_tags.loc[18,'expertise'] = 'AE'
exp_vac_tags.loc[19,'expertise'] = 'AE'
exp_vac_tags.loc[20,'expertise'] = 'AE'
exp_vac_tags.loc[21,'expertise'] = 'AE'
exp_vac_tags.loc[22,'expertise'] = 'TE'
exp_vac_tags.loc[23,'expertise'] = 'TE'
exp_vac_tags.loc[24,'expertise'] = 'AE'
exp_vac_tags.loc[25,'expertise'] = 'AE'
exp_vac_tags.loc[26,'expertise'] = 'IntE'
exp_vac_tags.loc[27,'expertise'] = 'InnE'
exp_vac_tags.loc[28,'expertise'] = 'InnE'
exp_vac_tags.loc[29,'expertise'] = 'InnE'
exp_vac_tags.loc[30,'expertise'] = 'InnE'
exp_vac_tags.loc[31,'expertise'] = 'IntE'
exp_vac_tags.loc[32,'expertise'] = 'TE'
exp_vac_tags.loc[33,'expertise'] = 'TE'
exp_vac_tags.loc[34,'expertise'] = 'IntE'
exp_vac_tags.loc[35,'expertise'] = 'AE'
exp_vac_tags.loc[36,'expertise'] = 'IntE'
exp_vac_tags.loc[37,'expertise'] = 'InnE'
exp_vac_tags.loc[38,'expertise'] = 'IntE'
exp_vac_tags.loc[39,'expertise'] = 'InnE'
exp_vac_tags.loc[40,'expertise'] = 'TE'
exp_vac_tags.loc[41,'expertise'] = 'IntE'
exp_vac_tags.loc[42,'expertise'] = 'IntE'


# In[ ]:


exp_vac_stat = exp_vac_tags.groupby(['expertise']).sum()
exp_vac_stat.to_excel('exp_vac_stat.xlsx')


# In[ ]:


exp_vac_stat['exp'] = exp_vac_stat.index


# In[ ]:


exp_vac_stat['count %'] = exp_vac_stat['count'].apply(lambda x: x / 2211 * 100)


# In[ ]:


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


# In[ ]:


fig = plt.figure(figsize =(10, 7))
plt.pie(exp_vac_stat['count'], labels = exp_vac_stat['exp'], autopct=make_autopct(exp_vac_stat['count']), colors = ['lightskyblue', 'steelblue', 'aliceblue', 'lightslategray'])
plt.savefig('exp_vac_stat.png')


# In[ ]:


exp_vac_stat['count'].sum()


# In[ ]:


exp_vac_tags.groupby(['expertise']).count()


# In[ ]:


exp_vac_tags = exp_vac_tags.sort_values(['expertise', 'count'], ascending=[False, False])
exp_vac_tags.index = range(1, len(exp_vac_tags)+1)


# In[ ]:


exp_vac_tags.to_excel('exp_vac_tags.xlsx')


# In[ ]:


vals = exp_vac_tags['expertise'].tolist()
colors = ["pink" if i == 'TE' else "lavenderblush" if i == 'InnE' else 'palevioletred' if i == 'IntE' else 'crimson' for i in vals]


# In[ ]:


plt.figure(figsize=(15,15))
plt.bar(exp_vac_tags['tag'], exp_vac_tags['% full'], color = colors)
plt.ylabel(f'% в заполненных резюме')
plt.xlabel('Навык')
plt.xticks(rotation=90)
plt.savefig('exp_vac_tags.png')

