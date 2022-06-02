#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fake_useragent')
get_ipython().system('pip install pyopenssl ndg-httpsclient pyasn1')


# In[ ]:


import re
import requests
import fake_useragent
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from os import name
from bs4 import BeautifulSoup
from randan.descriptive_statistics import ScaleStatistics


# In[ ]:


def get_links(text):
    ua =fake_useragent.UserAgent()
    res = requests.get(
        url=f"https://hh.ru/search/resume?area=113&relocation=living_or_relocation&gender=unknown&text={text}&isDefaultArea=true&exp_period=all_time&logic=normal&pos=full_text&fromSearchLine=false&search_period=0",
        headers={"user-agent":ua.random}
    )
    if res.status_code != 200:
        return
    soup = BeautifulSoup(res.content, "lxml")
    try:
        page_count = int(soup.find("div",attrs={"class":"pager"}).find_all("span",recursive=False)[-1].find("a").find("span").text)
    except:
        return
    for page in range(page_count):
        try:
            res = requests.get(
                url=f"https://hh.ru/search/resume?area=113&relocation=living_or_relocation&gender=unknown&text={text}&isDefaultArea=true&exp_period=all_time&logic=normal&pos=full_text&fromSearchLine=false&search_period=0&page={page}",
                headers={"user-agent":ua.random}
            )
            if res.status_code == 200:
                soup = BeautifulSoup(res.content, "lxml")
                for a in soup.find_all("a",attrs={"class":"resume-search-item__name"}):
                    yield f'https://hh.ru{a.attrs["href"].split("?")[0]}'
        except Exception as e:
            print(f"{e}")
        time.sleep(1)
    print(page_count)


# In[ ]:


def get_resume(link):
    ua = fake_useragent.UserAgent()
    data = requests.get(
        url=link,
        headers={"user-agent":ua.random}
    )
    if data.status_code != 200:
        return
    soup = BeautifulSoup(data.content, "lxml")
    try:
        name = soup.find(attrs={"class":"resume-block__title-text"}).text
    except:
        name = ""
    try:
        spec = soup.find_all("div", attrs={"class":"bloko-gap bloko-gap_bottom"})[1].text
    except:
        spec = []
    try:
        salary = soup.find(attrs={"class":"resume-block__title-text_salary"}).text.replace("\u2009","").replace("\xa0"," ")
    except:
        salary = ""
    try:
        exp = soup.find(attrs={"class":"resume-block__title-text resume-block__title-text_sub"}).text.replace("&nbsp;","").replace("\xa0"," ")
    except:
        exp = ""
    try:
        tags = [tag.text for tag in soup.find(attrs={"class":"bloko-tag-list"}).find_all("span",attrs={"class":"bloko-tag__section_text"})]
    except:
        tags = []
    resume = {
        "name":name,
        "salary":salary,
        "specialisation":spec,
        "experience":exp,
        "tags":tags,
    }
    return resume


# In[ ]:


if __name__ == "__main__":
    data1 = []
    for a in get_links("журналист"):
        data1.append(get_resume(a))
        time.sleep(1)
        with open("data.json","w",encoding="utf-8")as f:
            json.dump(data1,f,indent = 4, ensure_ascii=False)


# In[ ]:


df_resume1 = pd.json_normalize(data1)


# In[ ]:


df_resume1.to_csv('resume_data_VKR.csv')


# In[ ]:


res1 = df_resume1.copy()


# In[ ]:


res1['tags'].dropna()


# In[ ]:


matchPattern = r"([^A-Z]+)([A-Z]+)"
replacePattern = r"\1 \2"
res['specialisation'] = res['specialisation'].apply(lambda x: ''.join(x[j]if j not in[i for i in range(len(x)) if x[i]!=x.lower()[i]] else ' '+x[j]for j in range(len(x))).lstrip())
res1['specialisation'] = res['specialisation'].apply(lambda x: x[14:])


# In[ ]:


res1['specialisation'] = res1['specialisation'].apply(lambda x: x[14:])


# In[ ]:


res1['spec_count'] = res1['specialisation'].apply(lambda x: sum(1 for c in x if c.isupper()))


# In[ ]:


ss = ScaleStatistics(res1, ['spec_count'],
                    normality_test=True)


# In[ ]:


tt = res1.spec_count.value_counts()


# In[ ]:


res1['spec_count'].value_counts()/res1['spec_count'].count()*100


# In[ ]:


svar ='spec_count'

plt.figure(figsize=(8,6))
plt.hist(res1[svar].dropna(), color='pink')
plt.xticks(rotation=90)
plt.xlabel(f'Количество специализаций')
plt.ylabel('Количество резюме')
plt.savefig('spec_count.png')


# In[ ]:


res1['spec1'] = res1['specialisation'].apply(lambda x: x.replace(" ", ""))


# In[ ]:


res1['spec'] = res1['spec1'].apply(lambda x: re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z])|[а-я](?=[А-Я]))|[a-z](?=[А-Я])', r'\1 ', x).split())


# In[ ]:


listed11 = res11['tags'].tolist()


# In[ ]:


flat_list1 = [item for sublist in listed11 for item in sublist]


# In[ ]:


listed = res1['spec'].tolist()


# In[ ]:


flat_list = [item for sublist in listed for item in sublist]


# In[ ]:


counts = Counter(flat_list)
print(counts)


# In[ ]:


res_specs = pd.DataFrame.from_dict(counts, orient='index').reset_index()
res_specs = res_specs.rename(columns={'index':'spec', 0:'count'})


# In[ ]:


res_specs['%'] = res_specs['count'].apply(lambda x: x / 2117 * 100)
res_specs = res_specs.sort_values('count', ascending=False)
res_specs.index = range(1, len(res_specs)+1)


# In[ ]:


res_specs.to_excel('res_specs.xlsx')


# In[ ]:


counts1 = Counter(flat_list1)
print(counts1)


# In[ ]:


res_tags = pd.DataFrame.from_dict(counts1, orient='index').reset_index()
res_tags = res_tags.rename(columns={'index':'tag', 0:'count'})


# In[ ]:


res_tags['%'] = res_tags['count'].apply(lambda x: x / 2117 * 100)
res_tags = res_tags.sort_values('count', ascending=False)
res_tags.index = range(1, len(res_tags)+1)


# In[ ]:


res_tags['% full'] = res_tags['count'].apply(lambda x: x / 1326 * 100)


# In[ ]:


hist_res_tags = res_tags[res_tags['% full']>3.5]


# In[ ]:


hist_res_tags['expertise'] = ""


# In[ ]:


hist_res_tags


# In[ ]:


hist_res_tags.loc[1,'expertise'] = 'TE'
hist_res_tags.loc[2,'expertise'] = 'TE'
hist_res_tags.loc[3,'expertise'] = 'TE'
hist_res_tags.loc[4,'expertise'] = 'InnE'
hist_res_tags.loc[5,'expertise'] = 'TE'
hist_res_tags.loc[6,'expertise'] = 'IntE'
hist_res_tags.loc[7,'expertise'] = 'TE'
hist_res_tags.loc[8,'expertise'] = 'AE'
hist_res_tags.loc[9,'expertise'] = 'InnE'
hist_res_tags.loc[10,'expertise'] = 'TE'
hist_res_tags.loc[11,'expertise'] = 'AE'
hist_res_tags.loc[12,'expertise'] = 'IntE'
hist_res_tags.loc[13,'expertise'] = 'AE'
hist_res_tags.loc[14,'expertise'] = 'IntE'
hist_res_tags.loc[15,'expertise'] = 'InnE'
hist_res_tags.loc[16,'expertise'] = 'TE'
hist_res_tags.loc[17,'expertise'] = 'TE'
hist_res_tags.loc[18,'expertise'] = 'IntE'
hist_res_tags.loc[19,'expertise'] = 'IntE'
hist_res_tags.loc[20,'expertise'] = 'InnE'
hist_res_tags.loc[21,'expertise'] = 'IntE'
hist_res_tags.loc[22,'expertise'] = 'AE'
hist_res_tags.loc[23,'expertise'] = 'TE'
hist_res_tags.loc[24,'expertise'] = 'InnE'
hist_res_tags.loc[25,'expertise'] = 'TE'
hist_res_tags.loc[26,'expertise'] = 'TE'
hist_res_tags.loc[27,'expertise'] = 'IntE'
hist_res_tags.loc[28,'expertise'] = 'AE'
hist_res_tags.loc[29,'expertise'] = 'IntE'
hist_res_tags.loc[30,'expertise'] = 'IntE'
hist_res_tags.loc[31,'expertise'] = 'IntE'
hist_res_tags.loc[32,'expertise'] = 'TE'
hist_res_tags.loc[33,'expertise'] = 'InnE'
hist_res_tags.loc[34,'expertise'] = 'TE'
hist_res_tags.loc[35,'expertise'] = 'TE'
hist_res_tags.loc[36,'expertise'] = 'IntE'
hist_res_tags.loc[37,'expertise'] = 'TE'
hist_res_tags.loc[38,'expertise'] = 'TE'
hist_res_tags.loc[39,'expertise'] = 'InnE'
hist_res_tags.loc[40,'expertise'] = 'IntE'


# In[ ]:


hist_res_tags.to_excel('hist_res_tags.xlsx')


# In[ ]:


exp_res_tags = hist_res_tags.copy()


# In[ ]:


exp_res_tags = exp_res_tags.sort_values(['expertise', 'count'], ascending=[False, False])
exp_res_tags.index = range(1, len(exp_res_tags)+1)


# In[ ]:


exp_res_tags.to_excel('exp_res_tags.xlsx')


# In[ ]:


vals = exp_res_tags['expertise'].tolist()
colors = ["pink" if i == 'TE' else "lavenderblush" if i == 'InnE' else 'palevioletred' if i == 'IntE' else 'crimson' for i in vals]


# In[ ]:


plt.figure(figsize=(15,15))
plt.bar(exp_res_tags['tag'], exp_res_tags['% full'], color = colors)
plt.ylabel(f'% в заполненных резюме')
plt.xlabel('Навык')
plt.xticks(rotation=90) 
plt.savefig('exp_res_tags.png')


# In[ ]:


exp_res_stat = exp_res_tags.groupby(['expertise']).sum()
exp_res_stat


# In[ ]:


exp_res_stat.sum()


# In[ ]:


exp_res_stat.to_excel('exp_res_stat.xlsx')


# In[ ]:


exp_res_stat['exp'] = exp_res_stat.index


# In[ ]:


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


# In[ ]:


fig = plt.figure(figsize =(10, 7))
plt.pie(exp_res_stat['count'], labels = exp_res_stat['exp'], autopct=make_autopct(exp_res_stat['count']), colors = ['olive', 'lightgoldenrodyellow', 'yellowgreen', 'darkolivegreen'])
plt.savefig('exp_res_stat.png')

