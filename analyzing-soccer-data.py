#!/usr/bin/env python
# coding: utf-8

# 

# 

# 

# ## <a id='import'></a> Import and load data

# In[1]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[2]:


import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[3]:


con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
table_names = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# Read all sql tables into data frames to be analyzed. 

# In[4]:


player_table = pd.read_sql_query("SELECT * FROM Player", con)
player_att_table = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
match_table = pd.read_sql_query("SELECT * FROM Match", con)
league_table = pd.read_sql_query("SELECT * FROM League", con)
country_table = pd.read_sql_query("SELECT * FROM Country", con)
team_table = pd.read_sql_query("SELECT * FROM Team", con)
team_att_table = pd.read_sql_query("SELECT * FROM Team_Attributes", con)


# ## <a id='data-analysis-and-viz'></a> Data Analysis and Visualization 

# ### <a id='country'></a> Analyzing Country Table

# In[5]:


print("Dimension of Country Table is: {}".format(country_table.shape))
print(100*"*")
print(country_table.info())
print(100*"*")
print(country_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(country_table.describe())
print(100*"*")
print(country_table.isnull().sum(axis=0))


# In[6]:


country_table


# ### <a id='league'></a> Analyzing League Table

# In[7]:


print("Dimension of League Table is: {}".format(league_table.shape))
print(100*"*")
print(league_table.info())
print(100*"*")
print(league_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(league_table.describe())
print(100*"*")
print(league_table.isnull().sum(axis=0))


# In[8]:


league_table


# Data is available only for the european leagues. Note that top 5 leaguesa are: Ligue 1, Bundesliga, Serie A, Premier League and LIGA BBVA

# ### <a id='analyze-player-table'></a>Analyzing Player Table

# In[9]:


print("Dimension of Player Table is: {}".format(player_table.shape))
print(100*"*")
print(player_table.info())
print(100*"*")
print(player_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(player_table.describe())
print(100*"*")
print(player_table.isnull().sum(axis=0))
#Player table has no missing data


# In[10]:


fig1, ax1 = plt.subplots(nrows = 1, ncols = 2)
fig1.set_size_inches(14,4)
sns.boxplot(data = player_table.loc[:,["height",'weight']], ax = ax1[0])
ax1[0].set_xlabel('Player Table Features')
ax1[0].set_ylabel('')
sns.distplot(a = player_table.loc[:,["height"]], bins= 10, kde = True, ax = ax1[1],             label = 'Height')
sns.distplot(a = player_table.loc[:,["weight"]], bins= 10, kde = True, ax = ax1[1],             label = 'Weight')
ax1[1].legend()
sns.jointplot(x='height',y = 'weight',data = player_table,kind = 'scatter')
fig1.tight_layout()


# In[11]:


print("Cardinality of Feature: Height - {:0.3f}%".format(         100 * (len(np.unique(player_table.loc[:,'height'])) / len(player_table.loc[:,'height']))))
print("Cardinality of Feature: Weight - {:0.3f}%".format(         100 * (len(np.unique(player_table.loc[:,'weight'])) / len(player_table.loc[:,'weight']))))


# Very low cardinality for continuous variable for both the weight and height features considering we have 11060 instances. As expected instances from both the weight and height features follow a normal distribution and follow a linear relationship. 

# ### <a id='analyze-player-att-table'></a> Analyzing Player Attributes Table

# In[12]:


print("Dimension of Player Attributes Table is: {}".format(player_att_table.shape))
print(100*"*")
print(player_att_table.info())
print(100*"*")
print(player_att_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(player_att_table.describe())
print(100*"*")
print(player_att_table.isnull().sum(axis=0))
#Player Attributes Table has some missing data


# In[13]:


np.unique(player_att_table.dtypes.values)


# In[14]:


player_att_table.select_dtypes(include =['float64','int64']).head().loc[:,player_att_table.select_dtypes(include =['float64','int64']).columns[3:]].head()


# Analyze the correlation between the continuous features. We should see a positive correlation between the attacking features, a positive correlation between the defensive features and a negative correlation between the attacking and defensive features.  

# In[15]:


corr2 = player_att_table.select_dtypes(include =['float64','int64']).loc[:,player_att_table.select_dtypes(include =['float64','int64']).columns[3:]].corr()


# In[16]:


fig2,ax2 = plt.subplots(nrows = 1,ncols = 1)
fig2.set_size_inches(w=24,h=24)
sns.heatmap(corr2,annot = True,linewidths=0.5,ax = ax2)


# In[17]:


fig3, ax3 = plt.subplots(nrows = 1, ncols = 3)
fig3.set_size_inches(12,4)
sns.countplot(x = player_att_table['preferred_foot'],ax = ax3[0])
sns.countplot(x = player_att_table['attacking_work_rate'],ax = ax3[1])
sns.countplot(x = player_att_table['defensive_work_rate'],ax = ax3[2])
fig3.tight_layout()


# Figure out which columns have strange attacking and defensive work rate. Is there a correlation between attacking and defensive work rate values? If one is strange, is the other strange? 

# In[18]:


print(player_att_table['attacking_work_rate'].value_counts())
print(100*'*')
print(player_att_table['defensive_work_rate'].value_counts())
print(100*'*')
print(player_att_table.shape)


# The levels for both the attacking work rate and defensive work rate cateorical features should be 'low', 'medium', and 'high'. Note that they also account for the majority of instances. The remaining of the levels do not make sense so remove the instances that contain them. 

# In[19]:


player_att_table.loc[~(player_att_table['attacking_work_rate'].                                                  isin(['medium','high','low'])                       | player_att_table['defensive_work_rate'].isin(['medium','high','low'])),:].head()


# In[20]:


player_att_table_updated1 = player_att_table.loc[(player_att_table['attacking_work_rate'].                                                  isin(['medium','high','low'])                       & player_att_table['defensive_work_rate'].isin(['medium','high','low'])),:]
print(player_att_table_updated1.shape)
player_att_table_updated1.head()


# In[21]:


fig4, ax4 = plt.subplots(nrows = 1, ncols = 3)
fig4.set_size_inches(12,3)
sns.countplot(x = player_att_table_updated1['preferred_foot'],ax = ax4[0])
sns.countplot(x = player_att_table_updated1['attacking_work_rate'],ax = ax4[1])
sns.countplot(x = player_att_table_updated1['defensive_work_rate'],ax = ax4[2])
fig4.tight_layout()


# In[22]:


fig4, ax4 = plt.subplots(nrows = 1, ncols = 3)
fig4.set_size_inches(12,3)
sns.barplot(x ='preferred_foot', y = 'preferred_foot', data = player_att_table_updated1,            estimator = lambda x: len(x)/len(player_att_table_updated1) * 100, ax = ax4[0],           orient = 'v')
ax4[0].set(ylabel = 'Percentage',title = 'Preferred Foot')
sns.barplot(x ='attacking_work_rate', y = 'attacking_work_rate', data = player_att_table_updated1,            estimator = lambda x: len(x)/len(player_att_table_updated1) * 100, ax = ax4[1],           orient = 'v')
ax4[1].set(ylabel = 'Percentage',title = 'Attacking Work Rate')
sns.barplot(x ='defensive_work_rate', y = 'defensive_work_rate', data = player_att_table_updated1,            estimator = lambda x: len(x)/len(player_att_table_updated1) * 100, ax = ax4[2],           orient = 'v')
ax4[2].set(ylabel = 'Percentage',title = 'Defensive Work Rate')
fig4.tight_layout()


# In[23]:


att_work_rate = player_att_table_updated1.groupby('attacking_work_rate').size().values.tolist()
def_work_rate = player_att_table_updated1.groupby('defensive_work_rate').size().values.tolist()


# In[24]:


print("Attacking work rate factor, Medium, accounts for: {:0.3f}% of features".format(100 * att_work_rate[2]/np.sum(att_work_rate)))
print("Defensive work rate factor, Medium, accounts for: {:0.3f}% of features".format(100 * def_work_rate[2]/np.sum(def_work_rate)))


# Percentage of rows eliminated due to invalid/strange attacking and defensive work rate values:

# In[25]:


print("Percentage of instances removed from player attributes table: {:0.2f}%".      format(100* (1 - player_att_table_updated1.shape[0]/player_att_table.shape[0])))
print("We removed {} instances from Player Attributes table".      format(-player_att_table_updated1.shape[0] + player_att_table.shape[0]))


# In[26]:


print("Dimension of Player Attributes Table Updated 1 is: {}".format(player_att_table_updated1.shape))
print(100*"*")
print(player_att_table_updated1.info())
print(100*"*")
print(player_att_table_updated1.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(player_att_table_updated1.describe())
print(100*"*")
print(player_att_table_updated1.isnull().sum(axis=0))
#No more missing data


# Use all features in Player Attributes Updated table? Use Principal Component Analysis to reduce number of features in this table? Use only overall rating since this number is an accumulation of all other features for each player? Do not use any features? Will look into it in second kernel for prediction. 

# Analyze distribution and spead of continuous features based off of categorical features. Do the levels of each categorical features drastically change the distribution / spead of the continuous features?

# In[27]:


pat = player_att_table_updated1.loc[:,player_att_table_updated1.columns.tolist()[3:]]


# In[28]:


fig5, ax5 = plt.subplots(nrows=5,ncols=7)
fig5.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.distplot(pat.loc[:,j],kde = False,hist = True, ax = ax5[int(i/7)][i%7])
fig5.tight_layout()


# In[29]:


fig6, ax6 = plt.subplots(nrows=5,ncols=7)
fig6.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "preferred_foot", y = j, data= pat, ax = ax6[int(i/7)][i%7])
fig6.tight_layout()


# Preferred Foot does not distinguish any of the variables. Distribution of features the same regardless of preferred foot

# In[30]:


fig7, ax7 = plt.subplots(nrows=5,ncols=7)
fig7.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "attacking_work_rate", y = j, data= pat, ax = ax7[int(i/7)][i%7])
fig7.tight_layout() 


# Attacking work rate does a better job (better than preferred foot) of separating the features but not in a significant manner. Note that it does do a decent job of separating instances with high and low attacking work rate. Also remember that for attacking work rate feature, the factor, Medium', accounts for 70% of the instances. From closer examination, it appears that for features related to atacking attributes, attacking work rate feature does a good job of distinguishing instances of high and low categorical values. 

# In[31]:


fig8, ax8 = plt.subplots(nrows=5,ncols=7)
fig8.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "defensive_work_rate", y = j, data= pat, ax = ax8[int(i/7)][i%7])
fig8.tight_layout()


# Similar to attacking work rate, defensive work rate does a better job (better than preferred foot) of separating the features but not in a significant manner. Note that it does do a decent job of separating instances with high and low defensive work rate values for certain features. Also remember that for attacking work rate feature, the factor, Medium', accounts for 70% of the instances. From closer examination, it appears that for features related to defensive attributes, defensive work rate feature does a good job of distinguishing instances of high and low categorical values. 

# >### <a id='analyze-team-table'></a> Analyzing Team Table 

# In[32]:


print("Dimension of Team Table is: {}".format(team_table.shape))
print(100*"*")
print(team_table.info())
print(100*"*")
print(team_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_table.describe())
print(100*"*")
print(team_table.isnull().sum(axis=0))


# In[33]:


team_table[team_table.loc[:,'team_fifa_api_id'].isnull()]


# In[34]:


team_table_updated = team_table[~team_table.loc[:,'team_fifa_api_id'].isnull()]


# In[35]:


print("Dimension of Team Table Updated is: {}".format(team_table_updated.shape))
print(100*"*")
print(team_table_updated.info())
print(100*"*")
print(team_table_updated.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_table_updated.describe())
print(100*"*")
print(team_table_updated.isnull().sum(axis=0))
print(100*"*")
print(team_table_updated.select_dtypes(exclude=['float64','int64']).apply(lambda x: len(x.unique().tolist()),axis = 0))


# In[36]:


print(len(team_table_updated['team_long_name'].unique().tolist()),      len(team_table_updated['team_short_name'].unique().tolist()))


# In[37]:


my_team = dict()
for i,j in list(team_table_updated.iloc[:,3:].groupby('team_short_name')):
    my_team[i] = j.iloc[:,0].values.tolist()


# In[38]:


{k:v for k,v in my_team.items() if len(v) > 1}
#List of teams with similar short team names


# ### <a id='analyze-team-att-table'></a> Analyzing Team Attributes Table

# In[39]:


print("Dimension of Team Attributes Table is: {}".format(team_att_table.shape))
print(100*"*")
print(team_att_table.info())
print(100*"*")
print(team_att_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_att_table.describe())
print(100*"*")
print(team_att_table.isnull().sum(axis=0))


# Only attribute "buildUpPlayDribbling" has missing values.Look into it. See if other variables at NA instance are strange

# In[40]:


team_att_table.loc[team_att_table['buildUpPlayDribbling'].isnull(),:].head()


# In[41]:


team_att_table.loc[~team_att_table['buildUpPlayDribbling'].isnull(),:].head()


# Does not seem to be related to the other featues and more than 50% of the features are missing. In order to not skew data, drop feaure from dataset and continue analyzing the rest

# In[42]:


team_att_table_updated1 = team_att_table.drop(['buildUpPlayDribbling'],axis = 1)
print("Dimension of Team Attributes Table updated is: {}".format(team_att_table_updated1.shape))
print(100*"*")
print(team_att_table_updated1.info())
print(100*"*")
print(team_att_table_updated1.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_att_table_updated1.describe())
print(100*"*")
print(team_att_table_updated1.isnull().sum(axis=0))


# In[43]:


tat = team_att_table_updated1.loc[:,team_att_table_updated1.columns.tolist()[3:]]


# In[44]:


sns.pairplot(tat)
#Little to no correlation beween any of the continuous features


# In[45]:


fig9, ax9 = plt.subplots(nrows=2,ncols=4)
fig9.set_size_inches(12,6)
for i,j in enumerate(team_att_table_updated1.select_dtypes(include = ['int64']).columns[3:].tolist()):
    sns.distplot(tat.loc[:,j],kde =True,hist = True, ax = ax9[int(i/4)][i%4])
fig9.tight_layout()


# None of the continuous features are normaly distributted or appear to follow exponential family distributions. Multimodal maybe?

# In[46]:


team_att_table_updated1.select_dtypes(include = ['int64']).head()


# In[47]:


sns.boxplot(data = team_att_table_updated1.select_dtypes(include = ['int64']).iloc[:,3:],           orient = 'h')


# In[48]:


fig9, ax9 = plt.subplots(nrows=3,ncols=4)
fig9.set_size_inches(14,8)
for i,j in enumerate(team_att_table_updated1.select_dtypes(include = ['object']).columns[1:].tolist()):
    #sns.countplot(tat.loc[:,j], ax = ax9[int(i/4)][i%4])
    sns.barplot(x = j, y = j, data = tat,            estimator = lambda x: len(x)/len(tat) * 100, ax = ax9[int(i/4)][i%4],           orient = 'v')
    ax9[int(i/4)][i%4].set(xlabel = "")
fig9.tight_layout()


# In[49]:


tat.select_dtypes(include = ['int64']).columns.tolist()


# In[50]:


sns.pairplot(tat,hue = tat.select_dtypes(include = ['object']).          columns.tolist()[1]) 


# When build up play speed is plotted versus the remaining features, build up play speed class appears to perfectly distinguish the plot into sections. This makes sense because as build up play increases in value, the categorical feature of the observation changes from slow to balanced to fast. Most likely, the build up play speed variable was cut into three different and distinct regions, creating the build up play speed class. Either feature can be used (build up play speed or build up play speed class for prediction) but not both since they represent the same thing. Same principle applies for:
# * Build up play speed
# * Build up play passing
# * Chance creation passing
# * Chance creation crossing
# * Chance creation shooting
# * Defense pressure
# * Defense aggression
# * Defense team width
# 
# As for the remaining continuous features, the remaining categorical variables do a poor job of separating/clustering the data. See below for an example

# In[51]:


sns.pairplot(tat,hue = tat.select_dtypes(include = ['object']).          columns.tolist()[12]) 


# In[52]:


fig9, ax9 = plt.subplots(nrows=2,ncols=4)
fig9.set_size_inches(12,6)
for i,j in enumerate(team_att_table_updated1.select_dtypes(include = ['int64']).columns[3:].tolist()):
    sns.boxplot(data = tat, y = j, x = tat.select_dtypes(include = ['object']).columns[3],                                                      ax = ax9[int(i/4)][i%4])
fig9.tight_layout()


# Boxplots, as displayed above, also confirm the fact that some continuous and categorical features are duplicates of each other since the categorical features do an amazing job of separating its respective continuous feature

# ### <a id='conclusion'></a> Conclusion

# Analyzing the league, country, player, player attributes, team and team atributes tables gave a better understanding of the data. Once the respective features are joined and merged with the match table, machine learning algorithms can be used to predict the winner of the future soccer matches in the european league. This kernel also allows for the opportunity to practice using the seaborn library and visualizing the data.   
