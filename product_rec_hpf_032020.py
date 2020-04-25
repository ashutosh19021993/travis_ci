# %%
# data management and transformation 
import pandas as pd 
import numpy as np
import glob
from datetime import datetime as dt
from datetime import timedelta 
from datetime import date

# viz
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

# prediction
import sklearn 
from sklearn.model_selection import train_test_split

# TODO alter calc_cust_rfm_features & calc_date_rfm_summary to take type dict
# TODO class for cusotmer (brand) and products 
# TODO fix error "time data '2018-01-03' does not match format '%Y%m%d'" in rfm functions...    
    # current solution is to rerun combine_files
# TODO error handling / try catch to functions

# %% 
def combine_files(path):

    all_files = glob.glob(path + "/*.csv")
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0) #assumes identical headers
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)    
    return frame

def create_brand_dict(transaction_df):

    frames = {}
    brands = transaction_df['BRAND'].unique()

    for brand in brands:
        #do some calcs to get a dataframe called 'df'
        df = transaction_df[(transaction_df['BRAND'] == brand)]
        frames[brand] = df

    return frames
 
def calc_cust_rfm_features(transaction_df): 
    # returns a df with additional features (columns) 
    # grouped on customer id 
        # do we need to include a transaction count limiter? 
        # e.g. frequency || season || time 
    

    mod_transaction_df = pd.DataFrame()
    todays_date = dt.today().date()

    # remove outliers 
    # TODO outlier removal may be necessary at transaction level, prior to grouping (quantiles)
    t_df = transaction_df[(transaction_df['AMOUNT'] > 0)]
    
    # solves for unexpected datetime behavior (likely an environment logic error)
    if (isinstance(t_df['DATE_D'].loc[0], date) == False): # if date is not of type datetime
        # convert str from YYYYMMDD to MMDDYYY then convert to datetime obj     
        t_df['DATE_D'] = t_df['DATE_D'].apply(lambda x: dt.strptime(str(x), '%Y%m%d').strftime('%m-%d-%y')) 
        t_df['DATE_D'] = t_df['DATE_D'].apply(lambda x: dt.strptime(x, '%m-%d-%y').date()) 

    # sort group values by date
    t_df = t_df.sort_values(['MEMBER_WID', 'DATE_D'])

    # calc time between purchases 
    t_df['previous_visit'] = t_df.groupby(['MEMBER_WID'])['DATE_D'].shift()
    t_df['days_bw_visits'] = t_df['DATE_D'] - t_df['previous_visit']
    t_df['days_bw_visits'] = t_df['days_bw_visits'].apply(lambda x: x.days)

    t_df['days_to_churn'] = pd.to_timedelta(30, 'd') # arbitrary 30 day value...
    t_df['cutoff_date'] = t_df['previous_visit'] + t_df['days_to_churn'] 
    t_df['churned_flag'] = np.where(t_df['cutoff_date'] > t_df['previous_visit'], 0, 1) 
    # 0 = has not churned 
    # 1 = has churned 
    # TODO deal with NaT values... i.e. those that purchased once, as they may skew churn pcts
    # TODO update churn probanility logic. current issue likeyl with arbitrary 'days to churn' value
 
    # create general aggregations on groups
    aggregations = {
                    'AMOUNT': ['max', 'min', 'mean', 'std','count'],
                    'DATE_D': ['max', 'min'],
                    'TIME_D': ['max', 'min'],
                    'days_bw_visits': ['max', 'min', 'mean', 'std'],
                    'churned_flag': ['count','mean']
                    }

    mod_transaction_df = t_df.groupby(['MEMBER_WID']).agg(aggregations) 
    
    # rename columns
    columns_names = ['max_spend','min_spend','avg_spend','std_spend','frequency',
                     'last_purchase','first_purchase','latest_purchase','earliest_purchase',
                     'max_days_bw_visits','min_days_bw_visits','avg_days_bw_visits','std_days_bw_visits',
                     'churn_count', 'avg_time_churned']

    mod_transaction_df.columns = columns_names

    # add more columns
    mod_transaction_df['age'] = todays_date - mod_transaction_df['first_purchase']
    mod_transaction_df['age_days'] = mod_transaction_df['age'].dt.days.astype('int16')
    mod_transaction_df['recency'] = mod_transaction_df['last_purchase'] - mod_transaction_df['first_purchase'] # duration between first purchase and latest 
    mod_transaction_df['recency_days'] = mod_transaction_df['recency'].dt.days.astype('int16')
    # mod_transaction_df['fl_transaction_diff'] =  mod_transaction_df['latest_purchase'] - mod_transaction_df['first_purchase']
    # mod_transaction_df['t_since_last_visit'] = todays_date - mod_transaction_df['latest_purchase']
    # TODO fix error ^ " unsupported operand type(s) for -: 'str' and 'datetime.date' "

    return mod_transaction_df

'''
hierarchical Poisson factorization, a form of probabilistic matrix 
factorization used for recommender systems with implicit count data, 
based on the paper Scalable Recommendation with Hierarchical Poisson Factorization

why ALS (old)
pro 
- generally good out of the box performance
- comparitively fewer parameters to optimize (easier to build) 
cons
- popularity bias
- item cold start (will not refer to new / lesser selected items)
- lack of scalability (slow / expensive)
- lack of metadata use

why HPF (current) 
pro
- effecient modeling at scale  
- handles sparsity well 
con
- less forgiving w/params
- lack of metadata use

why hybrid via lightFM (future)
pro
- includes user and item meta data
- improved performance  
con
- long run time
- many params 

NOTE 
- requires c compiler
- references
    https://github.com/david-cortes/hpfrec 
    http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf 
    https://github.com/lyst/lightfm 

'''
# %%
path = r'C:/Users/joelp/Google Drive/work/brightloom/data/alsea/test'

# %%
product_counts =  combine_files(path)
# TODO sub combine for brand dict

# %%
aggregations = {
             'PRODUCT_WID': ['count']
                    }

product_counts = product_counts.groupby(['MEMBER_WID', 'PRODUCT_WID']).agg(aggregations) 
product_counts

# %%
product_counts.reset_index(inplace=True) 
product_counts

# %%
columns_names = ['UserId','ItemId','Count']

product_counts.columns = columns_names

# %%
product_info = pd.read_csv(r'C:/Users/joelp/Google Drive/work/brightloom/data/alsea/dims/cat_prods.csv')

# %%
product_info.head(10)

# %%
train, test = train_test_split(product_counts, test_size=.25, random_state=1)

# %%
users_train = set(train.UserId)
items_train = set(train.ItemId)

# %%
test = test.loc[(test.UserId.isin(users_train)) & (test.ItemId.isin(items_train))].reset_index(drop=True)
del users_train, items_train
del product_counts
test.shape

# %%
from hpfrec import HPF

## Full call would be like this:
# recommender = HPF(k=50, a=0.3, a_prime=0.3, b_prime=1.0,
#                  c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
#                  stop_crit='train-llk', check_every=10, stop_thr=1e-3,
#                  maxiter=150, reindex=True, random_seed = 123,
#                  allow_inconsistent_math=False, verbose=True, full_llk=False,
#                  keep_data=True, save_folder=None, produce_dicts=True)

# For more information see the documentation:
# http://hpfrec.readthedocs.io/en/latest/


recommender = HPF(k=50, full_llk=False, random_seed=123,
                  check_every=10, maxiter=150, reindex=True,
                  allow_inconsistent_math=True, ncores=24,
                  verbose=True,
                  save_folder=r'C:/Users/joelp/Google Drive/work/brightloom/data/alsea/results')
recommender.fit(train)

# %%
# common sense checks 
test['Predicted'] = recommender.predict(user=test.UserId, item=test.ItemId)
test['RandomItem'] = np.random.choice(train.ItemId, size=test.shape[0])
test['PredictedRandom'] = recommender.predict(user=test.UserId, item=test.RandomItem)
print("Average prediction for combinations in test set: ", test.Predicted.mean())
print("Average prediction for random combinations: ", test.PredictedRandom.mean())

# %%
test

# %%
'''
As some common sense checks, the predictions should:
- Be higher for this non-zero hold-out sample than for random items
- Produce a good discrimination between random items and those in the hold-out sample (very related to the first point).
- Be correlated with the counts in the hold-out sample
- Follow an exponential distribution rather than a normal or some other symmetric distribution.
'''
# %%
from sklearn.metrics import roc_auc_score

was_purchased = np.r_[np.ones(test.shape[0]), np.zeros(test.shape[0])]
score_model = np.r_[test.Predicted.values, test.PredictedRandom.values]
roc_auc_score(was_purchased, score_model)

# %%
print('correlation of predictions to hold out sample. we want this to be close to 1')
np.corrcoef(test.Count, test.Predicted)[0,1]

# %%
print('this output should follow an exponential distribution rather than a normal or some other symmetric distribution')

_ = plt.hist(test.Predicted, bins=1000) # try alternate bin sizes 
plt.xlim(0,0.4) # adjust x axis range as needed
plt.show()

# %%
'''
Pick 3 random users with a reasonably long history of purchases
Check which products exist in the training data with which the model was fit
See which products the model recommend to them among those which they have not yet purchased
Top-N lists can be made among all items, or across some user-provided subset only
'''
# %%
total_purchases_by_user = train.groupby('UserId').agg({'Count':np.sum})
total_purchases_by_user = total_purchases_by_user[total_purchases_by_user.Count > 3]

np.random.seed(1)
sample_users = np.random.choice(total_purchases_by_user.index, 3)

# %%
print('sample users')
sample_users

# %%
# TODO overlap in test and train sets ocurrs with smaller data sets
print('recommended products for sample user 1')
recommender.topN(user = sample_users[0],
                 n=3, exclude_seen = True) # n = product recommendations

# %%
print('examine individual recommendation details of sample users')
train_rec = pd.merge(train, product_info, left_on='ItemId', right_on='PROD_WID')
train_rec
# TODO potential improvement to model: build model on product line (PR_PROD_LN) then randomly 
# select item from group

# %%
print('top items purchased from sample customer #1')
x = train_rec.loc[train_rec.UserId==sample_users[0]]\
[['PR_PROD_LN','PART_NUM','CATEGORY_CD','Count']].sort_values('Count', ascending=False)\
.head(15)
x

# %%
print('recommendations for sample customer #1')
recommended_list = recommender.topN(sample_users[0], n=5)

product_info[['PR_PROD_LN', 'PART_NUM', 'CATEGORY_CD']]\
[product_info.PROD_WID.isin(recommended_list)].drop_duplicates()

# %%
print('top items purchased from sample customer #2')
y = train_rec.loc[train_rec.UserId==sample_users[1]]\
[['PR_PROD_LN','PART_NUM','CATEGORY_CD','Count']].sort_values('Count', ascending=False)\
.head(15)
y

# %%
print('recommendations for sample customer #2')
recommended_list = recommender.topN(sample_users[1], n=5)

product_info[['PR_PROD_LN', 'PART_NUM', 'CATEGORY_CD']]\
[product_info.PROD_WID.isin(recommended_list)].drop_duplicates()

# %%
print('top items purchased from sample customer #3')
y = train_rec.loc[train_rec.UserId==sample_users[2]]\
[['PR_PROD_LN','PART_NUM','CATEGORY_CD','Count']].sort_values('Count', ascending=False)\
.head(15)
y

# %%
print('recommendations for sample customer #3')
recommended_list = recommender.topN(sample_users[2], n=5)

product_info[['PR_PROD_LN', 'PART_NUM', 'CATEGORY_CD']]\
[product_info.PROD_WID.isin(recommended_list)].drop_duplicates()

# TODO notice WOW ticket recommendations present - wonder if outlier and should be removed
