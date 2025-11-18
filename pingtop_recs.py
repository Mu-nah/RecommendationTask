import os
import pandas as pd
import numpy as np
from datetime import timedelta

# ---------- CONFIG ----------
INTERACTIONS_CSV = "interactions.csv"   # columns: user_id,ping_id,event_type,watch_time_sec,event_timestamp
PINGS_CSV = "pings.csv"                # columns: ping_id,creator_id,main_hashtag,category,duration_sec,created_at
USERS_CSV = "users.csv"                # columns: user_id,country,signup_date,age
OUT_DIR = "./output"
EXAMPLE_USERS = ["u1","u2","u3"]       # change as needed

# ---------- helpers ----------
def load_csv(path):
    return pd.read_csv(path)

# ---------- load data ----------
inter = load_csv(INTERACTIONS_CSV)
pings = load_csv(PINGS_CSV)
users = load_csv(USERS_CSV)

# normalize column names (flexible)
inter.columns = inter.columns.str.strip().str.lower()
pings.columns = pings.columns.str.strip().str.lower()
users.columns = users.columns.str.strip().str.lower()

# column names handling
# interactions: user_id, ping_id, event_type, watch_time_sec, event_timestamp
# pings: ping_id / oping_id, creator_id / ocreator_id, main_hashtag / omain_hashtag, category / ocategory, duration_sec / oduration_sec, created_at / ocreated_at
# users: user_id / ouser_id, signup_date / osignup_date

# make column name canonical mapping (scalable measures)
def canonicalize(df, mapping):
    df = df.rename(columns={k:v for k,v in mapping.items() if k in df.columns})
    return df

inter = canonicalize(inter, {
    'ouser_id':'user_id','ouser_id':'user_id',
    'oping_id':'ping_id'
})
pings = canonicalize(pings, {
    'oping_id':'ping_id','ocreator_id':'creator_id','omain_hashtag':'main_hashtag','ocategory':'category','oduration_sec':'duration_sec','ocreated_at':'created_at'
})
users = canonicalize(users, {
    'ouser_id':'user_id','osignup_date':'signup_date'
})

# parse timestamps
if 'event_timestamp' in inter.columns:
    inter['event_timestamp'] = pd.to_datetime(inter['event_timestamp'], errors='coerce')
if 'created_at' in pings.columns:
    pings['created_at'] = pd.to_datetime(pings['created_at'], errors='coerce')
if 'signup_date' in users.columns:
    users['signup_date'] = pd.to_datetime(users['signup_date'], errors='coerce')

# fallback: if ping durations missing, fill with median or default 30s (scalable measures)
if 'duration_sec' not in pings.columns or pings['duration_sec'].isna().all():
    pings['duration_sec'] = 30
else:
    pings['duration_sec'] = pings['duration_sec'].fillna(pings['duration_sec'].median())

# ensure string ids
inter['ping_id'] = inter['ping_id'].astype(str)
inter['user_id'] = inter['user_id'].astype(str)
pings['ping_id'] = pings['ping_id'].astype(str)
users['user_id'] = users['user_id'].astype(str)

# ------------ Engagement computation -------------
# compute watch_time_ratio for view events
inter['watch_time_sec'] = inter.get('watch_time_sec', 0).fillna(0).astype(float)
inter = inter.merge(pings[['ping_id','duration_sec']], on='ping_id', how='left')
inter['duration_sec'] = inter['duration_sec'].fillna(30)
inter['watch_time_ratio'] = np.where(inter['event_type']=='view', inter['watch_time_sec'] / inter['duration_sec'], np.nan)

# event weights
weights = {
    'view': 1.0,           # multiplied by watch_time_ratio
    'like': 2.0,
    'comment': 3.0,
    'share': 4.0,
    'follow_creator': 2.0,
    'impression': 0.1
}

def event_score(row):
    et = row['event_type']
    if et == 'view':
        r = row['watch_time_ratio'] if not pd.isna(row['watch_time_ratio']) else 0.0
        return weights['view'] * r
    return weights.get(et, 0.0)

inter['event_score'] = inter.apply(event_score, axis=1)

# aggregate per (user,ping)
user_ping = inter.groupby(['user_id','ping_id'], as_index=False).agg(
    engagement_score=('event_score','sum'),
    last_ts=('event_timestamp','max'),
    n_events=('event_type','count')
)

# global ping score
ping_global = user_ping.groupby('ping_id', as_index=False).agg(
    global_engagement=('engagement_score','sum'),
    users_interacted=('user_id','nunique')
).sort_values('global_engagement', ascending=False)

# save results

os.makedirs(OUT_DIR, exist_ok=True)
ping_global.to_csv(os.path.join(OUT_DIR,'ping_global.csv'), index=False)
user_ping.to_csv(os.path.join(OUT_DIR,'user_ping.csv'), index=False)

# ------------ Metrics asked ------------
# distribution of watch_time_ratio (only view events)
views = inter[inter['event_type']=='view'].copy()
wtr_stats = views['watch_time_ratio'].describe(percentiles=[0.25,0.5,0.75,0.9])
wtr_stats.to_csv(os.path.join(OUT_DIR,'watch_time_ratio_stats.csv'))

# top 10 pings by global engagement
top10 = ping_global.merge(pings, on='ping_id', how='left').head(10)
top10.to_csv(os.path.join(OUT_DIR,'top10_pings.csv'), index=False)

# compare new users vs existing users
if 'signup_date' in users.columns:
    max_signup = users['signup_date'].max()
    cutoff = max_signup - pd.Timedelta(days=7)
    users['is_new'] = users['signup_date'] >= cutoff
else:
    users['is_new'] = False

inter_u = inter.merge(users[['user_id','is_new']], on='user_id', how='left')

avg_watch_by_group = inter_u[inter_u['event_type']=='view'].groupby('is_new').agg(avg_watch_time_ratio=('watch_time_ratio','mean'), view_count=('watch_time_ratio','count')).reset_index()
avg_watch_by_group.to_csv(os.path.join(OUT_DIR,'avg_watch_by_group.csv'), index=False)

# average number of pings interacted (view/like/comment/share) per user
valid = inter[inter['event_type'].isin(['view','like','comment','share'])]
pings_per_user = valid.groupby('user_id')['ping_id'].nunique().reset_index(name='n_pings')
pings_per_user = pings_per_user.merge(users[['user_id','is_new']], on='user_id', how='right').fillna({'n_pings':0})
avg_pings_by_group = pings_per_user.groupby('is_new').agg(avg_pings=('n_pings','mean'), user_count=('user_id','count')).reset_index()
avg_pings_by_group.to_csv(os.path.join(OUT_DIR,'avg_pings_by_group.csv'), index=False)

# ------------ Short interpretation (3-6 bullets) ------------
# I'll print these in final report (noting them here for reference)

# ------------ Simple recommender (heuristic) ------------
# Build item features: global_popularity (normalized), category, hashtag, creator, freshness
items = pings.copy().rename(columns={'ping_id':'ping_id'})
items['global_pop'] = items['ping_id'].map(ping_global.set_index('ping_id')['global_engagement']).fillna(0.0)
if items['global_pop'].max() > items['global_pop'].min():
    items['global_pop_norm'] = (items['global_pop'] - items['global_pop'].min()) / (items['global_pop'].max() - items['global_pop'].min())
else:
    items['global_pop_norm'] = 0.0
items['created_at'] = pd.to_datetime(items.get('created_at'))
now = items['created_at'].max() if items['created_at'].notna().any() else pd.to_datetime('2024-02-20')
items['age_days'] = (now - items['created_at']).dt.days.fillna(30)
items['freshness'] = 1 / (1 + items['age_days'])

# user affinity features
int_meta = inter.merge(items[['ping_id','category','main_hashtag','creator_id']], on='ping_id', how='left')
user_cat = int_meta.groupby(['user_id','category']).size().reset_index(name='count')
user_total = int_meta.groupby('user_id').size().reset_index(name='total')
user_cat = user_cat.merge(user_total, on='user_id')
user_cat['cat_affinity'] = user_cat['count'] / user_cat['total']

user_creator = int_meta.groupby(['user_id','creator_id']).size().reset_index(name='count')
user_creator = user_creator.merge(user_total, on='user_id')
user_creator['creator_affinity'] = user_creator['count'] / user_creator['total']

# scoring function
def recommend_for_user(uid, topk=10, alpha=0.5, beta=0.25, gamma=0.15, delta=0.10):
    df = items.copy()
    ucat = user_cat[user_cat['user_id']==uid].set_index('category')['cat_affinity'].to_dict() if not user_cat.empty else {}
    ucre = user_creator[user_creator['user_id']==uid].set_index('creator_id')['creator_affinity'].to_dict() if not user_creator.empty else {}
    df['user_cat_aff'] = df['category'].map(ucat).fillna(0.0)
    df['user_cre_aff'] = df['creator_id'].map(ucre).fillna(0.0)
    df['score'] = alpha * df['global_pop_norm'] + beta * df['user_cat_aff'] + gamma * df['user_cre_aff'] + delta * df['freshness']
    # exclude items user already interacted with
    seen = inter[inter['user_id']==uid]['ping_id'].unique().tolist()
    df = df[~df['ping_id'].isin(seen)]
    df = df.sort_values('score', ascending=False).head(topk)
    # add reason column
    reasons=[]
    for _, r in df.iterrows():
        parts=[]
        if r['user_cat_aff']>0: parts.append(f"prefers cat={r['category']}")
        if r['user_cre_aff']>0: parts.append("engaged this creator")
        if r['global_pop_norm']>0.5: parts.append("globally popular")
        if r['freshness']>0.02: parts.append("recent")
        reasons.append("; ".join(parts) if parts else "popular/new")
    df['reason'] = reasons
    return df[['ping_id','score','category','main_hashtag','creator_id','reason']]

# produce recommendations for example users
os.makedirs(os.path.join(OUT_DIR,'recs'), exist_ok=True)
for uid in EXAMPLE_USERS:
    recs = recommend_for_user(uid, topk=10)
    recs.to_csv(os.path.join(OUT_DIR,'recs', f'recs_{uid}.csv'), index=False)

print("Done. Outputs in:", OUT_DIR)
