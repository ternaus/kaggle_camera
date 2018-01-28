# For validation I will use:
#  * 25 last images from original train data
#  * 25 images from the external data
#  * 25 images from most confident preedictions in test
#  
# Everything is per class.

from pathlib import Path
import pandas as pd


data_path = Path('data')


test_preds = pd.read_csv(str(data_path / 'Votings_stats.csv'))


test_preds.head()

temp = []

for a, df in test_preds[['fname', 'is_manip', 'best_model', 'votes']].groupby(['best_model', 'is_manip']):
    temp += [df.sort_values('votes', ascending=False).head(25)]

test_preds_trunc = pd.concat(temp)


test_preds_trunc.head()


test_preds_trunc['file_name'] = test_preds_trunc['fname'].apply(lambda x: (data_path / 'test' / x).absolute(), 1)


test_preds_trunc = test_preds_trunc.rename(columns={'best_model': 'target'})


test_preds_trunc.to_csv(str(data_path / 'test_preds_trunc.csv'), index=False)


train_path = data_path / 'train'


train_file_names = list(train_path.glob('**/*.*'))


train_file_names = [x.absolute() for x in train_file_names]


df = pd.DataFrame({'file_name': train_file_names})


df['fname'] = df['file_name'].apply(lambda x: x.name, 1)


df['target'] = df['file_name'].apply(lambda x: x.parent.name, 1)


df['ind'] = df['fname'].str.extract('.*\)(\d+)\..*').astype(int)


temp_val = []

for target, dft in df.groupby('target'):
    temp_val += [dft.sort_values(by='ind', ascending=False).head(25)]


df_train_val = pd.concat(temp_val)


df_train_train = df[df['ind'] <= 250]


df_train_val.to_csv(str(data_path / 'val_df.csv'), index=False)


df_train_train.to_csv(str(data_path / 'train_df.csv'), index=False)


flickr_path = data_path / 'flickr_images'


flickr_file_names = list(flickr_path.glob('**/*.*'))


flickr_file_names = [x.absolute() for x in flickr_file_names]


good_files = pd.read_csv(str(data_path / 'good_jpgs'), header=None)


good_files['target'] = good_files[0].str.split('/').str.get(1)


good_files['fname'] = good_files[0].apply(lambda x: x.split('/')[-1], 1)


good_files['file_name'] = good_files.apply(lambda x: (flickr_path / x['target'] / x['fname']).absolute(), 1)

# There is no Motorola X!


map_phone = {'iphone_6': 'iPhone-6', 
             'nexus_6': 'Motorola-Nexus-6', 
             'nexus_5x': 'LG-Nexus-5x', 
             'moto_maxx': 'Motorola-Droid-Maxx', 
             'htc_m7': 'HTC-1-M7',
             'iphone_4s': 'iPhone-4s',
             'sony_nex7': 'Sony-NEX-7', 
             'samsung_s4': 'Samsung-Galaxy-S4', 
             'samsung_note3': 'Samsung-Galaxy-Note3'}


good_files['target'] = good_files['target'].apply(lambda x: map_phone[x], 1)


df_flickr = pd.DataFrame({'file_name': flickr_file_names})


df_flickr['target'] = df_flickr['file_name'].apply(lambda x: x.parent.name, 1)


df_flickr['fname'] = df_flickr['file_name'].apply(lambda x: x.name, 1)


Moto_X = df_flickr[df_flickr['target'] == 'Motorola-X']


to_good_moto_x = Moto_X.head(25)

good_files = pd.concat([good_files, to_good_moto_x]).reset_index(drop=True)


temp_flickr = []

for target, dft in good_files.groupby('target'):
    temp_flickr += [dft.sort_values(by='fname', ascending=False).head(25)]

flickr_val = pd.concat(temp_flickr)

flickr_val.to_csv(str(data_path / 'flickr_val.csv'), index=False)

flickr_val['id'] = flickr_val['target'] + '_' + flickr_val['fname']


df_flickr['id'] = df_flickr['target'] + '_' + df_flickr['fname']


a = pd.DataFrame({'id': flickr_val['id'].values, 'is_val': flickr_val['file_name'].notnull()})

flickr_val['id'] = flickr_val['id'].astype(str)
df_flickr['id'] = df_flickr['id'].astype(str)

train_good = df_flickr.merge(a, on='id', how='outer')


flickr_val_df = train_good[train_good['is_val'].notnull()]
flickr_train_df = train_good[train_good['is_val'].isnull()]


flickr_val_df.to_csv(str(data_path / 'flickr_val.csv'), index=False)


flickr_train_df.to_csv(str(data_path / 'flickr_train.csv'), index=False)

