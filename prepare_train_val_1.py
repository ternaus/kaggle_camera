

from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

data_path = Path('data')

train_path = data_path / 'train'

train_file_names = list(train_path.glob('**/*.*'))

train_file_names = [x.absolute() for x in train_file_names]

df = pd.DataFrame({'file_name': train_file_names})

df['fname'] = df['file_name'].apply(lambda x: x.name, 1)

df['target'] = df['file_name'].apply(lambda x: x.parent.name, 1)

df['ind'] = df['fname'].str.extract('.*\)(\d+)\..*').astype(int)


df['fold'] = np.nan


for fold in range(5):
    df.loc[(fold * 55 < df['ind']) & (df['ind'] <= (fold + 1) * 55), 'fold'] = fold


df['fold'] = df['fold'].astype(int)


# In[351]:


df.to_csv('../data/train_df.csv', index=False)


# # Good data, but not from low quality

# In[352]:


flickr_path = data_path / 'new_flickr'


# In[353]:


flickr_file_names = list(flickr_path.glob('**/*.*'))

flickr_file_names = [x.absolute() for x in flickr_file_names]

flickr_df = pd.DataFrame({'file_name': flickr_file_names})

flickr_df['fname'] = flickr_df['file_name'].apply(lambda x: x.name, 1)

flickr_df['target'] = flickr_df['file_name'].apply(lambda x: x.parent.name, 1)


# In[354]:


def is_file(x):
    try:
        img = Image.open(str(x))
        return img.size[1] > 512
    except:
        return False

flickr_df = flickr_df[~flickr_df['fname'].isin(['38549248515_2c81ef242e_o.jpg', '38549248515_2c81ef242e_o.jpg'])]


flickr_df['is_file'] = flickr_df['file_name'].apply(is_file)


flickr_df['is_file'].mean()


flickr_df = flickr_df[flickr_df['is_file'] ]


flickr_df['is_file'].mean()


flickr_df['target'].value_counts()


flickr_df.head()


flickr_df['id'] = flickr_df['target'] + '_' + flickr_df['fname']


flickr_df['target'].value_counts()


flickr_df.to_csv(str(data_path / 'flickr_df.csv'), index=False)


val_path = data_path / 'new_val'

val_file_names = list(val_path.glob('**/*.*'))

val_file_names = [x.absolute() for x in val_file_names]

val_df = pd.DataFrame({'file_name': val_file_names})

val_df['fname'] = val_df['file_name'].apply(lambda x: x.name, 1)

val_df['target'] = val_df['file_name'].apply(lambda x: x.parent.name, 1)


val_df.head()


max_in_val = val_df['target'].value_counts().max()

val_counts = val_df['target'].value_counts()


grouped = flickr_df.groupby('target')


flickr_df['target'].unique()


keep_in_train = []

add_to_val = []

for class_name in val_counts.index:
    count = val_counts[class_name]
    if count >= max_in_val:
        continue
    dft = grouped.get_group(class_name)
    indx = dft.index
    
    diff = max_in_val - count
    
    keep_in_train += [dft.iloc[diff:]]
    add_to_val += [dft.iloc[:diff]]

flickr_df_new = pd.concat(keep_in_train).reset_index(drop=True)


flickr_df_new['target'].value_counts()


val_df_new = pd.concat([pd.concat(add_to_val), val_df]).reset_index(drop=True)


val_df_new['target'].value_counts()


flickr_df_new.to_csv(str(data_path / 'flickr_df.csv'), index=False)


val_df_new.to_csv(str(data_path / 'val_df.csv'), index=False)

