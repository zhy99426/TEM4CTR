import pickle as pkl
import pandas as pd
import random
import numpy as np
random.seed(42)
np.random.seed(42)

RAW_DATA_FILE = './data/Alibaba/raw_sample.csv'
DATASET_PKL = './data/Alibaba/dataset.pkl'
Test_File = "./data/processed/Alibaba/alibaba_test_align.txt"
Train_File = "./data/processed/Alibaba/alibaba_train_align.txt"
Train_handle = open(Train_File, 'w')
Test_handle = open(Test_File, 'w')
Feature_handle = open("./data/processed/Alibaba/alibaba_feature_align.pkl",'wb')

MAX_LEN_ITEM = 30
RANGE = 5

def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE)
    return df

def remap(df):
    
    item_key = sorted(df['adgroup_id'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))

    df['adgroup_id'] = df['adgroup_id'].map(lambda x: item_map[x])

    user_key = sorted(df['user'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(item_len, item_len + user_len)))
    df['user'] = df['user'].map(lambda x: user_map[x])

    print(item_len, user_len)
    return df, item_len, user_len + item_len


def gen_user_item_group(df, item_cnt):

    user_df = df.sort_values(['user', 'time_stamp']).groupby('user')
    item_df = df.sort_values(['adgroup_id', 'time_stamp']).groupby('adgroup_id')

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, dataset_pkl):
    train_sample_list = []
    test_sample_list = []

    # get each user's last touch point time

    print(len(user_df))

    user_last_touch_time = []
    for user, hist in user_df:
        user_last_touch_time.append(hist['time_stamp'].tolist()[-1])
    print("get user last touch time completed")

    split_time = 1494604799

    cnt = 0
    for user, hist_full in user_df:
        cnt += 1
        print(cnt)

        hist_full = hist_full.reset_index()
           
        click_index = hist_full.index[hist_full["clk"] == 1]
        unclick_index = hist_full.index[hist_full["nonclk"] == 1]
        # import pdb;pdb.set_trace()
        if hist_full.iloc[-1]["time_stamp"] < split_time:
            rand_num = min(len(click_index), len(unclick_index))
            rand = np.append(np.random.choice(click_index, rand_num, replace=False), np.random.choice(unclick_index, rand_num, replace=False))

        else:
            rand = []
            if len(click_index):
                rand.append(click_index[-1])
            if len(unclick_index):
                rand.append(unclick_index[-1])

            
        unclick_hist_full = []
        hist_full_v = hist_full.values[:, 1:]

        for i in click_index:
            count = 0
            j = 0
            while i > j and count < RANGE:
                j += 1
                if hist_full_v[i-j][-2] == 1:
                    unclick_hist_full.insert(len(unclick_hist_full)-count, hist_full_v[i-j])
                    count += 1
            for _ in range(RANGE-count):
                unclick_hist_full.insert(len(unclick_hist_full)-count, np.array([0,0,0,0,0,0]))
            count = 0
            j = 0
            while len(hist_full)-i-1 > j and count < RANGE:
                j += 1
                if hist_full_v[i+j][-2] == 1:
                    count += 1
                    unclick_hist_full.append(hist_full_v[i+j])
            for _ in range(RANGE-count):
                unclick_hist_full.append(np.array([0,0,0,0,0,0]))
        
        for l in rand:
            if l <2 : continue
            hist = hist_full[:int(l)+1]
            item_hist = hist['adgroup_id'].tolist()
            target_item_time = hist['time_stamp'].tolist()[-1]
            
            target_item = item_hist[-1]

            label = hist['clk'].tolist()[-1]
            test = (target_item_time > split_time)

            target_item = item_hist[-1]
            
            if label == 0:
                while target_item == item_hist[-1]:
                    target_item = random.randint(0, item_cnt-1)

            click_hist = hist[hist["clk"]==1]
            click_index = hist.index[hist["clk"]==1]
            unclick_hist = pd.DataFrame(columns=['index', 'user', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk'])
            if len(click_hist) == 0:
                unclick_hist = hist[hist["nonclk"]==1][:-1]
            else:
                unclick_hist = pd.DataFrame(unclick_hist_full[:2*RANGE*len(click_hist)], columns=['user', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk'])
                unclick_hist[unclick_hist["time_stamp"] > target_item_time] = 0
                unclick_hist[(unclick_hist["time_stamp"] == target_item_time) & (unclick_hist["adgroup_id"] == target_item)] = 0
            
            click_item_hist = click_hist['adgroup_id'].tolist()
            unclick_item_hist = unclick_hist['adgroup_id'].tolist()


            # the item history part of the sample
            click_item_part = []
            for i in range(len(click_item_hist) - 1 if label else len(click_item_hist)):
                click_item_part.append([user, click_item_hist[i]])

            unclick_item_part = []
            for i in range(len(unclick_item_hist) - 2*RANGE if label else len(unclick_item_hist)):
                unclick_item_part.append([user, unclick_item_hist[i]])
            if len(click_item_part)!= 0 and len(click_item_part) * 2* RANGE != len(unclick_item_part):
                import pdb;pdb.set_trace()
            # choose the item side information: which user has clicked the target item
            # padding history with 0
            if len(click_item_part) <= MAX_LEN_ITEM:
                click_item_part_pad =  [[0] * 2] * (MAX_LEN_ITEM - len(click_item_part)) + click_item_part
            else:
                click_item_part_pad = click_item_part[len(click_item_part) - MAX_LEN_ITEM:len(click_item_part)]
            if len(unclick_item_part) <= 2*RANGE*MAX_LEN_ITEM:
                unclick_item_part_pad =  [[0] * 2] * (2*RANGE*MAX_LEN_ITEM - len(unclick_item_part)) + unclick_item_part
            else:
                unclick_item_part_pad = unclick_item_part[len(unclick_item_part) - 2*RANGE*MAX_LEN_ITEM:len(unclick_item_part)]
            # gen sample

            if test:
                click_item_list = []
                unclick_item_list = []
                for i in range(len(click_item_part_pad)):
                    click_item_list.append(click_item_part_pad[i][1])
                for i in range(len(unclick_item_part_pad)):
                    unclick_item_list.append(unclick_item_part_pad[i][1])
                test_sample_list.append(str(user) + "\t" + str(target_item) + "\t" + str(label) + "\t" + ",".join(map(str, click_item_list)) + "\t" + ",".join(map(str, unclick_item_list)) + "\n")
            else:
                click_item_list = []
                unclick_item_list = []
                for i in range(len(click_item_part_pad)):
                    click_item_list.append(click_item_part_pad[i][1])
                for i in range(len(unclick_item_part_pad)):
                    unclick_item_list.append(unclick_item_part_pad[i][1])
                train_sample_list.append(str(user) + "\t" + str(target_item) + "\t" + str(label) + "\t" + ",".join(map(str, click_item_list)) + "\t" + ",".join(map(str, unclick_item_list)) +"\n")

    train_sample_length_quant = len(train_sample_list)//256*256
    test_sample_length_quant = len(test_sample_list)//256*256
        
    print("length",len(train_sample_list))
    train_sample_list = train_sample_list[:int(train_sample_length_quant)]
    test_sample_list = test_sample_list[:int(test_sample_length_quant)]
    random.shuffle(train_sample_list)
    print("length",len(train_sample_list))
    return train_sample_list, test_sample_list


def produce_neg_item_hist_with_cate(train_file, test_file):
    for line in train_file:
        Train_handle.write(line.strip() + "\n" )
        
    for line in test_file:
        Test_handle.write(line.strip() + "\n" )

def main():
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size = remap(df)
    print("item count:", item_cnt)
    pkl.dump(item_cnt+1, Feature_handle)

    user_df, item_df = gen_user_item_group(df, item_cnt)
    train_sample_list, test_sample_list = gen_dataset(user_df, item_df, item_cnt, DATASET_PKL)
    produce_neg_item_hist_with_cate(train_sample_list, test_sample_list)


if __name__ == '__main__':
    main()

