import pickle as pkl
import pandas as pd
import random
import numpy as np
random.seed(42)
np.random.seed(42)

RAW_TRAIN_DATA_FILE = "./data/MicroVideo-1.7M/train_data.csv"
RAW_TEST_DATA_FILE = "./data/MicroVideo-1.7M/test_data.csv"
DATASET_PKL = "./data/MicroVideo-1.7M/dataset.pkl"
Test_File = "./data/processed/MicroVideo-1.7M/microvideo_test_align.txt"
Train_File = "./data/processed/MicroVideo-1.7M/microvideo_train_align.txt"

Train_handle = open(Train_File, "w")
Test_handle = open(Test_File, "w")

Feature_handle = open("./data/processed/MicroVideo-1.7M/microvideo_feature_align.pkl", "wb")

MAX_LEN_ITEM = 30
RANGE = 5
MAX_LEN_ITEM_UNCLICK = MAX_LEN_ITEM*2*RANGE

def to_df():
    df_train, df_test = pd.read_csv(RAW_TRAIN_DATA_FILE), pd.read_csv(
        RAW_TEST_DATA_FILE
    )
    df_train.columns = ["user_id", "item_id", "cate_id", "is_click", "timestamp"]
    return df_train, df_test


def remap(df_train, df_test):

    item_len = 984983 + 719897

    df_train["item_id"] = df_train["item_id"].map(lambda x: x + 1)
    df_test["item_id"] = df_test["item_id"].map(lambda x: x + 984984)
    df_train["cate_id"] = df_train["cate_id"].map(lambda x: x + 1)
    df_test["cate_id"] = df_test["cate_id"].map(lambda x: x + 1)
    df = pd.concat([df_train, df_test])
    
    user_key = sorted(df["user_id"].unique().tolist())
    user_len = len(user_key)


    print(item_len, user_len)
    return df_train, df_test, item_len, user_len + item_len


def gen_user_item_group(df, item_cnt):
    user_df = df.sort_values(["user_id", "timestamp"]).groupby("user_id")
    item_df = df.sort_values(["item_id", "timestamp"]).groupby("item_id")

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, dataset_pkl, train=True):
    data_list = []

    # get each user's last touch point time
    cate_list = dict()
    for item, hist_full in item_df:
        cate_list[item] = (hist_full['cate_id'].tolist()[0])
    print(len(user_df))

    cnt = 0
    for user, hist_full in user_df:
        cnt += 1
        print(cnt)
        hist_full = hist_full.reset_index()
        click_index = hist_full.index[hist_full["is_click"] == 1]
        unclick_index = hist_full.index[hist_full["is_click"] == 0]
        
        if train:
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
                if hist_full_v[i-j][-2] == 0:
                    unclick_hist_full.insert(len(unclick_hist_full)-count, hist_full_v[i-j])
                    count += 1
            for _ in range(RANGE-count):
                unclick_hist_full.insert(len(unclick_hist_full)-count, np.array([0,0,0,0,0]))
            count = 0
            j = 0
            while len(hist_full)-i-1 > j and count < RANGE:
                j += 1
                if hist_full_v[i+j][-2] == 0:
                    count += 1
                    unclick_hist_full.append(hist_full_v[i+j])
            for _ in range(RANGE-count):
                unclick_hist_full.append(np.array([0,0,0,0,0]))


        for l in rand:
            if l <2 : continue
            
            hist = hist_full[:int(l)+1]
            label = hist["is_click"].tolist()[-1]
            item_hist = hist["item_id"].tolist()
            cate_hist = hist["cate_id"].tolist()

            
            target_item_time = hist["timestamp"].tolist()[-1]

            target_item = item_hist[-1]
            target_cate = cate_hist[-1]
            
            if label == 0:
                while target_item == item_hist[-1]:
                    if train:
                        target_item = random.randint(1, 984983)
                    else:
                        target_item = random.randint(984985, item_cnt)
                    target_cate = cate_list[target_item]
                
            
            click_hist = hist[hist["is_click"] == 1]
            click_index = hist.index[hist["is_click"] == 1]

            if len(click_hist) == 0:
                unclick_hist = hist[hist["is_click"] == 0][:-1]
            else:
                unclick_hist = pd.DataFrame(unclick_hist_full[:2*RANGE*len(click_hist)], columns=["user_id", "item_id", "cate_id", "is_click", "timestamp"])
                unclick_hist[unclick_hist["timestamp"] > target_item_time] = 0
                unclick_hist[(unclick_hist["timestamp"] == target_item_time) & (unclick_hist["item_id"] == target_item)] = 0

            click_item_hist = click_hist["item_id"].tolist()
            unclick_item_hist = unclick_hist["item_id"].tolist()
            click_cate_hist = click_hist["cate_id"].tolist()
            unclick_cate_hist = unclick_hist["cate_id"].tolist()

            
            # the item history part of the sample
            click_item_part = []
            for i in range(len(click_item_hist) - 1 if label else len(click_item_hist)):
                click_item_part.append([user, click_item_hist[i], click_cate_hist[i]])
            # click_item_part.append([user, target_item])
            unclick_item_part = []
            for i in range(len(unclick_item_hist) - 2*RANGE if label else len(unclick_item_hist)):
                unclick_item_part.append([user, unclick_item_hist[i], unclick_cate_hist[i]])
                
            if len(click_item_part)!= 0 and len(click_item_part) * 2* RANGE != len(unclick_item_part):
                import pdb;pdb.set_trace()
            # choose the item side information: which user has clicked the target item
            # padding history with 0
            if len(click_item_part) <= MAX_LEN_ITEM:
                click_item_part_pad = [[0] * 3] * (
                    MAX_LEN_ITEM - len(click_item_part)
                ) + click_item_part
            else:
                click_item_part_pad = click_item_part[
                    len(click_item_part) - MAX_LEN_ITEM : len(click_item_part)
                ]
            if len(unclick_item_part) <= MAX_LEN_ITEM_UNCLICK:
                unclick_item_part_pad = [[0] * 3] * (
                    MAX_LEN_ITEM_UNCLICK - len(unclick_item_part)
                ) + unclick_item_part
            else:
                unclick_item_part_pad = unclick_item_part[
                    len(unclick_item_part) - MAX_LEN_ITEM_UNCLICK : len(unclick_item_part)
                ]
            # gen sample
            click_item_list = []
            unclick_item_list = []
            click_cate_list = []
            unclick_cate_list = []
            for i in range(len(click_item_part_pad)):
                click_item_list.append(click_item_part_pad[i][1])
                click_cate_list.append(click_item_part_pad[i][2])
            for i in range(len(unclick_item_part_pad)):
                unclick_item_list.append(unclick_item_part_pad[i][1])
                unclick_cate_list.append(unclick_item_part_pad[i][2])
            data_list.append(
                str(user)
                + "\t"
                + str(target_item)
                + "\t"
                + str(target_cate)
                + "\t"
                + str(label)
                + "\t"
                + ",".join(map(str, click_item_list))
                + "\t"
                + ",".join(map(str, click_cate_list))
                + "\t"
                + ",".join(map(str, unclick_item_list))
                + "\t"
                + ",".join(map(str, unclick_cate_list))
                + "\n"
            )

    data_list_length_quant = len(data_list) // 256 * 256

    print("length", len(data_list))
    data_list = data_list[: int(data_list_length_quant)]
    if train:
        random.shuffle(data_list)
    print("length", len(data_list))
    return data_list


def produce_neg_item_hist_with_cate(train_file, test_file):
    
    for line in train_file:
        Train_handle.write(line.strip() + "\n")

    for line in test_file:
        Test_handle.write(line.strip() + "\n")


def main():
    df_train, df_test = to_df()
    df_train, df_test, item_cnt, feature_size = remap(df_train, df_test)
    print("item count:", item_cnt)
    pkl.dump(item_cnt + 1, Feature_handle)

    user_df_train, item_df_train = gen_user_item_group(df_train, item_cnt)
    user_df_test, item_df_test = gen_user_item_group(df_test, item_cnt)

    train_sample_list = gen_dataset(user_df_train, item_df_train, item_cnt, DATASET_PKL)
    test_sample_list = gen_dataset(user_df_test, item_df_test, item_cnt, DATASET_PKL, train=False)
    produce_neg_item_hist_with_cate(train_sample_list, test_sample_list)


if __name__ == "__main__":
    main()
