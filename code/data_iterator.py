import numpy as np

class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen_click=100,
                 maxlen_unclick=100,
                 skip_empty=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 parall=False,
                 with_cate=False
                ):
        self.file_name = source
        self.source = open(source, 'r')
        #self
        self.source_dicts = []
        
        self.batch_size = batch_size
        self.maxlen_click = maxlen_click
        self.maxlen_unclick = maxlen_unclick
        self.minlen = minlen
        self.skip_empty = skip_empty


        self.sort_by_length = sort_by_length

        self.source_buffer = []

        self.k = batch_size
        self.with_cate = with_cate
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
    
    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        hist_click_list = []
        hist_unclick_list = []
        hist_click_cate_list = []
        hist_unclick_cate_list = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                if self.with_cate:
                    uid = int(ss[0])
                    item_id = int(ss[1])
                    cate_id = int(ss[2])
                    label = int(ss[3])

                    click_hist_item = list(map(int, ss[4].split(",")))
                    click_hist_cate = list(map(int, ss[5].split(",")))
                    unclick_hist_item = list(map(int, ss[6].split(",")))
                    unclick_hist_cate = list(map(int, ss[7].split(",")))
                    
                    
                    source.append([uid, item_id, cate_id])
                    target.append([label, 1-label])
                    hist_click_list.append(click_hist_item[-self.maxlen_click:])
                    hist_click_cate_list.append(click_hist_cate[-self.maxlen_click:])
                    hist_unclick_list.append(unclick_hist_item[-self.maxlen_unclick:])
                    hist_unclick_cate_list.append(unclick_hist_cate[-self.maxlen_unclick:])
                else:
                    uid = int(ss[0])
                    item_id = int(ss[1])
                    label = int(ss[2])

                    click_hist_item = list(map(int, ss[3].split(",")))
                    unclick_hist_item = list(map(int, ss[4].split(",")))
                    
                    
                    source.append([uid, item_id])
                    target.append([label, 1-label])
                    hist_click_list.append(click_hist_item[-self.maxlen_click:])
                    hist_unclick_list.append(unclick_hist_item[-self.maxlen_unclick:])
                
                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
        
        uid_array = np.array(source)[:,0]
        item_array = np.array(source)[:,1]

        target_array = np.array(target)

        history_click_array = np.array(hist_click_list)        
        history_unclick_array = np.array(hist_unclick_list)     
        
        history_click_mask_array = np.greater(history_click_array, 0)*1.0
        history_unclick_mask_array = np.greater(history_unclick_array, 0)*1.0   

        if self.with_cate:
            cate_array = np.array(source)[:,2]
            history_click_cate_array = np.array(hist_click_cate_list)        
            history_unclick_cate_array = np.array(hist_unclick_cate_list)
            return (uid_array, item_array, cate_array), (target_array, history_click_array, history_unclick_array, history_click_cate_array, history_unclick_cate_array, history_click_mask_array, history_unclick_mask_array)
        return (uid_array, item_array), (target_array, history_click_array, history_unclick_array, history_click_mask_array, history_unclick_mask_array)


