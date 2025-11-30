from .datasets.my_eeg import MyEEGDataset  # ?뚯씪 寃쎈줈??留욊쾶 import

def build_dataset(args, split):
    if args.dataset == 'MYEEG':
        return MyEEGDataset(
            root=args.root_path,
            split_key=args.split_key,
            split=split,             # 'train' | 'val' | 'test'
            image_size=512,
        )


