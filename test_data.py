import h5py, torch, numpy as np

with h5py.File('preproc_data/sub_01.mat','r') as f:
    X = np.array(f['X'])            # 현재 [C,T,N]이면 밑에서 전치
    y = np.array(f['y']).squeeze().astype(np.uint8)
if X.shape[0] == 32:  # C×T×N -> N×C×T
    X = np.transpose(X, (2,0,1)).astype('float32')
# 정규화 통계(5~95pcts) 계산 → eeg_5_95_std.pth
p05 = np.percentile(X, 5, axis=(0,2))
p95 = np.percentile(X,95, axis=(0,2))
torch.save({'p05': torch.from_numpy(p05).float(),
            'p95': torch.from_numpy(p95).float()}, 'datasets/eeg_5_95_std.pth')
# split 인덱스 저장(예: 7:1:2)
N = X.shape[0]; idx = np.random.RandomState(42).permutation(N)
ntr, nva = int(N*0.7), int(N*0.1)
splits = {'train': torch.from_numpy(idx[:ntr]).long(),
          'val':   torch.from_numpy(idx[ntr:ntr+nva]).long(),
          'test':  torch.from_numpy(idx[ntr+nva:]).long()}
torch.save(splits, 'datasets/block_splits_by_image_single.pth')
# (필요 시) X, y도 torch.save로 별도 저장하고 dataset.py에서 읽게 구성
