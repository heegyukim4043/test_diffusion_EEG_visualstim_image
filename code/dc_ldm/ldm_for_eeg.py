import numpy as np
import wandb
import torch
from .util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
#from einops import repeat

import platform

import os
from .models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ..sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping 
from PIL import Image
def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=512,in_chans=128, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self,
                 metafile,
                 num_voxels=440,
                 cond_dim=1280,          # ★ UNet의 context_dim(=1280)로 맞춤
                 clip_embed_dim=768,     # CLIP 임베딩용(clip loss 쓴다면)
                 global_pool=False,
                 clip_tune=True,
                 cls_tune=False):
        super().__init__()

        if metafile is not None:
            model = create_model_from_config(metafile['config'], num_voxels, global_pool)
            model.load_checkpoint(metafile['model'])
        else:
            model = eeg_encoder(time_len=512, in_chans=128, global_pool=global_pool)

        self.mae = model
        self.fmri_latent_dim = model.embed_dim  # 1024
        self.cond_dim = cond_dim  # 768
        self.global_pool = global_pool

        if cls_tune:
            self.cls_net = classify_network()

        # 1024 -> 1280로 채널 매핑
        self.to_ctx = nn.Linear(self.fmri_latent_dim, self.cond_dim, bias=True)

        # 토큰 길이 128 -> 77로 축약(Stable Diffusion 호환)
        self.seq_pool = nn.AdaptiveAvgPool1d(77)

        self.mapping = None
        if clip_tune:
            self.mapping = mapping(out_dim=clip_embed_dim)

        self.cond_dim = cond_dim
        self.clip_embed_dim = clip_embed_dim

    def forward(self, x):
        z = self.mae(x)              # (B, 128, 1024)
        ctx = self.to_ctx(z)         # (B, 128, 768)
        ctx = ctx.transpose(1, 2)    # (B, 768, 128)
        ctx = self.seq_pool(ctx)     # (B, 768, 77)
        ctx = ctx.transpose(1, 2)    # (B, 77, 768)
        return ctx, z



    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        assert self.mapping is not None
        target_emb = self.mapping(x)    # (.., 768) 보통 CLIP 임베딩과 같은 차원
        return 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()








class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml')
        config = OmegaConf.load(self.config_path)

        # ★ UNet이 1280을 기대한다면 강제로 맞춰주세요
        #config.model.params.unet_config.params.context_dim = 1280

        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim  # == 768


        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
        m, u = model.load_state_dict(pl_sd, strict=False)

        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(
            metafile, num_voxels, cond_dim=self.cond_dim,
            global_pool=global_pool, clip_tune=clip_tune, cls_tune=cls_tune,
            clip_embed_dim=768  # CLIP과 비교하는 경우를 위해 유지
        )

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        #print('\n##### Stage One: only optimize conditional encoders #####')

        IS_WIN = platform.system() == 'Windows'

        num_workers = min(8, os.cpu_count() or 8)
        dataloader = DataLoader(
            dataset,
            batch_size=bs1,
            shuffle=True,
            num_workers=0 if IS_WIN else 8,  # ← 윈도우는 0부터 시작
            pin_memory=True,
            persistent_workers=False,  # ← 윈도우는 False
            # prefetch_factor=2  # 윈도우에선 주석 또는 제거
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=bs1,
            shuffle=False,
            num_workers=0 if IS_WIN else 8,
            pin_memory=True,
            persistent_workers=False,
            # prefetch_factor=2
        )

        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                # print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)



