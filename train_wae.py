import sys
from tqdm import tqdm
import pandas as pd

import torch
import torch.optim as optim

from models.mutils import save_model
import utils
from models import losses, mutils
from torch.utils.data import DataLoader
from tb_json_logger import log_value
from sample_pipeline import compute_modlamp


def train_wae(cfg, model, dataset, device):
    cfgv = cfg.wae
    dataloader = DataLoader(dataset, batch_size=cfgv.batch_size, shuffle=True)
    print('Training base wae ...')
    trainer_Discriminator = optim.RMSprop(model.discriminator_params(), lr=cfgv.lr_D)
    trainer_Decoder = optim.RMSprop(model.decoder_params(), lr=cfgv.lr_G)
    trainer_Encoder = optim.RMSprop(model.encoder_params(), lr=cfgv.lr_G)

    for it in tqdm(range(cfgv.s_iter, cfgv.s_iter + cfgv.n_iter + 1), disable=None):
        if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
            def tblog(k, v):
                log_value('train_' + k, v, it)
        else:
            tblog = lambda k, v: None

        for index,item in enumerate(dataloader):
            inputs = item[0].to(device)
            input_lens = item[-1]

            beta = utils.anneal(cfgv.beta, it)

            trainer_Discriminator.zero_grad()
            trainer_Decoder.zero_grad()
            trainer_Encoder.zero_grad()

            # ============ Train Discriminator ============ #

            if index % cfgv.n_critic != 0:
                utils.frozen_params(model)
                utils.free_params(model.discriminator)

                (z_mu, z_logvar), z, (dec_logits_z, dec_logits_gau) = model(inputs, input_lens, sample_z=1)

                feature_real, critic_real = model.forward_discriminator(inputs)
                feature_recon, critic_recon = model.forward_discriminator(dec_logits_z)
                feature_fake, critic_fake = model.forward_discriminator(dec_logits_gau)

                loss_discriminator = (-torch.mean(critic_real) + torch.mean(critic_fake)\
                                        + torch.mean(critic_recon) * cfgv.lambda_logvar_Dis)
                loss_discriminator = utils.check_is_nan(loss_discriminator)

                loss_discriminator.backward(retain_graph=True)

                grad_norm_Dis = torch.nn.utils.clip_grad_norm_(model.discriminator_params(), cfgv.clip_grad)

                trainer_Discriminator.step()

                for p in model.discriminator.parameters():
                    p.data.clamp_(-cfgv.clip_value, cfgv.clip_value)

                tblog('z_mu_L1', z_mu.data.abs().mean().item())
                tblog('z_logvar', z_logvar.data.mean().item())
                tblog('beta', beta)
                tblog('L_discriminator',loss_discriminator.item())

            # ============ Train VAE ============ #

            if index % cfgv.n_critic == 0:
                utils.frozen_params(model)
                utils.free_params(model.word_emb)
                utils.free_params(model.encoder)
                utils.free_params(model.decoder)

                (z_mu, z_logvar), z, (dec_logits_z, dec_logits_gau) = model(inputs, input_lens, sample_z=1)

                feature_real, critic_real = model.forward_discriminator(inputs)
                feature_recon, critic_recon = model.forward_discriminator(dec_logits_z)
                feature_fake, critic_fake = model.forward_discriminator(dec_logits_gau)

                wae_mmd_loss = losses.wae_mmd_gaussianprior(z, method='full_kernel')
                wae_mmd_loss = utils.check_is_nan(wae_mmd_loss)

                z_logvar_L1 = z_logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mb.

                loss_recon_logits = losses.recon_dec(inputs, dec_logits_z)
                loss_recon_logits = utils.check_is_nan(loss_recon_logits)
                loss_recon_feature = torch.mean(torch.sqrt(torch.sum(torch.square(feature_real-feature_recon),dim=1)))

                loss_dis = - torch.mean(critic_fake) - torch.mean(critic_recon)
                
                loss_decoder = cfgv.lambda_recon * (loss_recon_logits + loss_recon_feature) \
                    + beta * wae_mmd_loss + cfgv.lambda_logvar_Dis * loss_dis \
                    + cfgv.lambda_logvar_L1 * z_logvar_L1

                loss_decoder.backward(retain_graph=True)
                grad_norm_De = torch.nn.utils.clip_grad_norm_(model.decoder_params(), cfgv.clip_grad)

                loss_encoder = (loss_recon_logits + loss_recon_feature) + beta * wae_mmd_loss \
                    + cfgv.lambda_logvar_L1 * z_logvar_L1

                loss_encoder.backward(retain_graph=True)
                grad_norm_En = torch.nn.utils.clip_grad_norm_(model.encoder_params(), cfgv.clip_grad)

                trainer_Decoder.step()
                trainer_Encoder.step()

                tblog('z_mu_L1', z_mu.data.abs().mean().item())
                tblog('z_logvar', z_logvar.data.mean().item())
                tblog('beta', beta)
                tblog('L_wae_recon_logits',loss_recon_logits.item())
                tblog('L_wae_recon_feature',loss_recon_feature.item())
                tblog('z_logvar_L1', z_logvar_L1.item())
                tblog('L_wae_mmd', wae_mmd_loss.item())
                tblog('L_dis_decoder', loss_dis.item())
                tblog('L_decoder', loss_decoder.item())
                tblog('L_encoder', loss_encoder.item())

        if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
            tqdm.write(
                'ITER {} TRAINING (phase 1). loss_discriminator: {:.4f}; loss_decoder: {:.4f}; loss_encoder: {:.4f}; '
                'wae_mmd_loss: {:.4f}; loss_recon_logits: {:.4f}; loss_recon_feature: {:.4f}; '
                'Grad_norm_Dis: {:.4e};  Grad_norm_De: {:.4e};  Grad_norm_En: {:.4e}'
                    .format(it, loss_discriminator.item(), loss_decoder.item(), loss_encoder.item(), 
                    wae_mmd_loss.item(), loss_recon_logits.item(), loss_recon_feature.item(), grad_norm_Dis.item(),
                    grad_norm_De.item(), grad_norm_En.item()))

            log_sent, _ = model.generate_sentences(1, sample_mode='categorical')
            tqdm.write('Sample (cat T=1.0): "{}"'.format(dataset.idx2sentence(log_sent.squeeze())))
            sys.stdout.flush()
            
            if it % cfgv.expsvlog_every == 0 and it > 0:
                def decode_from_z(z, model, dataset):
                    sall = []
                    for zchunk in torch.split(z, 1000):
                        s, _ = model.generate_sentences(zchunk.size(0),
                                           zchunk,
                                           sample_mode='categorical')
                        sall += s
                    return dataset.idx2sentences(sall, print_special_tokens=False)
                save_model(model, cfgv.chkpt_path.format(it))
                samples_z = model.sample_z_prior(cfg.evals.sample_size)
                samples = decode_from_z(samples_z, model, dataset)
                df = pd.DataFrame({'peptide': samples,'z': [tuple(z.tolist()) for z in samples_z]})
                df = df[~df['peptide'].isin([''])]
                df = compute_modlamp(df)
                df.to_csv(cfg.savepath+'/samples_{}.csv'.format(it),sep=',')

