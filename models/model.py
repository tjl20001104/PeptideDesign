from models.classifier import build_classifier
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from itertools import chain


from models.decoder import build_decoder
from models.encoder import build_encoder
from models.discriminator import build_discriminator
from models.classifier import build_classifier
from models.Beam import Beam
from models.mutils import UNK_IDX, PAD_IDX, START_IDX, EOS_IDX
from models.mutils import soft_embed, onehot_embed


class RNN_WAE(nn.Module):
    """
    1. Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
    2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015).
    3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self,
                 n_vocab,
                 max_seq_len,
                 z_dim,
                 emb_dim,
                 pretrained_emb,
                 freeze_embeddings,
                 E_args,
                 G_args,
                 D_args,
                 C_args
                 ):
        super(RNN_WAE, self).__init__()
        self.MAX_SEQ_LEN = max_seq_len
        self.n_vocab = n_vocab
        self.z_dim = z_dim
        self.device = torch.device('cuda')

        """
        Word embeddings layer
        """
        self.emb_dim = emb_dim
        self.word_emb = nn.Embedding(n_vocab, self.emb_dim, PAD_IDX)
        if pretrained_emb is not None:
            assert self.emb_dim == pretrained_emb.size(1), 'emb dim dont match with pretrained'
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, PAD_IDX)
            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_emb)
        if freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        '''
        Initialize all the modules
        '''
        self.encoder = build_encoder('gru',
                                     emb_dim=self.emb_dim,
                                     z_dim=z_dim,
                                     **E_args)
        self.decoder = build_decoder(embedding=self.word_emb,
                                     emb_dim=self.emb_dim + z_dim,
                                     output_dim=n_vocab,
                                     h_dim=z_dim,
                                     **G_args)
        self.discriminator = build_discriminator(input_dim=self.emb_dim,
                                     **D_args)


    def decoder_params(self):
        return filter(lambda p: p.requires_grad, self.decoder.parameters())


    def encoder_params(self):
        params = [self.word_emb.parameters(),
                  self.encoder.parameters()]
        return filter(lambda p: p.requires_grad, chain(*params))


    def generator_params(self):
        params = [self.word_emb.parameters(),
                  self.encoder.parameters(),
                  self.decoder.parameters()]
        return filter(lambda p: p.requires_grad, chain(*params))


    def discriminator_params(self):
        params = [self.word_emb.parameters(),
                  self.discriminator.parameters()]
        return filter(lambda p: p.requires_grad, chain(*params))
    

    def wae_params(self):
        params = [self.word_emb.parameters(),
                  self.encoder.parameters(),
                  self.decoder.parameters(),
                  self.discriminator.parameters()]
        return filter(lambda p: p.requires_grad, chain(*params))


    def forward_encoder(self, inputs, input_lens):
        '''
        Inputs is batch of sentences: seq_len x mbsize
               or batch of soft sentences: seq_len x mbsize x n_vocab.
        '''
        if inputs.dim() == 2:
            inputs = self.word_emb(inputs)
        else:  # dim == 3.
            inputs = soft_embed(self.word_emb, inputs)

        input_lens_sorted, indices = input_lens.sort(descending=True)
        input_lens_sorted = input_lens_sorted.to(self.device)
        indices = indices.to(self.device)
        inputs_sorted = inputs.index_select(0, indices)
        inputs_sorted = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        mu, logvar = self.encoder(inputs_sorted)
        _, inv_indices = indices.sort()
        mu = mu.index_select(0, inv_indices)
        logvar = logvar.index_select(0, inv_indices)
        return mu, logvar


    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = torch.randn(mu.size(0), self.z_dim).to(self.device)
        return mu + torch.exp(logvar / 2) * eps


    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = torch.randn(mbsize, self.z_dim).to(self.device)
        return z


    def forward_decoder(self, inputs, z):
        """
        Inputs are indices: seq_len x mbsize
        """
        return self.decoder(inputs, z)


    def forward_discriminator(self, inputs):
        """
        Inputs is batch of sentences: mbsize x seq_len
               or batch of soft sentences: mbsize x seq_len x n_vocab.
        """
        if inputs.dim() == 2:
            inputs = self.word_emb(inputs)
        else:  # dim == 3.
            inputs = soft_embed(self.word_emb, inputs)
        return self.discriminator(inputs)


    def forward(self, sequences, input_lens,sample_z=1):
        """
        Forwards a batch of sequences through encoder and decoder (with teacher forcing).

        Arguments
        ----
        sequences: indices in vocab (mbsize x seq_len)
        q_c: prior | classifier | tensor with ground truth c's.
        sample_z: 'max' / number: sample N times (mbsize x N)

        Returns:
        --------
        (mu, logvar) - from encoder. [mbsize x z_dim]
        (z, c)       - input to decoder. [mbsize x z_dim] [mbsize x c_dim]
        dec_logits   - [mbsize x seq_len x vocabsize]
        """
        # both encoder inputs and decoder inputs are plain sentence
        mbsize = sequences.size(0)

        # Encoder: sequences -> z
        mu, logvar = self.forward_encoder(sequences, input_lens)
        assert mu.size(0) == logvar.size(0) == mbsize
        if sample_z == 'max':
            z = mu
        else:
            assert sample_z == 1, 'TODO deal with sample_z > 1 do multiple samples, add another dimension?'
            z = self.sample_z(mu, logvar)

        # if isinstance(q_c, torch.Tensor):
        #     # Convert idx (float) to labels (long tensor)
        #     labels = q_c.unsqueeze(1)
        #     c = torch.zeros(mbsize, 2).to(self.device)
        #     c.scatter_(1, labels, 1)
        # elif q_c == 'prior':
        #     c = self.sample_c_prior(mbsize)
        # elif q_c == 'classifier':
        #     c = self.forward_classifier(sequences)
        #     c = F.softmax(c, dim=1)  # mbsize x 2
        # else:
        #     raise ValueError("q_c is not labels, prior, or classifier")
        # TODO future with multiple sif isinstance(q_c, torch.Tensor):
        #     # Convert idx (float) to labels (long tensor)
        #     labels = q_c.unsqueeze(1)
        #     c = torch.zeros(mbsize, 2).to(self.device)
        #     c.scatter_(1, labels, 1)
        # elif q_c == 'prior':
        #     c = self.sample_c_prior(mbsize)
        # elif q_c == 'classifier':
        #     c = self.forward_classifier(sequences)
        #     c = F.softmax(c, dim=1)  # mbsize x 2
        # else:
        #     raise ValueError("q_c is not labels, prior, or classifier")tructured code c; allow some codes to be present, some filled in from D/prior.

        # Decoder: (x,z) -> y
        dec_logits = self.forward_decoder(sequences, z)
        dec_logits_gau = self.forward_decoder(sequences, self.sample_z_prior(mbsize))
        return (mu, logvar), z, (dec_logits,dec_logits_gau)


    def generate_sentences(self, mbsize, z=None, eval_mode=True, **sample_kwargs):
        """
        Generate sentences of (mbsize x max_seq_len) when hard sampling,
                           or (mbsize x max_seq_len x emb_dim) when soft sampling.
        If z and c not specified: sampled from prior.
        If soft sampling (specify with sample_mode), eval_mode has to be False.小胡同学

        kwargs get passed into sample_G: temperatures etc.
        return sentences, z, c
        """
        if z is None:
            z = self.sample_z_prior(mbsize)

        if eval_mode:
            self.eval()

        sentences = self.sample_G(mbsize, z, **sample_kwargs)
        if eval_mode:
            self.train()
        return sentences, z


    def sample_G(self, mbsize, z,
                 sample_mode='categorical',
                 temp=1.0,
                 gumbel_temp=1.0,
                 prepend_start_idx=True,
                 prevent_empty=True,
                 min_length=1,
                 beam_size=5,
                 n_best=3):
        """
        This function samples a minibatch of mbsize from the decoder, given a z input.
        sample_mode determines hard sampling (categorical / greedy / gumbel_max) vs soft (gumbel_soft, gumbel_ST, XX_softmax)
        prepend_start_idx will prepend dummy <start> token, matches dataloader format.
        prevent_empty will modify the probabilities before hard sampling from them.
        min_length will not modify sampling, but just have at least this length output even if it's just all padding.
        """
        sample_mode_soft = sample_mode in ['gumbel_soft', 'gumbel_ST', 'greedy_softmax', 'categorical_softmax',
                                           'none_softmax']
        assert not (
                sample_mode_soft and prevent_empty), 'cant prevent_empty when soft sampling, ' \
                                                     'we dont wanna modify softmax in place before feeding back into ' \
                                                     'next timestep '
        assert beam_size >= n_best, "Can't return more than max hypothesis"
        assert mbsize == z.size(0), 'oops sizes dont match {} {}'.format(mbsize, z.size(0))

        # Collecting sampled sequences - Note: does not work for beam search
        seqIx = []
        seqSoftIx = []

        # to mask out after EOS
        finished = torch.zeros(mbsize, dtype=torch.bool).to(self.device)

        if sample_mode == 'beam':
            def unbottle(m):
                return m.view(beam_size, mbsize, -1)

            # Repeat inputs beam_size times
            z = z.repeat(beam_size, 1)
            # Initialize Beams
            beam = [Beam(beam_size,
                         n_best=n_best,
                         device=self.device,
                         pad=PAD_IDX,
                         bos=START_IDX,
                         eos=EOS_IDX,
                         min_length=min_length)
                    for ___ in range(mbsize)]
            # Start: first beam BOS, rest PAD.
            sampleIx = torch.stack([b.get_current_state() for b in beam]) \
                .t().contiguous().view(-1)
        else:
            # Start: all BOS.
            sampleIx = torch.LongTensor(mbsize).to(self.device).fill_(START_IDX)
        sampleSoftIx = None

        # RNN state
        h = self.decoder.init_hidden(z)  # [mbsize x z]
        h = h.unsqueeze(0)  # prepend 1 = num_layers * num_directions

        # seqLogProbs = [] # unused for now
        # collecting sampled logprobs would be basis for all policy gradient algos (seqGAN etc)

        # include start_idx in the output
        if prepend_start_idx:
            seqIx.append(sampleIx)
            if sample_mode_soft:
                seqSoftIx.append(onehot_embed(sampleIx, self.n_vocab).detach())

        for i in range(self.MAX_SEQ_LEN):
            # 1) FORWARD PASS THIS TIMESTEP
            logits, h = self.decoder.forward_sample(sampleSoftIx, sampleIx, z, h)
            # END TODO use forward_decoder()
            if prevent_empty and i == 0:
                # kinda hacky: force first char to be real character by  masking out the logits corresponding to
                # pad/start/eos.
                large_neg = -2 * torch.abs(
                    logits.min())  # dont wanna throw off downstream softmaxes by just putting -inf
                for maskix in [PAD_IDX, START_IDX, EOS_IDX]:
                    logits[:, maskix] = large_neg

            # 2) GIVEN LOGITS, SAMPLE -> sampleIx, sampleLogProbs, sampleSoftIx
            if sample_mode == 'categorical':
                sampleIx = torch.distributions.Categorical(logits=logits / temp).sample()
            elif sample_mode == 'greedy':
                sampleIx = torch.argmax(logits, 1)
            elif sample_mode == 'gumbel_max':
                tmp = """hard decision, same as Categorical sampling."""
            elif sample_mode == 'beam':
                logits = unbottle(logits)
                # Update the beams
                for j, b in enumerate(beam):
                    if not b.done():
                        # b.advance(logits[:, j])
                        logprobs = F.log_softmax(logits[:, j], dim=1)
                        b.advance(logprobs)
                    # Update corresponding hidden states
                    # NOTE if not advanced, the hidden will be reset and sampleIx will remain.

                    self._update_hidden(h, j, b.get_current_origin(), beam_size)
                # Get the current predictions
                sampleIx = torch.stack([b.get_current_state() for b in beam]) \
                    .t().contiguous().view(-1)
            # ABOVE: HARD SAMPLING, BELOW: SOFT SAMPLING
            elif sample_mode == 'gumbel_soft':
                tmp = """keep the softmax as seqSoftIx, not straight through."""
            elif sample_mode == 'gumbel_ST':
                tmp = """sampleSoftIx are straight-through onehot(argmax(gumbel_softmax)) which will pass through 
                biased gradients """
            # below: sampleIx none/greedy/categorical. softmax for softIx. Return seqIx, seqSoftIx.
            # The hard sample mode matters for when we'll run into EOS and mask out all subsequent softmaxes.
            elif sample_mode == 'none_softmax':
                sampleSoftIx = F.softmax(logits / temp, dim=1)
            elif sample_mode == 'greedy_softmax':
                sampleIx = torch.argmax(logits, 1)
                sampleSoftIx = F.softmax(logits / temp, dim=1)
            elif sample_mode == 'categorical_softmax':
                sampleIx = torch.distributions.Categorical(logits=logits / temp).sample()
                sampleSoftIx = F.softmax(logits / temp, dim=1)
            else:
                raise Exception('Sample mode {} not implemented.'.format(sample_mode))

            # 3) FINISHED SENTENCES: MASK OUT sampleIx, sampleLogProbs, sampleSoftIx
            # Not in beam-search: implemented inside of Beam.py
            if not sample_mode == "beam":
                sampleIx.masked_fill_(finished, PAD_IDX)  # (mask, value)
                finished[sampleIx == EOS_IDX] = True  # new EOS reached, mask out in the future.
                seqIx.append(sampleIx)

                if sample_mode_soft:
                    sampleSoftIx = sampleSoftIx.masked_fill(finished.unsqueeze(1).clone(), 0)
                    # set "one-hots" to 0, will embed to 0 vector. Note not exactly the same as sampleIx=0 which will
                    # map to embedweight[0,:]
                    seqSoftIx.append(sampleSoftIx)

            # 4) UPDATE MASK FOR NEXT ITERATION; BREAK (if all done)
            if finished.sum() == mbsize and len(seqIx) >= min_length:
                break  # everyone is done
            if sample_mode == "beam":
                if all((b.done() for b in beam)):
                    break
        if sample_mode == "beam":
            seqIx = []
            for b in beam:
                scores, ks = b.sort_finished(minimum=n_best)
                hyps = []
                for i, (times, k) in enumerate(ks[:n_best]):
                    hyp = b.get_hyp(times, k)
                    hyps.append(hyp)
                seqIx.append(hyps)
            return seqIx

        # End of loop. Assemble seqIx, seqSoftIx into tensor.
        seqIx = torch.stack(seqIx, dim=1)  # bs x seqlen
        if sample_mode_soft:
            seqSoftIx = torch.stack(seqSoftIx, dim=1)  # bs x seqlen x vocab. Note seqlen dim is inserted in the middle.
            assert seqIx.size(1) == seqSoftIx.size(1), 'messup with prepending startIx?'
            return seqIx, seqSoftIx
        else:
            return seqIx  # only hard sampling.


    def _update_hidden(self, h, beamIx, origins, beam_size):
        """
        Updates the hidden states after a beam step.
        Origins is a tensor that contains indices WHERE a new beam index comes from,
        i.e. the third best beam was previously the second best.
        This function copies the hidden states around to match this.
        """
        sizes = h.size()
        br = sizes[1]
        # Select correct set of beams
        sent_states = h.view(sizes[0],
                             beam_size,
                             br // beam_size,
                             sizes[2])[:, :, beamIx]
        # Update the states

        sent_states.data.copy_(
            sent_states.data.index_select(1, origins))


