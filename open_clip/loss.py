import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_only_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features



def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            use_noun_labels=None,
            tag_mode='mixed',
            use_longcap=False,
            use_multipos_loss=False,
            use_ot_loss=False,
            use_finegrained_loss=False,
            use_declip_loss=False,
            gamma=32,
            margin=0.25,
            sgl_delta=0.1,
            sgl_norm='minmax',
            clip_lossweight=1.0,
            sgl_lossweight=1.0
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.use_noun_labels = use_noun_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.prev_num_logits_multicap = 0
        self.labels_multicap = {}
        self.prev_num_logits_ot = 0
        self.labels_ot = {}
        if self.use_noun_labels:
            self.tagging_loss_function = AsymmetricLoss(gamma_neg=7,
                                                        gamma_pos=0,
                                                        clip=0.05)
        self.tag_mode = tag_mode
        self.max_iter = 100
        self.eps = 0.1
        self.use_multipos_loss = use_multipos_loss
        if self.use_multipos_loss:
            self.mpl = proxy_anchor_loss() #AsymmetricLoss(gamma_neg=7, gamma_pos=3, clip=0.05)
        self.use_ot_loss = use_ot_loss
        self.use_finegrained_loss = use_finegrained_loss
        self.use_declip_loss = use_declip_loss

        self.gamma = gamma
        self.margin = margin

        self.sgl_norm = sgl_norm
        self.sgl_delta = sgl_delta

        self.clip_lossweight = clip_lossweight
        self.sgl_lossweight = sgl_lossweight
        
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    
    def get_ground_truth_ot(self, device, num_logits, M, N) -> torch.Tensor:  
        # calculated ground-truth and cache if enabled
        # num_logits: 256 
        if device not in self.labels_ot:
            labels = torch.eye(num_logits, device=device, dtype=torch.long).repeat(M,N)
            if self.cache_labels:
                self.labels_ot[device] = labels
                self.prev_num_logits_ot = num_logits
        else:
            labels = self.labels_ot[device]
        return labels

    def get_ground_truth_multicap(self, device, num_logits, M, N) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        # num_logits: 256 
        if device not in self.labels_multicap:
            labels = torch.eye(num_logits, device=device, dtype=torch.long).repeat(M,N)
            if self.cache_labels:
                self.labels_multicap[device] = labels
                self.prev_num_logits_multicap = num_logits
        else:
            labels = self.labels_multicap[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad,
                self.rank, self.world_size, self.use_horovod)

            if len(all_text_features.shape)==3:
                bs, N_t, c = all_text_features.shape
                all_text_features = all_text_features.permute(1, 0, 2).reshape(N_t * bs, c)
            if len(all_image_features.shape)==3:
                bs,  N_i, c = all_image_features.shape
                all_image_features = all_image_features.permute(1, 0, 2).reshape(N_i * bs, c)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            text_features = all_text_features
            image_features = all_image_features
        else:
            if len(text_features.shape)==3:
                bs, N_t, c = text_features.shape
                text_features = text_features.permute(1, 0, 2).reshape(N_t * bs, c)
            if len(image_features.shape)==3:
                bs, N_i, c = image_features.shape
                image_features = image_features.permute(1, 0, 2).reshape(N_i * bs, c)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text, text_features, image_features

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-3
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def pairwise_contrastive_loss_instance(self, a, b, device, logit_scale=1.0):
        batch_size, seq_len, _ = a.shape
        labels = torch.eye(seq_len, device=device, dtype=torch.float).repeat(batch_size, 1)
        logits = torch.einsum('bmd,bnd->bmn', a, b) * logit_scale
        loss = F.cross_entropy(logits.view(-1, seq_len), labels) 
        return loss

    def pairwise_contrastive_loss(self, a, b, device, logit_scale=1.0):
        batch_size, seq_len, c = a.shape
        labels = torch.eye(seq_len*batch_size, device=device, dtype=torch.float)#.repeat(batch_size, 1)
        # logits = torch.einsum('bd,bd->bmn', a, b) * logit_scale
        text_features = a.contiguous().view(seq_len*batch_size, c)
        image_features = b.contiguous().view(seq_len*batch_size, c)

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        loss = 1/2 * (F.cross_entropy(logits_per_text, labels) + F.cross_entropy(logits_per_image, labels))
        return loss

    def pairwise_circleloss(
            self,
            dist_mat,
            is_pos,
            margin=0.25,
            gamma=64):

        N = dist_mat.size(0)
        is_neg = 1.0-is_pos
        
        # Mask scores related to itself
        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
        delta_p = 1 - margin
        delta_n = margin

        logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

    def forward(self, image_features, text_features, logit_scale, logit_scale_local, local_image_features=None, output_dict=False, noun_labels=None, classess_embs=None):
        device = image_features.device
        out_losses = {}
        if text_features is not None and len(text_features.shape) ==3:
            bs, number_text, c = text_features.shape
            if image_features.shape[0]!=text_features.shape[0]:
                image_features = image_features.view(bs, -1, c).contiguous()
                number_img = image_features.shape[1]
            else:
                number_img = 1
                image_features = image_features.unsqueeze(1)
            
            if local_image_features is not None:
                local_image_features = local_image_features.contiguous().view(text_features.shape[0], number_img, -1, c)
            global_losses = []

            all_text_features_list=[]
            all_img_features_list=[]
            for j in range(number_img):
                if self.world_size > 1:
                    all_img_features_list.append(gather_only_features( 
                            image_features[:,j,:],
                            False, self.gather_with_grad,
                            self.rank, self.world_size, self.use_horovod))
                else:
                    all_img_features_list.append(image_features[:,j,:])
            for i in range(number_text):
                if self.world_size > 1:
                    all_text_features_list.append(gather_only_features( 
                            text_features[:,i,:],
                            False, self.gather_with_grad,
                            self.rank, self.world_size, self.use_horovod))
                else:
                    all_text_features_list.append(text_features[:,i,:])
            
            logits_per_image = logit_scale * all_img_features_list[0] @ all_text_features_list[0].T
            logits_per_text = logit_scale * all_text_features_list[0] @  all_img_features_list[0].T
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            out_losses['CLIP_loss'] = self.clip_lossweight * (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / (2*number_text)
            if self.use_declip_loss:
                for j in range(number_img):
                    for i in range(number_text):
                        if j==0 and i==0:continue
                        logits_per_image = logit_scale * all_img_features_list[j] @ all_text_features_list[i].T
                        logits_per_text = logit_scale * all_text_features_list[i] @ all_img_features_list[j].T
                        global_losses.append((F.cross_entropy(logits_per_image, labels) +
                                              F.cross_entropy(logits_per_text, labels)) / (2*number_text))
                out_losses['(De)CLIP_loss'] = self.clip_lossweight * sum(global_losses)

            all_text_features_list = torch.stack(all_text_features_list, dim=0).view(-1,c)
            if self.use_multipos_loss:
                labels_mul = self.get_ground_truth_multicap(device, 
                                    all_img_features_list[0].shape[0], number_text, 1)
                # all_img_features_list = torch.stack(all_img_features_list, dim=0).view(-1,c)
                all_img_features_list = all_img_features_list[0]
                logits_per_image =  all_img_features_list @ all_text_features_list.T
                logits_per_text = logits_per_image.T
                
                # out_losses['T2mI_loss'] = self.pairwise_circleloss(logits_per_text, labels_mul, margin = self.margin, gamma = self.gamma) / number_text
                # # labels_mul = labels_mul / labels_mul.sum(1, keepdim=True).clamp(min=1.0)
                # # out_losses['I2mT_loss'] = self.mpl(logits_per_text,
                # #             labels_mul / labels_mul.sum(1, keepdim=True).clamp(min=1.0)) * bs
                # out_losses['T2mI_loss'] = self.mpl(logits_per_image, labels_mul.T) / bs

                out_losses['I2mT_loss'] = self.pairwise_circleloss(logits_per_image, labels_mul.T, margin = self.margin, gamma = self.gamma) / 10.0
            if self.use_finegrained_loss:
                if self.world_size > 1:
                    local_image_features = gather_only_features(
                        local_image_features,
                        False, self.gather_with_grad,
                        self.rank, self.world_size, self.use_horovod)[:,0:1,:,:]
                bs, num_i, M, c = local_image_features.shape
                b = bs * num_i
                local_image_features = local_image_features.permute(1, 0, 2, 3).contiguous().view(b, M, c)
                all_text_features = all_text_features_list.contiguous().view(bs, number_text, c)

                local_image_features = F.normalize(local_image_features, p=2, dim=-1)
                all_text_features = F.normalize(all_text_features, p=2, dim=-1)
                similarity = torch.einsum('btd,bpd->btp', all_text_features, local_image_features)
                min_val = torch.min(similarity, dim=-1, keepdim=True).values
                max_val = torch.max(similarity, dim=-1, keepdim=True).values
                epsilon = 1e-10
                if self.sgl_norm=='minmax':
                    normalized_similarity = (similarity - min_val) / (max_val - min_val + epsilon)
                else:
                    normalized_similarity = F.softmax(similarity, dim=-1) #(similarity - min_val) / (max_val - min_val + epsilon)
                if self.sgl_delta<0:
                    similarity_threshold = 1./local_image_features.shape[1]
                else:
                    similarity_threshold = self.sgl_delta
                normalized_similarity = torch.where(normalized_similarity < similarity_threshold, torch.tensor(0.0, device=normalized_similarity.device), normalized_similarity)
                sum_similarity = torch.sum(normalized_similarity, dim=-1, keepdim=True)
                v_align_weights = normalized_similarity / sum_similarity
                l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, local_image_features)
                l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)

                # loss_vl_local = self.pairwise_contrastive_loss(l_grouped_v_patch_embed, all_text_features, device)
                # loss_lv_local = self.pairwise_contrastive_loss(all_text_features, l_grouped_v_patch_embed, device)
                # out_losses['finegrained_loss'] = (loss_vl_local+loss_lv_local)/2.
                out_losses['finegrained_loss'] = self.sgl_lossweight * self.pairwise_contrastive_loss(all_text_features, l_grouped_v_patch_embed, device, logit_scale_local)

            if self.use_ot_loss:

                if self.world_size > 1:
                    local_image_features = gather_only_features(
                        local_image_features,
                        False, self.gather_with_grad,
                        self.rank, self.world_size, self.use_horovod)[:,0:1,:,:]
                bs, num_i, M, c = local_image_features.shape
                b = bs * num_i
                local_image_features = local_image_features.permute(1, 0, 2, 3).contiguous().view(b, M, c)
                all_text_features = all_text_features.contiguous().view(bs, number_text, c)

                sim = torch.einsum('mbd,ncd->mnbc', local_image_features.permute(1,0,2), all_text_features.permute(1,0,2)).contiguous()  
                sim = sim.view(M,number_text,b*bs)
                sim = sim.permute(2,0,1)
                wdist = 1.0 - sim
                xx=torch.zeros(b*bs, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
                yy=torch.zeros(b*bs, number_text, dtype=sim.dtype, device=sim.device).fill_(1. / number_text)
                with torch.no_grad():
                    KK = torch.exp(-wdist / self.eps)
                    T = self.Sinkhorn(KK,xx,yy)
                try:
                    torch.isnan(T).any()
                except None:
                    print('There is none value in your tensor, please try to adjust #thre and #eps to align data.')

                sim_op = torch.sum(T * sim, dim=(1, 2))
                sim_op = logit_scale * sim_op.contiguous().view(b,bs)
                out_losses['ot_local_loss'] = F.cross_entropy(sim_op, self.get_ground_truth_ot(device, b, 1, 1).float())
        
            return out_losses
        else:
            logits_per_image, logits_per_text,_ ,_ = self.get_logits(image_features,
                                                                text_features,
                                                                logit_scale)
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            total_loss = (F.cross_entropy(logits_per_image, labels) +
                          F.cross_entropy(logits_per_text, labels)) / 2
        

        if self.use_noun_labels:
            mixed_tag_loss = self.tagging_loss_function(classess_embs, noun_labels)/1000.0
            return {"contrastive_loss": total_loss, f'{self.tag_mode}_tag_loss':mixed_tag_loss}
        else:
            return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


# Tagging loss function
# copy from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

class proxy_anchor_loss(nn.Module):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    def __init__(self, scale=16, margin=0.1):
        super(proxy_anchor_loss, self).__init__()
        self.alpha = scale
        self.delta = margin

    def forward(self, dist_mat, target):
        
        
        pos_target = target.float()
        neg_target = 1.0 - pos_target
        
        pos_mat = torch.exp(-self.alpha * (dist_mat - self.delta)) * pos_target
        neg_mat = torch.exp(self.alpha * (dist_mat + self.delta)) * neg_target

        pos_term = 1.0 / torch.unique(target).shape[0] * torch.sum(torch.log(1.0 + torch.sum(pos_mat, axis=0)))
        neg_term = 1.0 / dist_mat.shape[0] * torch.sum(torch.log(1.0 + torch.sum(neg_mat, axis=0)))

        loss = pos_term + neg_term

        return loss