import torch.nn as nn
import torch


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args
        self.model_name = args.kge
        self.nrelation = args.num_rel
        self.emb_dim = args.emb_dim
        self.epsilon = 2.0

        self.gamma = torch.Tensor([args.gamma])

        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.emb_dim])
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.args.rel_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.model_name not in ['TransE', 'DistMult', 'ComplEx', 'ConvE']:
            raise ValueError('model %s not supported' % self.model_name)

    def forward(self, sample, ent_emb, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        self.entity_embedding = ent_emb

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            if head_part != None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            if head_part == None:
                head = self.entity_embedding.unsqueeze(0)
            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            if tail_part != None:
                try:
                    batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                except IndexError:
                    print(tail_part)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part == None:
                tail = self.entity_embedding.unsqueeze(0)
            else:
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        elif mode == 'rel-batch':
            head_part, tail_part = sample
            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            if tail_part == None:
                relation = self.relation_embedding.unsqueeze(0)
            else:
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'ConvE': self.ConvE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score
    
    def ConvE(self, head, relation, tail, mode = None):
        self.emb_dim1 = self.args.emb_shape
        self.emb_dim2 = self.args.emb_dim // self.emb_dim1
        self.emb_ent = torch.nn.Embedding(self.args.num_ent, self.args.emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(self.args.num_rel, self.args.emb_dim, padding_idx=0)
        torch.nn.init.xavier_normal_(self.emb_ent.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)
        self.inp_drop = torch.nn.Dropout(self.args.inp_drop)
        self.hidden_drop = torch.nn.Dropout(self.args.hidden_drop)
        self.feg_drop = torch.nn.Dropout(self.args.feg_drop)
        self.conv1=torch.nn.Conv2d(1, 32, (3,3), 1, 0, bias = True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(self.args.emb)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.args.num_ent)))
        self.fc = torch.nn.Linear(self.args.hid_size,self.args.emb_dim)
        stacked_inputs = torch.cat([head, relation, tail], dim=2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feg_drop(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.mm(x, self.emb_ent.weight.transpose(1,0)) if mode == None \
            else torch.mm(x, mode.transpose(1,0))
        x += self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x


