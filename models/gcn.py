import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

import numpy as np

torch.set_printoptions(precision=2,threshold=float('inf'))

class AGCNBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim,gcn_layer=2,dropout=0.0,relu=0):
        super(AGCNBlock,self).__init__()
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.sort = 'sort'
        self.model='agcn'
        self.gcns=nn.ModuleList()
        self.bn = 0
        self.add_self = 1
        self.normalize_embedding = 1
        self.gcns.append(GCNBlock(input_dim,hidden_dim,self.bn,self.add_self,self.normalize_embedding,dropout,relu))
        self.pool = 'mean'
        self.tau = 1.
        self.lamda = 1.

        for i in range(gcn_layer-1):
            if i==gcn_layer-2 and (not 1):
                self.gcns.append(GCNBlock(hidden_dim,hidden_dim,self.bn,self.add_self,self.normalize_embedding,dropout,0))
            else:
                self.gcns.append(GCNBlock(hidden_dim,hidden_dim,self.bn,self.add_self,self.normalize_embedding,dropout,relu))
            
        if self.model=='diffpool':
            self.pool_gcns=nn.ModuleList()
            tmp=input_dim
            self.diffpool_k=200
            for i in range(3):
                self.pool_gcns.append(GCNBlock(tmp,200,0,0,0,dropout,relu))
                tmp=200

        self.w_a=nn.Parameter(torch.zeros(1,hidden_dim,1))
        self.w_b=nn.Parameter(torch.zeros(1,hidden_dim,1))
        torch.nn.init.normal_(self.w_a)
        torch.nn.init.uniform_(self.w_b,-1,1)

        self.pass_dim=hidden_dim

        if self.pool=='mean':
            self.pool=self.mean_pool
        elif self.pool=='max':
            self.pool=self.max_pool
        elif self.pool=='sum':
            self.pool=self.sum_pool

        self.softmax='global'
        if self.softmax=='gcn':
            self.att_gcn=GCNBlock(2,1,0,0,dropout,relu)
        self.khop=1
        self.adj_norm='none'

        self.filt_percent=0.25       #default 0.5
        self.eps=1e-10

        self.tau_config=1
        if 1==-1.:
            self.tau=nn.Parameter(torch.tensor(1),requires_grad=False)
        elif 1==-2.:
            self.tau_fc=nn.Linear(hidden_dim,1)
            torch.nn.init.constant_(self.tau_fc.bias,1)
            torch.nn.init.xavier_normal_(self.tau_fc.weight.t())
        else:
            self.tau=nn.Parameter(torch.tensor(self.tau))
        self.lamda1=nn.Parameter(torch.tensor(self.lamda))
        self.lamda2=nn.Parameter(torch.tensor(self.lamda))

        self.att_norm=0
        
        self.dnorm=0
        self.dnorm_coe=1

        self.att_out=0
        self.single_att=0


    def forward(self,X,adj,mask,is_print=False):
        '''
    input:
        X:  node input features , [batch,node_num,input_dim],dtype=float
        adj: adj matrix, [batch,node_num,node_num], dtype=float
        mask: mask for nodes, [batch,node_num]
    outputs:
        out:unormalized classification prob, [batch,hidden_dim]
        H: batch of node hidden features, [batch,node_num,pass_dim]
        new_adj: pooled new adj matrix, [batch, k_max, k_max]
        new_mask: [batch, k_max]
        '''
        hidden=X
        #adj = adj.float()
        # print('input size:')
        # print(hidden.shape)
        
        is_print1=is_print2=is_print
        if adj.shape[-1]>100:
            is_print1=False

        for gcn in self.gcns:
            hidden=gcn(hidden,adj,mask)
        #     print('gcn:')
        #     print(hidden.shape)
        # print('mask:')
        # print(mask.unsqueeze(2).shape)
        # print(mask.sum(dim=1))

        hidden=mask.unsqueeze(2)*hidden
        # print(hidden[0][0])
        # print(hidden[0][-1])

        if self.model=='unet':
            att=torch.matmul(hidden,self.w_a).squeeze()
            att=att/torch.sqrt((self.w_a.squeeze(2)**2).sum(dim=1,keepdim=True))
        elif self.model=='agcn':
            if self.softmax=='global' or self.softmax=='mix':
                if False:
                    dgree_w = torch.sum(adj, dim=2) / torch.sum(adj, dim=2).max(1, keepdim=True)[0]
                    att_a=torch.matmul(hidden,self.w_a).squeeze()*dgree_w+(mask-1)*1e10
                else:
                    att_a=torch.matmul(hidden,self.w_a).squeeze()+(mask-1)*1e10
                    # print(att_a[0][:10])
                    # print(att_a[0][-10:-1])
                att_a_1=att_a=torch.nn.functional.softmax(att_a,dim=1)
                # print(att_a[0][:10])
                # print(att_a[0][-10:-1])

                if self.dnorm:
                    scale=mask.sum(dim=1,keepdim=True)/self.dnorm_coe
                    att_a=scale*att_a
            if self.softmax=='neibor' or self.softmax=='mix':
                att_b=torch.matmul(hidden,self.w_b).squeeze()+(mask-1)*1e10
                att_b_max,_=att_b.max(dim=1,keepdim=True)
                if self.tau_config!=-2:
                    att_b=torch.exp((att_b-att_b_max)*torch.abs(self.tau))
                else:
                    att_b=torch.exp((att_b-att_b_max)*torch.abs(self.tau_fc(self.pool(hidden,mask))))
                denom=att_b.unsqueeze(2)
                for _ in range(self.khop):
                    denom=torch.matmul(adj,denom)
                denom=denom.squeeze()+self.eps
                att_b=(att_b*torch.diagonal(adj,0,1,2))/denom
                if self.dnorm:
                    if self.adj_norm=='diag':
                        diag_scale=mask/(torch.diagonal(adj,0,1,2)+self.eps)
                    elif self.adj_norm=='none':
                        diag_scale=adj.sum(dim=1)
                    att_b=att_b*diag_scale
                att_b=att_b*mask
                        
            if self.softmax=='global':
                att=att_a
            elif self.softmax=='neibor' or self.softmax=='hardnei':
                att=att_b
            elif self.softmax=='mix':
                att=att_a*torch.abs(self.lamda1)+att_b*torch.abs(self.lamda2)
        # print('att:')
        # print(att.shape)
        Z=hidden

        if self.model=='unet':
            Z=torch.tanh(att.unsqueeze(2))*Z
        elif self.model=='agcn':
            if self.single_att:
                Z=Z
            else:
                Z=att.unsqueeze(2)*Z
        # print('Z shape')
        # print(Z.shape)
        k_max=int(math.ceil(self.filt_percent*adj.shape[-1]))
        # print('k_max')
        # print(k_max)
        if self.model=='diffpool':
            k_max=min(k_max,self.diffpool_k)

        k_list=[int(math.ceil(self.filt_percent*x)) for x in mask.sum(dim=1).tolist()]
        # print('k_list')
        # print(k_list)
        if self.model!='diffpool': 
            if self.sort=='sample':
                att_samp = att * mask
                att_samp = (att_samp/att_samp.sum(1)).detach().cpu().numpy()
                top_index = ()
                for i in range(att.size(0)):
                    top_index = (torch.LongTensor(np.random.choice(att_samp.size(1), k_max, att_samp[i])) ,)
                top_index = torch.stack(top_index,1)
            elif self.sort=='random_sample':
                top_index = torch.LongTensor(att.size(0), k_max)*0
                for i in range(att.size(0)):
                    top_index[i,0:k_list[i]] = torch.randperm(int(mask[i].sum().item()))[0:k_list[i]] 
            else: #sort
                _,top_index=torch.topk(att,k_max,dim=1)
        # print('top_index')
        # print(top_index)
        # print(len(top_index[0]))
        new_mask=X.new_zeros(X.shape[0],k_max)
        # print('new_mask')
        # print(new_mask.shape)
        visualize_tools=None 
        if self.model=='unet':
            for i,k in enumerate(k_list):
                for j in range(int(k),k_max):
                    top_index[i][j]=adj.shape[-1]-1
                    new_mask[i][j]=-1.
            new_mask=new_mask+1
            top_index,_=torch.sort(top_index,dim=1)
            assign_m=X.new_zeros(X.shape[0],k_max,adj.shape[-1])
            for i,x in enumerate(top_index):
                assign_m[i]=torch.index_select(adj[i],0,x)
            new_adj=X.new_zeros(X.shape[0],k_max,k_max)
            H=Z.new_zeros(Z.shape[0],k_max,Z.shape[-1])
            for i,x in enumerate(top_index):
                new_adj[i]=torch.index_select(assign_m[i],1,x)
                H[i]=torch.index_select(Z[i],0,x)

        elif self.model=='agcn':
            assign_m=X.new_zeros(X.shape[0],k_max,adj.shape[-1])
            # print('assign_m.shape')
            # print(assign_m.shape)
            for i,k in enumerate(k_list):
                #print('top_index[i][j]')
                for j in range(int(k)):  
                    #print(str(top_index[i][j].item())+' ', end='')
                    assign_m[i][j]=adj[i][top_index[i][j]]
                    #print(assign_m[i][j])
                    new_mask[i][j]=1.

            assign_m=assign_m/(assign_m.sum(dim=1,keepdim=True)+self.eps)
            H=torch.matmul(assign_m,Z)
            # print('H')
            # print(H.shape)
            new_adj=torch.matmul(torch.matmul(assign_m,adj),torch.transpose(assign_m,1,2))
            # print(torch.matmul(assign_m,adj).shape)
            # print('new_adj:')
            # print(new_adj.shape)
            
        elif self.model=='diffpool':
            hidden1=X
            for gcn in self.pool_gcns:
                hidden1=gcn(hidden1,adj,mask)
            assign_m=X.new_ones(X.shape[0],X.shape[1],k_max)*(-100000000.)
            for i,x in enumerate(hidden1):
                k=min(k_list[i],k_max)
                assign_m[i,:,0:k]=hidden1[i,:,0:k]
                for j in range(int(k)):
                    new_mask[i][j]=1.

            assign_m=torch.nn.functional.softmax(assign_m,dim=2)*mask.unsqueeze(2)
            assign_m_t=torch.transpose(assign_m,1,2)
            new_adj=torch.matmul(torch.matmul(assign_m_t,adj),assign_m)
            H=torch.matmul(assign_m_t,Z)
        # print('pool')    
        if self.att_out and self.model=='agcn':
            if self.softmax=='global':
                out=self.pool(att_a_1.unsqueeze(2)*hidden,mask)
            elif self.softmax=='neibor':
                att_b_sum=att_b.sum(dim=1,keepdim=True)
                out=self.pool((att_b/(att_b_sum+self.eps)).unsqueeze(2)*hidden,mask)
        else:
            # print('hidden.shape')
            # print(hidden.shape)
            out=self.pool(hidden,mask)
            # print('out shape')
            # print(out.shape)
           
        if self.adj_norm=='tanh' or self.adj_norm=='mix':
            new_adj=torch.tanh(new_adj)
        elif self.adj_norm=='diag' or self.adj_norm=='mix':
            diag_elem=torch.pow(new_adj.sum(dim=2)+self.eps,-0.5)
            diag=new_adj.new_zeros(new_adj.shape)
            for i,x in enumerate(diag_elem):
                diag[i]=torch.diagflat(x)
            new_adj=torch.matmul(torch.matmul(diag,new_adj),diag)

        visualize_tools=[]
        '''
        if (not self.training) and is_print1:
            print('**********************************')
            print('node_feat:',X.type(),X.shape)
            print(X)
            if self.model!='diffpool':
                print('**********************************')
                print('att:',att.type(),att.shape)
                print(att)
                print('**********************************')
                print('top_index:',top_index.type(),top_index.shape)
                print(top_index)
            print('**********************************')
            print('adj:',adj.type(),adj.shape)
            print(adj)
            print('**********************************')
            print('assign_m:',assign_m.type(),assign_m.shape)
            print(assign_m)
            print('**********************************')
            print('new_adj:',new_adj.type(),new_adj.shape)
            print(new_adj)
            print('**********************************')
            print('new_mask:',new_mask.type(),new_mask.shape)
            print(new_mask)
        '''
        #visualization
        from os import path
        if not path.exists('att_1.pt'):
            torch.save(att[0], 'att_1.pt')
            torch.save(top_index[0], 'att_ind1.pt')
        elif not path.exists('att_2.pt'):
            torch.save(att[0], 'att_2.pt')
            torch.save(top_index[0], 'att_ind2.pt')
        else:
            torch.save(att[0], 'att_3.pt')
            torch.save(top_index[0], 'att_ind3.pt')

        if (not self.training) and is_print2:
            if self.model!='diffpool':
                visualize_tools.append(att[0])
                visualize_tools.append(top_index[0])
            visualize_tools.append(new_adj[0])
            visualize_tools.append(new_mask.sum())
        # print('**********************************')
        return out,H,new_adj,new_mask,visualize_tools
    
    def mean_pool(self,x,mask):
        return x.sum(dim=1)/(self.eps+mask.sum(dim=1,keepdim=True))
    
    def sum_pool(self,x,mask):
        return x.sum(dim=1)

    @staticmethod
    def max_pool(x,mask):
        #output: [batch,x.shape[2]]
        m=(mask-1)*1e10
        r,_=(x+m.unsqueeze(2)).max(dim=1)
        return r
# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0,add_self=0, normalize_embedding=0,
            dropout=0.0,relu=0, bias=True):
        super(GCNBlock,self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu=relu
        self.bn=bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.bn:
            index=mask.sum(dim=1).long().tolist()
            bn_tensor_bf=mask.new_zeros((sum(index),y.shape[2]))
            bn_tensor_af=mask.new_zeros(*y.shape)
            start_index=[]
            ssum=0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum+=index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i+1]]=y[i,0:index[i]]
            bn_tensor_bf=self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i,0:index[i]]=bn_tensor_bf[start_index[i]:start_index[i+1]]
            y=bn_tensor_af
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu=='relu':
            y=torch.nn.functional.relu(y)
            print('hahah')
        elif self.relu=='lrelu':
            y=torch.nn.functional.leaky_relu(y,0.1)
        return y

#experimental function, untested
class masked_batchnorm(nn.Module):
    def __init__(self,feat_dim,epsilon=1e-10):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones(feat_dim))
        self.beta=nn.Parameter(torch.zeros(feat_dim))
        self.eps=epsilon

    def forward(self,x,mask):
        '''
        x: node feat, [batch,node_num,feat_dim]
        mask: [batch,node_num]
        '''
        mask1 = mask.unsqueeze(2)
        mask_sum = mask.sum()
        mean = x.sum(dim=(0,1),keepdim=True)/(self.eps+mask_sum)
        temp = (x - mean)**2
        temp = temp*mask1
        var = temp.sum(dim=(0,1),keepdim=True)/(self.eps+mask_sum)
        rstd = torch.rsqrt(var+self.eps)
        x=(x-mean)*rstd
        return ((x*self.alpha) + self.beta)*mask1