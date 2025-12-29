import torch
import torch.nn as nn
from .gatv2_conv import GATv2Conv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    """
        Graph Attention Network (GAT) module with pruning and non-pruning branches.

        This class implements a GAT model that processes graph-structured data using two parallel GATv2Conv layers
        (pruned and non-pruned) for feature extraction, followed by linear transformations. It takes node features and
        edge indices (including a pruned edge index) as input and produces two output representations.

        Args:
            hidden_dims (List[int]): List of dimensions [in_dim, num_hidden, out_dim], where:
                - in_dim: Input feature dimension.
                - num_hidden: Hidden feature dimension for GAT layers.
                - out_dim: Output feature dimension for intermediate representation.
    """
    def __init__(self, hidden_dims):
        """
            Initializes the GAT module.

            Args:
                hidden_dims (List[int]): List of dimensions [in_dim, num_hidden, out_dim], where:
                    - in_dim: Input feature dimension.
                    - num_hidden: Hidden feature dimension for GAT layers.
                    - out_dim: Output feature dimension for intermediate representation.

            Attributes:
                conv1 (GATv2Conv): First non-pruned GATv2 layer.
                conv1_p (GATv2Conv): First pruned GATv2 layer.
                conv2 (torch.nn.Linear): Linear layer to transform concatenated features.
                conv3 (GATv2Conv): Second non-pruned GATv2 layer.
                conv3_p (GATv2Conv): Second pruned GATv2 layer.
                conv4 (torch.nn.Linear): Final linear layer to reconstruct input dimension.

            Example:
                >>> import torch
                >>> net = GAT(hidden_dims=[3000, 64, 16])
        """
        super(GAT, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv1_p = GATv2Conv(in_dim, num_hidden, heads=3, concat=False,
                                 dropout=0.2, add_self_loops=False, bias=True)

        self.conv2 = torch.nn.Linear(num_hidden * 2, out_dim, bias=True)

        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv3_p = GATv2Conv(out_dim, num_hidden, heads=3, concat=False,
                                 dropout=0.2, add_self_loops=False, bias=True)

        self.conv4 = torch.nn.Linear(num_hidden * 2, in_dim, bias=True)

    def forward(self, features, edge_index, edge_index_cluster):
        """
            Forward pass of the GAT module.

            Processes input node features through two parallel GATv2Conv branches (pruned and non-pruned), concatenates
            the results, and applies linear transformations to produce intermediate and final representations.

            Args:
                features (torch.Tensor): Input node features, shape (num_nodes, in_dim).
                edge_index (torch.Tensor): Edge indices for the non-pruned graph, shape (2, num_edges).
                edge_index_cluster (torch.Tensor): Edge indices for the pruned graph, shape (2, num_edges_cluster).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - h2: Intermediate representation after the second layer, shape (num_nodes, out_dim).
                    - h4: Final reconstructed representation, shape (num_nodes, in_dim).

            Example:
                >>> import torch
                >>> net = GAT(hidden_dims=[3000, 64, 16])
                >>> features = torch.randn(100, 3000)
                >>> edge_index = torch.randint(0, 100, (2, 200))
                >>> edge_index_cluster = torch.randint(0, 100, (2, 100))
                >>> h2, h4 = net(features, edge_index, edge_index_cluster)
        """

        h1_nonprune = self.conv1(features, edge_index)
        h1_prune = self.conv1_p(features, edge_index_cluster)
        h1 = F.elu(torch.cat([h1_nonprune, h1_prune], dim=1))

        h2 = self.conv2(h1)

        h3_nonprune = self.conv3(h2, edge_index)
        h3_prune = self.conv3_p(h2, edge_index_cluster)
        h3 = F.elu(torch.cat([h3_nonprune, h3_prune], dim=1))

        h4 = self.conv4(h3)

        return h2, h4


class GAT_noncluster(torch.nn.Module):
    """
        Graph Attention Network (GAT) module without pruning.

        This class implements a simplified GAT model using GATv2Conv layers without a pruning branch. It processes
        graph-structured data with a single set of edge indices and produces two output representations.

        Args:
            hidden_dims (List[int]): List of dimensions [in_dim, num_hidden, out_dim], where:
                - in_dim: Input feature dimension.
                - num_hidden: Hidden feature dimension for GAT layers (doubled internally).
                - out_dim: Output feature dimension for intermediate representation.
    """
    def __init__(self, hidden_dims):
        """
            Initializes the GAT_noncluster module.

            Args:
                hidden_dims (List[int]): List of dimensions [in_dim, num_hidden, out_dim], where:
                    - in_dim: Input feature dimension.
                    - num_hidden: Hidden feature dimension for GAT layers (doubled internally).
                    - out_dim: Output feature dimension for intermediate representation.

            Attributes:
                conv1 (GATv2Conv): First GATv2 layer.
                conv2 (torch.nn.Linear): Linear layer for intermediate representation.
                conv3 (GATv2Conv): Second GATv2 layer.
                conv4 (torch.nn.Linear): Final linear layer to reconstruct input dimension.

            Example:
                >>> import torch
                >>> net = GAT_noncluster(hidden_dims=[3000, 64, 16])
        """
        super(GAT_noncluster, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        num_hidden *= 2
        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv2 = torch.nn.Linear(num_hidden, out_dim, bias=True)
        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv4 = torch.nn.Linear(num_hidden, in_dim, bias=True)

    def forward(self, features, edge_index):
        """
            Forward pass of the GAT_noncluster module.

            Processes input node features through a sequence of GATv2Conv and linear layers to produce intermediate and
            final representations.

            Args:
                features (torch.Tensor): Input node features, shape (num_nodes, in_dim).
                edge_index (torch.Tensor): Edge indices for the graph, shape (2, num_edges).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - h2: Intermediate representation after the second layer, shape (num_nodes, out_dim).
                    - h4: Final reconstructed representation, shape (num_nodes, in_dim).

            Example:
                >>> import torch
                >>> net = GAT_noncluster(hidden_dims=[3000, 64, 16])
                >>> features = torch.randn(100, 3000)
                >>> edge_index = torch.randint(0, 100, (2, 200))
                >>> h2, h4 = net(features, edge_index)
        """

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1)
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = self.conv4(h3)
        return h2, h4


class CrossAttention(nn.Module):
    """
        Cross-attention module for combining two feature representations.

        This module computes attention weights between two feature sets (h1 and h2) using a learnable attention mechanism
        and combines them into a single representation.

        Args:
            dim (int): Dimension of the input feature vectors.
            dropout (float): Dropout probability, defaults to 0.0.
            alpha (float): Slope for the LeakyReLU activation, defaults to 0.2.
    """
    def __init__(self, dim, dropout=0.0, alpha=0.2):
        """
            Initializes the CrossAttention module.

            Args:
                dim (int): Dimension of the input feature vectors.
                dropout (float): Dropout probability, defaults to 0.0.
                alpha (float): Slope for the LeakyReLU activation, defaults to 0.2.

            Attributes:
                a1 (nn.Parameter): Learnable attention weights for the first feature, shape (2 * dim, 1).
                a2 (nn.Parameter): Learnable attention weights for the second feature, shape (2 * dim, 1).
                leakyrelu (nn.LeakyReLU): LeakyReLU activation function with the specified alpha.

            Example:
                >>> cross_attn = CrossAttention(dim=64, dropout=0.2, alpha=0.2)
        """
        super(CrossAttention, self).__init__()

        self.dropout = dropout
        self.NFeature = dim
        self.alpha = alpha

        self.a1 = nn.Parameter(torch.FloatTensor(2 * dim, 1))
        self.a2 = nn.Parameter(torch.FloatTensor(2 * dim, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def inference(self, h1=None, h2=None):
        """
            Forward pass of the CrossAttention module.

            Computes attention weights between two feature sets (h1 and h2) and combines them into a single representation.

            Args:
                h1 (torch.Tensor, optional): First feature tensor, shape (num_nodes, dim). Defaults to None.
                h2 (torch.Tensor, optional): Second feature tensor, shape (num_nodes, dim). Defaults to None.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - lamda: Attention weights, shape (num_nodes, 2).
                    - h: Combined feature representation, shape (num_nodes, dim).

            Example:
                >>> import torch
                >>> cross_attn = CrossAttention(dim=64, dropout=0.2)
                >>> h1 = torch.randn(100, 64)
                >>> h2 = torch.randn(100, 64)
                >>> lamda, h = cross_attn(h1, h2)
        """
        e = self._prepare_attentional_mechanism_input(h1, h2)
        lamda = F.softmax(e, dim=1)
        lamda = F.dropout(lamda, self.dropout, training=self.training)

        h_prime1 = lamda[:, 0].repeat(self.NFeature, 1).T * h1
        h_prime2 = lamda[:, 1].repeat(self.NFeature, 1).T * h2

        h = h_prime1 + h_prime2

        return lamda, h

    def forward(self, h1=None, h2=None):
        """
            Computes the attention mechanism and combines the feature representations.

            This method performs the core computation of the cross-attention mechanism, producing attention weights
            and a combined feature representation.

            Args:
                h1 (torch.Tensor, optional): First feature tensor, shape (num_nodes, dim). Defaults to None.
                h2 (torch.Tensor, optional): Second feature tensor, shape (num_nodes, dim). Defaults to None.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - lamda: Attention weights, shape (num_nodes, 2).
                    - h: Combined feature representation, shape (num_nodes, dim).
        """
        lamda, h = self.inference(h1, h2)

        return lamda, h

    def _prepare_attentional_mechanism_input(self, h1=None, h2=None):
        """
            Prepares the input for the attention mechanism.

            Computes the attention scores by applying linear transformations to the concatenated feature tensors.

            Args:
                h1 (torch.Tensor, optional): First feature tensor, shape (num_nodes, dim). Defaults to None.
                h2 (torch.Tensor, optional): Second feature tensor, shape (num_nodes, dim). Defaults to None.

            Returns:
                torch.Tensor: Attention scores after LeakyReLU activation, shape (num_nodes, 2).
        """
        Wh1 = torch.matmul(torch.cat([h1, h2], 1), self.a1)
        Wh2 = torch.matmul(torch.cat([h1, h2], 1), self.a2)
        # broadcast add
        e = torch.cat((Wh1, Wh2), 1)

        return self.leakyrelu(e)





class GraphAttentionLayer(nn.Module):
    """
        Graph Attention Layer for processing graph-structured data.

        This class implements a single graph attention layer, which applies attention mechanisms to node features
        based on an adjacency matrix. Supports tied weights and attention for shared parameter scenarios.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            dropout (float): Dropout probability.
            alpha (float): Slope for the LeakyReLU activation, defaults to 0.2.
            concat (bool): Whether to apply ELU activation to the output, defaults to True.
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        """
            Initializes the GraphAttentionLayer module.

            Args:
                in_features (int): Input feature dimension.
                out_features (int): Output feature dimension.
                dropout (float): Dropout probability.
                alpha (float): Slope for the LeakyReLU activation, defaults to 0.2.
                concat (bool): Whether to apply ELU activation to the output, defaults to True.

            Attributes:
                W (nn.Parameter): Learnable weight matrix, shape (in_features, out_features).
                a (nn.Parameter): Learnable attention weights, shape (2 * out_features, 1).
                leakyrelu (nn.LeakyReLU): LeakyReLU activation function with the specified alpha.

            Example:
                >>> attn_layer = GraphAttentionLayer(in_features=64, out_features=32, dropout=0.2)
        """
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, tied_W=None, tied_attention=None):
        """
            Forward pass of the GraphAttentionLayer module.

            Applies the attention mechanism to node features based on the adjacency matrix, optionally using tied weights
            or attention matrices.

            Args:
                h (torch.Tensor): Input node features, shape (num_nodes, in_features).
                adj (torch.Tensor): Adjacency matrix, shape (num_nodes, num_nodes).
                tied_W (torch.Tensor, optional): Tied weight matrix, shape (in_features, out_features). Defaults to None.
                tied_attention (torch.Tensor, optional): Tied attention matrix, shape (num_nodes, num_nodes). Defaults to None.

            Returns:
                torch.Tensor: Output features after attention, shape (num_nodes, out_features).

            Example:
                >>> import torch
                >>> attn_layer = GraphAttentionLayer(in_features=64, out_features=32, dropout=0.2)
                >>> h = torch.randn(100, 64)
                >>> adj = torch.ones(100, 100)
                >>> output = attn_layer(h, adj)
        """
        self.tied_W = tied_W
        self.tied_attention = tied_attention
        if self.tied_W is not None:
            # self.W = self.tied_W
            Wh = torch.mm(h, self.tied_W)
        Wh = torch.mm(h, self.W)
        self.W_res = self.W
        e = self._prepare_attentional_mechanism_input(Wh)

        if self.tied_attention is not None:
            attention = self.tied_attention
        else:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            # attention = F.dropout(attention, self.dropout, training=self.training)

        self.attention_res = attention
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
            Prepares the input for the attention mechanism.

            Computes attention scores by applying linear transformations to the transformed node features.

            Args:
                Wh (torch.Tensor): Transformed node features, shape (num_nodes, out_features).

            Returns:
                torch.Tensor: Attention scores after LeakyReLU activation, shape (num_nodes, num_nodes).
        """
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])# n,fout  * fout,1   n* 1
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])# n,fout  * fout,1   n* 1
        e = Wh1 + Wh2.T# n,1  + 1,n  n * n
        return self.leakyrelu(e)


if __name__ == '__main__':
    net = GAT(hidden_dims=[3000,64, 16])
    print(net)