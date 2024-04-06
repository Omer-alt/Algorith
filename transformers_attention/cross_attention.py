import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)



class CrossAttention:
    
    def __init__(self, x, x_hat):
        self.sentence = x
        self.sentence_2 = x_hat
        self.embedding_dim = 16
        self.dim_q, self.dim_k, self.dim_v = 24, 24, 28
        
        self.W_q = torch.rand(self.dim_q, self.embedding_dim)
        self.W_k = torch.rand(self.dim_k, self.embedding_dim)
        self.W_v = torch.rand(self.dim_v, self.embedding_dim)
        
        
        
    # To get Numeric representation of input sentence and it length
    def input_processing(self, sentence_input):

        sentence_array = sentence_input.lower().replace(',', '').split()
        d = len(sentence_array)
        dict_idx = {word: i for i, word in enumerate(sorted(sentence_array))}
        sentence_ints = torch.tensor([dict_idx[char] for char in sentence_array])

        print("Word dictionary :", dict_idx)
        print("Numeric representation of input sentence = ", sentence_ints)

        return sentence_ints, d
    
    def get_context(self):
        
        x1, n = self.input_processing(self.sentence)
        x2, n_hat = self.input_processing(self.sentence)
        
        # To encode the input
        embedding_layer_x = torch.nn.Embedding(n, self.embedding_dim)
        embedding_layer_x_hat = torch.nn.Embedding(n_hat, self.embedding_dim)
        embed_x1 = embedding_layer_x(x1).detach()
        embed_x2 = embedding_layer_x(x2).detach()
        
        # Compute Query, Key and Value
        queries =  embed_x1.matmul(self.W_q.T)
        keys = embed_x2.matmul(self.W_k.T)
        values = embed_x2.matmul(self.W_v.T)
        
        # Unnormalized attention scores
        omega = queries.matmul(keys.T)
        A = F.softmax(omega/(self.dim_k)**0.5, dim=1)
        Y = A.matmul(values)
        
        return Y
    
    
if __name__ == "__main__":
    
    x = "Life is too short, eat dessert first is life"
    x_hat = "Life is sweet, indulge in dessert first, for life is too short to skip the treats"
    cross_attention = CrossAttention(x, x_hat)
    Y = cross_attention.get_context()
    print("Our context matix has shape of :", Y.shape)
    
    self_attention = CrossAttention(x, x)
    y = self_attention.get_context()
    print("Our context matix for  has shape of :", y.shape)
    
    # Normaliser la matrice d'attention par softmax
    # attention_matrix = np.exp(attention_matrix) / np.sum(np.exp(attention_matrix), axis=1, keepdims=True)

    # Créer la heatmap .split()
    # plt.imshow(Y, cmap='hot', interpolation='nearest')
    
    # plt.xticks(range(len(x.replace(",", "").split())), x.replace(",", "").split())
    # plt.yticks(range(len(x_hat.replace(",", "").split())), x_hat.replace(",", "").split(), rotation=90)
    
    # plt.xlabel('Tokens de la chaîne x')
    # plt.ylabel('Tokens de la chaîne x_hat')
    # plt.title('Heatmap de la matrice d\'attention')
    # plt.colorbar()  # Ajouter une barre de couleur pour interpréter les valeurs
    # plt.show()

    
        
        
        
        
        
        
        
        
        

        
        
        
    












