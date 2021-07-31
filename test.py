# Arquivo teste pra garantir isolamento.

import torch
from torch import nn
import gensim
from torchtext.legacy import data

class MultiLayerRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        # Camada do LSTM
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #Empacota a sequência (remove paddings) e somente processa isso. Porém, retorna empacotado também.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths
            #, enforce_sorted=False
            )
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #Então, desempacota
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)

# Função para ficar claro onde fica a parte de carregamento de estado
def prepare_model():
    def tokenize(sentence):
        return gensim.parsing.preprocessing.preprocess_string(sentence)

    TEXT = data.Field(tokenize = tokenize,
                      include_lengths = True)

    #from torchtext.legacy import datasets
    #import random
    #LABEL = data.LabelField(dtype = torch.float)
    #SEED = 789
    #torch.manual_seed(SEED)
    #torch.backends.cudnn.deterministic = True    
    #train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    #train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    #MAX_VOCAB_SIZE = 30000
    #TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
    #print(TEXT)
    #print(TEXT.vocab)
    #torch.save(TEXT.vocab, "vocab2_test.pt")

    #print(vars(list(train_data[0:10])))

    #print(train_data[0:10].numpy())

    
    #train_data.head()



    TEXT.vocab = torch.load("vocab2_test.pt")
    #print("Palavras únicas:", len(TEXT.vocab))
    #print("20 palavras mais comuns:", TEXT.vocab.freqs.most_common(20))
    #print("10 palavras do vocabulário de palavras para analisar estrutura:", TEXT.vocab.itos[:10])
    
    #torch.backends.cudnn.deterministic = True

    #torch.save(TEXT.vocab, "vocab2.pt")

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # Cria o modelo com LSTM e hiperparâmetros
    model = MultiLayerRNN(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)    
    model.load_state_dict(torch.load('tf-model2.pt'))
    model.eval()

    return model, TEXT

# Rotina independente de predição
def predict(text : str):
    # Isso poderia ser separado para performance
    model, TEXT = prepare_model()

    #tokens = TEXT.preprocess(text)
    #print(tokens)
    texts, lengths = TEXT.process([text])
    #texts = TEXT.preprocess(text)
    #lengths = TEXT.numericalize(texts)
    print(texts, lengths)
    with torch.no_grad():
        preds = model(texts, lengths).squeeze(1)

    return torch.sigmoid(preds)

tests = ['very good movie', 'not bad film', 'terrible acting', 
    "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked",
    "A wonderful little production"]
for t in tests:
    s = predict(t)
    print('Teste:', t, 'POS' if s.round() else 'NEG', '[', s, ']')
