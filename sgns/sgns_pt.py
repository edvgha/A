import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import numpy as np
import random

# ==========================================
# 1. Toy Data & Vocabulary Preparation
# ==========================================
corpus = """
the quick brown fox jumps over the lazy dog
the smart fox runs away from the lazy dog
dogs are very smart and quick animals
foxes are smart and quick
"""

# Tokenize and build vocabulary
words = corpus.lower().split()
word_counts = Counter(words)
vocab = list(word_counts.keys())
vocab_size = len(vocab)

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# Convert corpus to indices
data = [word_to_idx[w] for w in words]

# ==========================================
# 2. Negative Sampling Distribution (Pn(w))
# ==========================================
# Word2Vec authors found that raising raw frequencies to the 3/4 power 
# works best for negative sampling to slightly boost rare words.
word_freqs = np.array([word_counts[w] for w in vocab], dtype=np.float32)
unigram_dist = word_freqs ** 0.75
unigram_dist = unigram_dist / unigram_dist.sum() # Normalize to create probabilities

# Convert to a PyTorch tensor for efficient sampling
unigram_tensor = torch.FloatTensor(unigram_dist)

# ==========================================
# 3. Generating Training Pairs
# ==========================================
def generate_training_data(data, window_size=2):
    """
    Generates (center_word, context_word) pairs using a sliding window.
    """
    pairs = []
    for i in range(len(data)):
        center = data[i]
        # Get context window boundaries
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        
        for j in range(start, end):
            if i != j: # Don't pair the center word with itself
                context = data[j]
                pairs.append((center, context))
    return pairs

training_pairs = generate_training_data(data)
print(f"Generated {len(training_pairs)} positive training pairs.")

# ==========================================
# 4. PyTorch SGNS Model
# ==========================================
class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNSModel, self).__init__()
        
        # Matrix V: Embeddings for Center Words
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Matrix U: Embeddings for Context / Negative Words
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize weights randomly (small values)
        init_range = 0.5 / embedding_dim
        self.v_embeddings.weight.data.uniform_(-init_range, init_range)
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, center_idx, context_idx, negative_idx):
        """
        center_idx:   [batch_size] 
        context_idx:  [batch_size]
        negative_idx: [batch_size, K] (K is number of negative samples)
        """
        # Look up embeddings
        v_c = self.v_embeddings(center_idx)    # Shape: [batch_size, emb_dim]
        u_o = self.u_embeddings(context_idx)   # Shape: [batch_size, emb_dim]
        u_k = self.u_embeddings(negative_idx)  # Shape: [batch_size, K, emb_dim]
        
        # 1. Positive Loss: maximize log(sigmoid(v_c * u_o))
        # Element-wise multiplication, then sum over the embedding dimension (dot product)
        pos_score = torch.sum(v_c * u_o, dim=1) # Shape: [batch_size]
        pos_loss = -F.logsigmoid(pos_score)     # Shape: [batch_size]
        
        # 2. Negative Loss: maximize sum(log(sigmoid(-v_c * u_k)))
        # Reshape v_c to [batch_size, emb_dim, 1] for batch matrix multiplication
        v_c_unsqueezed = v_c.unsqueeze(2) 
        # bmm multiplies [batch, K, emb_dim] by [batch, emb_dim, 1] to get [batch, K, 1]
        neg_score = torch.bmm(u_k, v_c_unsqueezed).squeeze(2) # Shape: [batch_size, K]
        # Take log sigmoid of the NEGATIVE dot products, sum over all K samples
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)       # Shape: [batch_size]
        
        # Total loss is the average over the batch
        return (pos_loss + neg_loss).mean()
        
    def get_embedding(self, word_idx):
        """Extract the learned center word embedding"""
        return self.v_embeddings.weight[word_idx].detach()

# ==========================================
# 5. Training Loop
# ==========================================
EMBEDDING_DIM = 10
BATCH_SIZE = 8
K_NEGATIVES = 5
EPOCHS = 200
LEARNING_RATE = 0.05

model = SGNSModel(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    total_loss = 0
    random.shuffle(training_pairs)
    
    # Process in batches
    for i in range(0, len(training_pairs), BATCH_SIZE):
        batch_pairs = training_pairs[i:i+BATCH_SIZE]
        
        center_batch = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long)
        context_batch = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long)
        
        # Draw K negative samples per training pair based on our unigram distribution
        current_batch_size = len(batch_pairs)
        # torch.multinomial is highly efficient for drawing from a probability distribution
        negative_batch = torch.multinomial(unigram_tensor, current_batch_size * K_NEGATIVES, replacement=True)
        negative_batch = negative_batch.view(current_batch_size, K_NEGATIVES)
        
        # Zero gradients, forward pass, backward pass, step
        optimizer.zero_grad()
        loss = model(center_batch, context_batch, negative_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 40 == 0:
        print(f"Epoch: {epoch+1}/{EPOCHS} | Average Loss: {total_loss / len(training_pairs):.4f}")

# ==========================================
# 6. Extracting and Testing the Embeddings
# ==========================================
print("\n--- Training Complete ---")

def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

def find_closest_words(target_word, top_n=3):
    if target_word not in word_to_idx:
        return "Word not in vocabulary."
        
    target_idx = word_to_idx[target_word]
    target_vector = model.get_embedding(target_idx)
    
    similarities = []
    for word, idx in word_to_idx.items():
        if word == target_word:
            continue
        vector = model.get_embedding(idx)
        sim = cosine_similarity(target_vector, vector).item()
        similarities.append((word, sim))
        
    # Sort by descending similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Test it out (keep in mind this is a tiny toy dataset, so results are limited!)
print("\nClosest words to 'fox':")
for word, sim in find_closest_words("fox"):
    print(f" - {word} (Similarity: {sim:.4f})")

print("\nClosest words to 'dog':")
for word, sim in find_closest_words("dog"):
    print(f" - {word} (Similarity: {sim:.4f})")