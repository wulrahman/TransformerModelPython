import numpy as np

# Activation functions
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(x, 0)

def leaky_relu(x, alpha=0.01):  
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=1.0):  
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# Layer normalization function
def layer_norm(x, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std_dev = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(std_dev ** 2 + epsilon)

# Dropout function
def dropout(x, dropout_prob):
    mask = np.random.binomial(1, 1 - dropout_prob, size=x.shape)
    return x * mask / (1 - dropout_prob)

# Position-wise feed-forward network
class PositionwiseFeedforward:
    def __init__(self, input_dim, hidden_dim, activation_function=relu):  
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.weights_1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.weights_2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / hidden_dim)

    def forward(self, X):
        hidden_output = np.dot(X, self.weights_1)
        self.X = X
        self.hidden_output = self.activation_function(hidden_output) 
        output = np.dot(self.hidden_output, self.weights_2)
        return output

    def backward(self, grad_output):
        grad_hidden_output = np.dot(grad_output, self.weights_2.T)
        grad_hidden_output[self.hidden_output <= 0] = 0  
        grad_weights_2 = np.dot(self.hidden_output.T, grad_output)
        grad_weights_1 = np.dot(self.X.T, grad_hidden_output)
        return grad_hidden_output, grad_weights_1, grad_weights_2

# Sequential Transformer model
class SequentialTransformer:
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, num_heads=1, dropout_prob=0.1, activation_function=relu):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.activation_function = activation_function
        self.layers = []

        for _ in range(num_layers):
            self.layers.append(self._build_transformer_layer())

        self.final_layer_weights = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)

    def _build_transformer_layer(self):
        positional_encoding = np.zeros((self.input_dim, self.hidden_dim))
        for pos in range(self.input_dim):
            for i in range(0, self.hidden_dim, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/self.hidden_dim)))
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/self.hidden_dim)))

        return [
            np.random.randn(self.input_dim, self.hidden_dim) + positional_encoding,
            np.random.randn(self.input_dim, self.hidden_dim) + positional_encoding,
            np.random.randn(self.input_dim, self.hidden_dim) + positional_encoding,
            np.random.randn(self.num_heads * self.hidden_dim, self.input_dim) * np.sqrt(2 / self.input_dim),
            np.random.randn(self.num_heads * self.hidden_dim, self.hidden_dim) * np.sqrt(2 / self.hidden_dim), 
            PositionwiseFeedforward(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, activation_function=self.activation_function)
        ]

    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = K.shape[-1]
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)  

        attention_weights = softmax(scores)
        output = np.matmul(attention_weights, V)
        return output

    def forward(self, X, mask=None):
        for layer in self.layers:
            Q = np.dot(X, layer[0])
            K = np.dot(X, layer[1])
            V = np.dot(X, layer[2])

            Q_split = np.array_split(Q, self.num_heads, axis=-1)
            K_split = np.array_split(K, self.num_heads, axis=-1)
            V_split = np.array_split(V, self.num_heads, axis=-1)

            heads = []
            for i in range(self.num_heads):
                head = self._scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i], mask)
                heads.append(head)

            concatenated = np.concatenate(heads, axis=-1)
            multihead_output = np.dot(concatenated, layer[4])  
            multihead_output = layer_norm(multihead_output)  
            multihead_output = dropout(multihead_output, self.dropout_prob)  
            feed_forward_output = layer[5].forward(multihead_output)  
            X = layer_norm(feed_forward_output + multihead_output)  

        output = np.dot(X, self.final_layer_weights)
        return output

    def backward(self, X, output, y, learning_rate, max_gradient_norm=None):
        grad_loss = 2 * (output - y) / len(X)
        grad_output = grad_loss

        for layer in reversed(self.layers):
            if isinstance(layer[-1], PositionwiseFeedforward):
                grad_hidden_output, grad_weights_1, grad_weights_2 = layer[-1].backward(grad_output)
                grad_output = grad_hidden_output

                layer[-1].weights_1 -= learning_rate * grad_weights_1
                layer[-1].weights_2 -= learning_rate * grad_weights_2
            else:
                grad_output = grad_output.dot(layer[3].T) * (X > 0)
                grad_layer = np.dot(X.T, grad_output)
                if max_gradient_norm is not None:
                    grad_layer_norm = np.linalg.norm(grad_layer)
                    if grad_layer_norm > max_gradient_norm:
                        grad_layer *= max_gradient_norm / grad_layer_norm
                layer[3] -= learning_rate * grad_layer

        grad_final_layer = np.dot(relu(np.dot(X, self.final_layer_weights)).T, grad_loss)
        if max_gradient_norm is not None:
            grad_final_layer_norm = np.linalg.norm(grad_final_layer)
            if grad_final_layer_norm > max_gradient_norm:
                grad_final_layer *= max_gradient_norm / grad_final_layer_norm
        self.final_layer_weights -= learning_rate * grad_final_layer

# Training function with learning rate scheduling
def train(X_train, y_train, model, epochs, initial_learning_rate, decay_rate):
    for epoch in range(epochs):
        learning_rate = initial_learning_rate / (1 + decay_rate * epoch)
        
        output = model.forward(X_train)
        
        loss = np.mean((output - y_train) ** 2)
        print(f"Epoch {epoch + 1}, Loss: {loss}, Learning Rate: {learning_rate}")
        
        model.backward(X_train, output, y_train, learning_rate)


# Define vocabulary for input and output sentences
input_vocab = {'hello': 0, 'how': 1, 'are': 2, 'you': 3, 'good': 4, 'morning': 5}  # Example
output_vocab = {'hi': 0, 'i': 1, 'am': 2, 'fine': 3, 'thank': 4, 'you': 5}  # Example

# Example input and output sentences
X_sentences = ['hello how are you', 'good morning']  # Example
y_sentences = ['hi i am fine thank you', 'hi']  # Example

# Determine maximum sentence length
max_length = max(max(len(sentence.split()) for sentence in X_sentences),
                 max(len(sentence.split()) for sentence in y_sentences))

# Function to convert sentence to one-hot encoded vector
def sentence_to_one_hot(sentence, vocab, vocab_size):
    words = sentence.split()
    one_hot = np.zeros((max_length, vocab_size))
    for i, word in enumerate(words):
        if word in vocab:
            one_hot[i, vocab[word]] = 1
    return one_hot

# Convert X_sentences and y_sentences to one-hot encoded arrays
X_train = np.array([sentence_to_one_hot(s, input_vocab, len(input_vocab)) for s in X_sentences])
y_train = np.array([sentence_to_one_hot(s, output_vocab, len(output_vocab)) for s in y_sentences])

print("X_train shape:", X_train)
print("y_train shape:", y_train)


# Example usage
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[0, 1], [2, 3], [6, 7], [10, 11]])
    y_train = np.array([[2, 3], [4, 5], [8, 9], [12, 13]])

    # Define and train model
    model = SequentialTransformer(input_dim=2, output_dim=2, hidden_dim=2, num_layers=2, num_heads=1, dropout_prob=0.1)
    train(X_train, y_train, model, epochs=10000, initial_learning_rate=0.0001, decay_rate= 0.01)

    # Test the model
    test_input = np.array([[0, 1]])
    print("Test Input:", test_input)
    print("Predicted Output:", model.forward(test_input))
