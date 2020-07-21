EMBEDDING_DIM = 128
HIDDEN_DIM = 128 

reader = StanfordSentimentTreeBankDatasetReader()
train_dataset = reader.read('path/to/sst/dataset/train.txt')
dev_dataset = reader.read('path/to/sst/dataset/dev.txt')

vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})
token_embedding = Embedding( num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmClassifier(word_embeddings, encoder, vocab) 
optimizer = optim.Adam(model.parameters())

iterator = BucketIterator(batch_size=32,  sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model, optimizer=optimizer, iterator=iterator, train_dataset=train_dataset, validation_dataset=dev_dataset, patience=10, num_epochs=20)
trainer.train()