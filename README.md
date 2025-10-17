This project takes all works of Shakespeare as a training input and trains on it. Later it produces new sentences based on a given seed.

Number of parameters:
Token Embedding : 19,200
Positional Embedding : 12,288
Total of Embedding = 31,488

Single self-attention block per layer: 36,864
Feed forward block per layer : 73,728
Total per layer : 110,976

Number of layers = 6
Total for 6 layers : 6*110,976 = 665,856

Final lm layer = 19,200

Total = 31,488 + 665,856 + 19,200 = 716,544 parameters





