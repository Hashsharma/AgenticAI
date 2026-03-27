⚙️ How it works (step-by-step)

Split text into small units (usually sentences)

Convert each sentence into embeddings using a model (like OpenAI embeddings)

Measure similarity between adjacent sentences

Group sentences together until similarity drops

Start a new chunk when topic changes

👉 So chunks are based on topic boundaries, not size alone.


The Hybrid Approach (often “best of both worlds”)

Step 1: Semantic chunking → splits by topic
Step 2: Recursive splitter → enforce max chunk size

# Context-Aware Semantic Chunker

# Topic-Based Semantic Chunker

# Adaptive Threshold Chunker

# topic_shift_indicators

# Text Wrapping

# relevance_score

# Exponential backoff

# Jitter is a small amount of randomness added to delay times, usually in retry logic like exponential backoff.

