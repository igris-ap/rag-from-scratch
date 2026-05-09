# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with language model generation. Instead of relying
solely on the model's trained knowledge, RAG fetches relevant documents
at query time and uses them as context for the answer.

This makes the system more accurate, more up-to-date, and more
transparent — you can always point to the source document that
produced a given answer.

## How Retrieval Works

The retrieval step converts both documents and queries into vector
embeddings — numerical representations of meaning. Documents that are
semantically similar to the query will have embeddings that are close
in vector space, measured by cosine similarity.

A vector database stores these embeddings and can find the top-K most
similar documents to any query in milliseconds, even across millions
of documents.

## How Generation Works

Once relevant chunks are retrieved, they are inserted into the LLM
prompt as context. The model is instructed to answer using only the
provided context, which grounds the response in real source material
and reduces hallucination.

## Parent-Child Chunking

A key design decision in production RAG systems is chunk size.
Small chunks give precise retrieval — the embedding captures a narrow,
specific meaning. Large chunks give the LLM enough context to write
a coherent, complete answer.

The parent-child strategy solves this by storing two versions of every
section: a small child chunk for retrieval, and a large parent chunk
for generation. When a child chunk matches a query, we fetch its parent
to get the full context.
