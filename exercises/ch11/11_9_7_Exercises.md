# 1. Is it possible to fine-tune T5 using a minibatch consisting of different tasks? Why or why not? How about for GPT-2?

It is possible to fine-tune T5 using a minibatch consisting of different tasks, as long as the tasks share the same input and output format, and the model can distinguish the tasks based on some prefix or label. For example, T5 can be fine-tuned on multiple text-to-text tasks, such as summarization, translation, question answering, etc., by adding a task-specific prefix to the input, such as "summarize:", "translate English to French:", or "answer:". This way, the model can learn to perform different tasks depending on the prefix, and leverage the shared knowledge and vocabulary across tasks ¹.

For GPT-2, it is also possible to fine-tune it using a minibatch consisting of different tasks, but it may be more challenging than for T5. This is because GPT-2 is an auto-regressive language model that only takes the input text as the context and generates the output text token by token, without any explicit separation between input and output. Therefore, it may be harder for GPT-2 to infer the task and the output format from the input text alone, especially if the tasks are diverse and complex. One possible solution is to use a special token or delimiter to mark the end of the input and the beginning of the output, such as "

- (1) Title: EncT5: A Framework for Fine-tuning T5 as Non .... https://arxiv.org/abs/2110.08426.
- (2) Can we fine-tune T5 for multiple tasks? - Hugging Face Forums. https://discuss.huggingface.co/t/can-we-fine-tune-t5-for-multiple-tasks/30268.
- (3) A Full Guide to Finetuning T5 for Text2Text and Building a .... https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887.
- (4) Fine-tuning GPT-2 from human preferences - OpenAI. https://openai.com/research/fine-tuning-gpt-2.
- (5) GitHub - steffen74/GPT-2: Fine-Tuning the GPT-2 with .... https://github.com/steffen74/GPT-2.
- (6) Fine tuning GPT2 with Hugging Face and Habana Gaudi. https://developer.habana.ai/blog/fine-tuning-gpt2-with-hugging-face-and-habana-gaudi/.
- (7) undefined. https://doi.org/10.48550/arXiv.2110.08426.

# 2. Given a powerful language model, what applications can you think of?

A powerful language model can enable many applications that require natural language understanding and generation, such as:

- Text summarization: This is the task of generating a concise and informative summary of a longer text, such as a news article, a research paper, or a book. A powerful language model can learn to extract the main points and key information from the input text, and rephrase them in a coherent and fluent way. For example, some language models can produce abstractive summaries that use novel words and phrases that are not in the original text .
- Machine translation: This is the task of translating a text from one language to another, while preserving the meaning and style of the original text. A powerful language model can learn to map the words and sentences from the source language to the target language, and handle the grammatical, syntactic, and semantic differences between languages. For example, some language models can perform zero-shot translation, which means they can translate between languages that they have not seen during training .
- Question answering: This is the task of answering a natural language question based on a given context, such as a passage, a document, or a knowledge base. A powerful language model can learn to comprehend the context and the question, and generate a relevant and accurate answer. For example, some language models can answer open-domain questions, which means they can retrieve information from a large corpus of documents without any predefined domain or schema .
- Text generation: This is the task of generating natural language text based on some input, such as a prompt, a keyword, or an image. A powerful language model can learn to produce coherent and diverse texts that match the input and the desired output format. For example, some language models can generate poems, stories, code, essays, songs, celebrity parodies, and more using their own words and knowledge  .

Some other possible applications of a powerful language model are:

- Sentiment analysis: This is the task of detecting and classifying the emotional tone or attitude of a text, such as positive, negative, or neutral. A powerful language model can learn to recognize the subtle cues and expressions that convey the sentiment of the text, and assign a label or a score accordingly. For example, some language models can perform fine-grained sentiment analysis, which means they can identify different aspects or attributes of the text and their corresponding sentiments .
- Text classification: This is the task of assigning one or more categories or labels to a text based on its content or purpose, such as topic, genre, authorship, or spam detection. A powerful language model can learn to extract the relevant features and keywords from the text, and match them with the predefined categories or labels. For example, some language models can perform multi-label text classification, which means they can assign multiple labels to a text that belongs to more than one category .
- Text simplification: This is the task of rewriting a text in a simpler and easier way, while preserving its meaning and information. A powerful language model can learn to reduce the complexity and difficulty of the text by using simpler words and sentences, removing unnecessary details or redundancies, and adding explanations or clarifications. For example, some language models can perform adaptive text simplification, which means they can adjust the level of simplification according to the needs and preferences of the target audience .

# 3. Say that you are asked to fine-tune a language model to perform text classification by adding additional layers. Where will you add them? Why?

One possible way to fine-tune a language model to perform text classification by adding additional layers is to add them on top of the language model, after the last hidden layer. This way, the additional layers can leverage the rich contextual representations learned by the language model, and adapt them to the specific task and data. For example, one common additional layer is a linear layer that maps the hidden state of the language model to the output logits, which can be used to predict the class label of the input text. Another possible additional layer is a pooling layer that aggregates the hidden states of the language model over the sequence dimension, which can provide a more compact and global representation of the input text. For example, some pooling methods are max pooling, mean pooling, or attention pooling ¹.


- (1) Universal Language Model Fine-tuning for Text Classification. https://arxiv.org/abs/1801.06146.
- (2) [1905.05583] How to Fine-Tune BERT for Text Classification .... https://arxiv.org/abs/1905.05583.
- (3) arXiv:1801.06146v5 [cs.CL] 23 May 2018. https://arxiv.org/pdf/1801.06146.pdf.
- (4) undefined. https://doi.org/10.48550/arXiv.1801.06146.
- (5) undefined. https://doi.org/10.48550/arXiv.1905.05583.

# 4. Consider sequence-to-sequence problems (e.g., machine translation) where the input sequence is always available throughout the target sequence prediction. What could be limitations of modeling with decoder-only Transformers? Why?

Some possible limitations of modeling with decoder-only Transformers for sequence-to-sequence problems are:

- Decoder-only Transformers may not be able to fully utilize the information in the input sequence, as they only rely on the encoder-decoder attention mechanism to access the input tokens. This may limit the ability of the model to capture the long-term dependencies and semantic relationships between the input and output sequences, especially if they are very long or complex. Encoder-decoder Transformers, on the other hand, can encode the input sequence into a latent representation using an encoder network, which can provide a more compact and global view of the input sequence for the decoder network.
- Decoder-only Transformers may suffer from exposure bias, which means that they may generate inconsistent or unrealistic output sequences due to the discrepancy between training and inference. During training, decoder-only Transformers use the ground-truth output tokens as inputs for the next time step, while during inference, they use their own predictions as inputs. This may cause the model to deviate from the true data distribution and produce errors that accumulate over time. Encoder-decoder Transformers can mitigate this problem by using teacher forcing, which means that they randomly replace some of the decoder inputs with the ground-truth output tokens during training, to make the model more robust to its own errors.
- Decoder-only Transformers may have difficulty in handling multiple tasks or domains, as they have to share the same decoder network for different types of input and output sequences. This may cause interference or confusion for the model, as it has to learn different vocabularies, grammars, and styles for different tasks or domains. Encoder-decoder Transformers can overcome this challenge by using task-specific or domain-specific encoder or decoder networks, which can adapt to different input and output formats and requirements. For example, T5 ¹ uses a text-to-text framework that can handle multiple natural language processing tasks by adding a task-specific prefix to the input sequence, such as "translate English to French:" or "summarize:".


- (1) SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers. https://arxiv.org/abs/2212.10325.
- (2) SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers. https://arxiv.org/pdf/2212.10325.pdf.
- (3) NLP From Scratch: Translation with a Sequence to Sequence .... https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
- (4) undefined. https://doi.org/10.48550/arXiv.2212.10325.
