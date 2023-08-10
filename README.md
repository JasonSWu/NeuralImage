# NeuralImage
----------EncoderDecoder------------
The folder contains the files for a failed attempt at constructing an encoder-decoder network using pretrained
LLMs as the decoder. It quickly became clear that such an architecture is unviable for these purposes. Thus,
there is no reason to pay any mind to these files. The generic training loop is already contained within
finetune.py. The only file worth noting is data.py, which loads in and processes a large dataset from HuggingFace.
Guidance on using tokenizers to encode and truncate in batches can be found in the code.

------------chatbot.py--------------
This file implements a chatbot and deploys it on gradio. The chatbot's components are a vectorstore, memory
buffer, and google search summarizer. The memory buffer simply saves past inputs and responses in order then
appends or inserts the interactions into the prompt before it is passed to the LLM. The current implementation
in chatbot.py is that every 5 user inputs, the memory buffer is flushed into the vectorstore, where
each input-response pair is considered a document (future work could include finding a different document
schema). In addition, the search functionality will decide whether a user input requires specialized
information that should be searched via the Google API. If the input is worthy of a Google search, the code
will query Google's API and append a summary of the result to the prompt. Read the "search.py" section below
for more information.

------------finetune.py-------------
This file is for finetuning a pretrained model. It retrieves the data from poems.html. When running the file,
there are five arguments you must input: The number of epochs to be trained, learning rate, model file (put 
"None" if you don't have any weights you want to load in), optimizer1 params file name ("None" if none),
and optimizer2 params file name ("None" if none). Number of epochs to be trained is actually the number of 
epochs the model is trained on each of the raw prompt and chat prompt. In other words, it's actually doubled.
Manual modifications must be made to the code to switch models. Read through the comments to understan the code.

------------search.py---------------
This file contains the methods used by the chatbot to query google. A google custom search API key, google API
key, and OpenAI API key should be listed in a file named ".env" in the format
OPENAI_API_KEY=[key]
GOOGLE_CSE_ID=[key]
GOOGLE_API_KEY=[key]
The necessity of the OpenAI API key is subject to change, as the chatbot may migrate to the finetuned, 
commercially-usable open source models referred to above in finetune.py. The methods decide whether a search
is necessary/beneficial for a given input ("How was your day?" vs. "How do you perform a deadlift properly?"). 
If a search is necessary, it will query . Further work should be done tweaking the identification of whether a
search is necessary. In times of high uncertainty, not searching should be preferred, as this reduces run-time,
and it's natural for a person to not have knowledge of a complext topic. Additionally, searching when it is 
not necessary can return odd search results, which can override any creative responses by the model, 
resulting in nonsensical responses (For example, searching "How was you day?" on Google overrides what the
chatbot would've said with weird information from Google).
* search.py does not contain any main code; only methods used in chatbot.py are present.
* If one reads the code, one will notice that the results from Google's search API are truncated. This can be
removed after migrating to an open-source model, as this is mostly done to reduce the cost of querying OpenAI

-------------scraping---------------
This folder contains the code and results of scraping https://zh.wikisource.org/wiki/%E6%9D%B1%E5%9D%A1%E5%85%A8%E9%9B%86
for poems by Su Shi or Su DongPo. get_links.py retrieves all links to pages with poems andstores them in poem_urls.txt. 
wonky_volumes.txt contains the section of html in the original page that is oddly formatted and is not detected by 
get_links.py. All links in wonky_volums.txt have already been moved to poem_urls.txt manually. scraping.py goes through
poem_urls.txt and saves the titles and poems in poems.html. There are two different formats for the sublinks containing 
poems, as can be seen by the logic branching in scraping.py. 

* Sometimes, scraping.py will halt on random pages because requests are not going through. This has not been fixed. 
A POTENTIAL fix is doing a try-except block and looping until success.

------------Next steps-------------- 
Implement autoregressive finetuning: The current finetuning script implements teacher forcing. With autoregressive
finetuning, you would start with the input or prompt, then generate the next token. The new token is appended onto
the input to create a partially formed output. This partially formed output is passed back into the transformer to

Batching: One of the pros of batching is more stability in each training step. One of the cons is that the larger
the batch, the more memory the GPU needs to allocate to said batch. The best batch size varies between situations,
so experimentation after implementation is necessary (batching has yet to be implemented). One common strategy
which eliminates the con of more memory but maintains the benefit of stability is passing in smaller batch sizes
while accumulating the loss over multiple batches.

Improve chatbot memory: Currently, the chatbot is implemented with LangChain. It uses a FAISS vectorstore to query
for past input-response pairs that have the highest similarity. A few things that can be experimented with are using
a different vectorstore (e.g. Redis), storing documents in a different manner (maybe grouped according to topic?), the
number of documents queried, and where/how the results will fit into the prompt.