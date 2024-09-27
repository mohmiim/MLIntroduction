# What are LLM models and how they are designed and trained

In this session, we will cover the concept of LLM models and how they are designed and trained. I am not planning to cover Prompt engineering in this session.

LLM stands for Large Language Model, and it is a model that is trained to predict the next word in a sentence. The model is trained on a large corpus of text, and it is designed to be able to generate text that is similar to the text it was trained on.

The large amount of training data, and the large number of parameters in the model allow it to discover relationships from the text that are not explicitly stated in the training data. This allows the model to generate text that is coherent and relevant to the input text. The training process
is both unsupervised and semi supervised.

Large language model uses exciting very well know techniques in deeplearning (and we covered some of them in previous sessions) like Neural networks, Transformer architecture and so pn

## How large is large

We can not underestimate the L in LLM models, the size of the model is what makes it powerful, and it is what makes it hard to train. The model has billions of parameters, and it requires a lot of computational resources to train. But also require powerful GPU and lots of memory at inference time

here are some examples of some LLM models sizes

<table style="width:100%; border: 1px solid black;">
  <thead>
    <tr>
      <th style="text-align:left;">Model</th>
      <th style="text-align:right;">Number of Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left;">GPT 3</td>
      <td style="text-align:right;">175 Billion</td>
    </tr>
    <tr>
      <td style="text-align:left;">LLama 3.2 </td>
      <td style="text-align:right;">up to 90 Billion</td>
    </tr>
  </tbody>
</table>

## Transformer architecture

To understand  the inner works of LLM models, we need to understand the transformer architecture, and how it is used in LLM models. The transformer
architecture is a type of neural network that is designed to process sequences of data. It is made up of two main components: the encoder and the decoder. 
The encoder takes the input sequence and processes it to create a representation of the input data. The decoder takes the representation created by the 
encoder and uses it to generate the output sequence. It all started with the amazing paper "Attention is all you need" by Vaswani et al. in 2017. here 
is a link for it [Attention is all you need](https://arxiv.org/abs/1706.03762)

In this Session we will delve into the transformer architecture and how it is used in LLM models.
