
# TheHitchikersGPT
Are you curious about training a smaller custom GPT for your own use? Maybe train it on your own writing? <br />
Or do you want to experiment with OpenAI's Whisper speech-to-text model, to train a language model from your speeches or audiobooks? <br />
Or maybe you're just a fan of Douglas Adams' hilarious books, and want to generate texts in a similarly witty, hilarious writing style?

Good news, I've managed to squeeze them all into this little project I hacked together!

This project was inspired by Andrej Karpathy's nanoGPT - The simplest, fastest repository for training/finetuning medium-sized GPTs. 

### Coming soon -
- Detailed instructions for training on any corpus
- Text-to-speech conversion of generated texts


## install

```
pip install torch torchaudio numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start
If you are just trying to get a feel for how the model works, there's a transcriptions.txt file in the [data/hitchikersguide](data/hitchikersguide) directory. 
You can prepare the dataset by using GPT-2 Byte Pair Encoding as follows:
```
$ python data/hitchikersguide/prepare.py
```
This creates a `train.bin` and `val.bin` in that data directory. 


## side note
In case you want to transcribe speeches or audiobooks that you own, I've added a script [transcribe.ipynb](transcribe.ipynb) which you can use for this purpose!<br />
For context, I could not get OpenAI's Whisper model to run on an M1 MacBook Pro GPU, and performing audio transcriptions on a CPU is painfully slow. To get around this, I opened the .ipynb on Google Collab, uploaded my audiobooks (read by the great Douglas Adams himself!!) in the [content/audiobooks](content/audiobooks) folder and placed the transcriptions.txt file in my repository!


## training from scratch
Now it is time to train your GPT. The size of it very much depends on the computational resources of your system: <br />
**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_hitchikers.py](config/train_hitchikers.py) config file:

```
$ python train.py config/train_hitchikers.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-hitchikers`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```
$ python sample.py --out_dir=out-hitchikers
```

This generates a few samples, for example:

```
Chapteristance were of the crater. No, the other the voice and then as the Galaxy began to make a great, and an last voice and have by the ship is more a white and I you keep a moment it, and closed. He told him, said, he was rather a dead at the water. The planet by something, not me, he said, now to a moment in a bit, he said, the ship. Then it, here, and he not you work, like it, a voice looked out a small barman, which, which. I think and if this— This, she said to the world. The other-meat. In it was a direct that the question, had going to a old full of top of course, and he said Arthur, which was a old man. "'Look, I've think it's able to the old ship. "'I know you do you is going at God, but, you, but it, OK to in the lot of the two. "'What were a great more of the wall of the big, that was a dead, but he was going for to have to lunch. I were going to see that he said, I'd go in I can you don't do what?' said. "'We are he said I are you're up in the very thoroughly much. "'Don't it, and sher was at the control air. "'I don't keep on them. "'Yes, the planet, you do we see you me,' what it, you'd have me!' "'Well,' said Arthur, a second one of the Heart of the best what, and only safe, Arthur. "'And, "'I are that, his end of his sky, "' "'Z Thought, "'Do you know, and it, and he hadn't you have you are,' said Deep, we do you do at the moment to the second right,' shouted Ford. "'But you just just not a moment.' "'Let?' "'Oh, the thin. "'They've think, and waited in the galaxy you you were I have to feel now to him about this later, "'I may have to I like it was so to it, but all the old man-c-etherayfect was a slightered in the reason, and and quite a curiousaf who was that? "'But you, is it, and much, and they've managed to the small, of the galaxy, and
```

lol  `¯\_(ツ)_/¯`. Not bad, but the fact that it starts with "Chapter" is annoying, because I know all my transcriptions start with the word Chapter (There's a lesson about the importance of data cleaning in there!)
Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section below).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```
$ python train.py config/train_hitchikers.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). 

```
$ python sample.py --out_dir=out-hitchikers --start="What is the answer to life, the Universe, and everything?" --device=cpu --num_samples=3 --max_new_tokens=200
```
Generates samples like this:

```
What is the answer to life, the Universe, and everything? said. For of the crater. No, said Arthur. In other, as, muttered headiker from the great, and if you do you have by the ship is going to be no, you keep a moment it, and used to be trying to me. But he was rather a moment at the water. The planet by something, not me, he said Benjged into them in a bit, he said, the ship. Then it, here, and he not you Arthur, like it, a voice looked enough, so had said Ford had been a lot of their first. An Chaptery-lawbleognating. The other-meat. In it was going to that the question, had going to a old full of top of course, and he said Arthur, which was a old man. Either the crater. But a air-ether, because it. Go of the great later the air-bljuman, but, you, but it, OK, in
---------------
What is the answer to life, the Universe, and everything? She was quite saying that. He said. I're more like? said Marvin, it, said Feted on the surface on Beth round it was trying to talk, "'Yes, and was in a planet, well, I can you don't do what?' said. "'We are he said I are you're up in the very thoroughly much. "'Don't it, and sher was at the control air-z, and something, he said. "'Yes, the planet, you do we see you me,' what it, you'd be trying to tell this,' said Arthur, a second one of the Heart of the best idea, and only safe, Arthur. "'And, "'I are that, but a millionbleartfib-ord. The only made a voice, and had been up onto the first moment that's read the crew that my ear. He said.
 Chapterank, "'Oh, you!' said Arthur. "'And he said Arthur
---------------
What is the answer to life, the Universe, and everything? Even now lookeds in the sense of the thin two bay. The fact, and waited in the galaxy, and looked towards the way on. The full of this later, and had his air. Oh, but it was to get the moment all the old man-c-ether idea you know that. The Vogon, they am and quite a curiousaf who was that? No, you, is it, and much, and they've managed to tell you was from the galaxy, and they said. We think of the ship. The voice much your extraelartb Zap on each me. "'I wouldn't be in the book. "'Oh, though you know around them,' said again. "'I'm not go of the bridge. Thessice. "'Oh, and then, no were a man, Why. "'Is, "'I Found of the planet and I't do you anything and they'm just not a surface of earth, and I are a small d
```

Not bad for the (tiny) scale of this model, as it still gives you a feel for the writing style and also throws in some familiar character names! Feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more (and really, really read through the thread. It will save you a lot of time!) :)

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/hitchikersguide` and run `prepare.py` to render the transcriptions into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```
$ python train.py config/finetune_hitchikers.py
```

This will load the config parameter overrides in `config/finetune_hitchikers.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-hitchikers` by default, per the config file. You can then run the code in `sample.py --out_dir=out-hitchikers`:


## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. `$ python sample.py --start=FILE:prompt.txt`.

## reproducing GPT-2 
### (This is something I can't afford to try yet - but if you can, have fun with it!)

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```
$ python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `$ python sample.py`.

Finally, to train on a single GPU simply run the `$ python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.


## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!


## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch Karpathy's [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.
