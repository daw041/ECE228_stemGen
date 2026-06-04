# 3 分钟 Highlight Presentation 演讲稿

## Slide 1

大家好，我们的项目是复现 StemGen 这篇工作。它的核心任务不是普通的音乐生成，而是 context-aware stem generation：给模型一段去掉 bass 的 mixture，再告诉它目标乐器是 bass，让它生成和上下文音乐对齐的 bass stem。这个问题符合课程要求，因为它是机器学习在音频和音乐工程里的应用，关键难点是生成结果不仅要像音频，还要和已有伴奏在节奏、音高和结构上匹配。

## Slide 2

我们的实现是一个缩小版的 audio-token StemGen pipeline。首先把 context audio 和 target bass stem 编码成 EnCodec 的 RVQ tokens。训练时，target tokens 的一部分会被 mask，模型输入 context tokens、被 mask 的 target tokens，以及 instrument condition，然后用 masked-token Transformer 预测被遮住的 token。最后通过 iterative mask-predict decoding，把预测 token 解码回音频。由于计算资源有限，我们使用 24 kHz EnCodec、2 个 codebook、10 秒 clip，以及 multi-codebook cross entropy。

## Slide 3

一开始 audio-token 路线的结果非常差，生成出来基本是噪声。所以我们做了几轮修正。第一，检查并对齐 EnCodec codebook 的处理方式。第二，过滤掉 bass 很弱或者接近静音的片段，否则模型会在无意义片段上获得虚高准确率。第三，把 codec tokens 预先缓存下来，避免训练时反复 encode 浪费 GPU。第四，我们把诊断拆成三个层次：codec reconstruction、partial-mask reconstruction 和 full generation。这样可以知道问题到底出在 tokenizer、模型学习，还是最终从 context 完整生成 target 的阶段。

## Slide 4

这是目前最重要的量化结果。550-track cached run 的 best validation loss 是 3.4507，validation accuracy 是 0.521。最新的 1000-track run 使用 10 秒固定 stride，也就是基本非重叠窗口，并且 warm start 自 550 的 best checkpoint。这个 run 的 best validation loss 降到 3.0279，validation accuracy 提升到 0.572。相对 550 run，validation loss 下降约 12.3%。这说明扩大 track 覆盖、减少重复片段、过滤弱 target，以及 token cache 都对这个小规模复现有帮助。

## Slide 5

从主观结果看，模型已经不是只输出纯噪声了。特别是在 partial-mask reconstruction 下，频谱和听感都能看到一些正确结构，说明模型确实学到了一部分 target token 的条件重建能力。但是 100% mask，也就是完全从 context 生成 target 的时候，结果仍然不稳定，容易退化成噪声。我们的理解是：现在的小模型已经能做局部 token reconstruction，但要达到论文里完整 stem generation 的效果，还需要更强的 decoding、condition guidance，或者更大的模型和 tokenizer 设置。

## Slide 6

总结一下，我们完成了一个小规模但结构上忠实的 StemGen audio-token 复现，并且找到了当前最主要的瓶颈。有效的部分包括 masked audio-token modeling、token cache、静音过滤，以及 1000-track 的非重叠训练。下一步我们会重点改进 full-mask decoding，比如 classifier-free guidance、condition dropout 和不同 mask schedule，同时用统一的音频样本和 spectrogram 做定性评估。这会成为 final report 里最核心的实验线索。

