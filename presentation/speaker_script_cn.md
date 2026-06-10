# 3 分钟 Highlight Presentation 演讲稿

## Slide 1

大家好，我们的项目是复现 StemGen 的核心思想：context-aware stem generation。具体来说，我们给模型一段去掉 bass 的 mixture，并告诉它目标乐器是 bass，让模型生成和上下文音乐对齐的 bass stem。这个任务不是单纯生成听起来像音乐的音频，而是要求生成结果和已有伴奏在节奏、和声和结构上匹配。所以它既是机器学习问题，也是音频和音乐工程问题。我们的研究问题是：在课程项目的有限算力下，能不能做出一个小规模但结构上忠实的 audio-token 复现。

## Slide 2

我们实现的是缩小版 StemGen pipeline。首先把 context audio 和 bass stem 编码成 EnCodec 的 RVQ tokens。训练时，我们随机 mask 一部分 target bass tokens，模型输入 context tokens、masked target tokens 和 instrument condition，然后用 masked-token Transformer 预测被遮住的 tokens。最后用 iterative mask-predict decoding 把 tokens 解码回音频。因为算力有限，我们使用 24 kHz EnCodec、2 个 codebook 和 10 秒 clip。为了让实验真正有效，我们还做了几个关键工程修正：过滤 bass 很弱或接近静音的片段，预先缓存 codec tokens，避免训练时重复 encode，并把诊断拆成 codec reconstruction、partial-mask reconstruction 和 full-mask generation。

## Slide 3

这是目前最重要的量化结果。550-track cached run 的 best validation loss 是 3.4507，validation accuracy 是 0.521。最新的 1000-track run 使用固定 10 秒 stride，也就是基本非重叠窗口，并且从 550 的 best checkpoint warm start。这个 run 一共得到 22,928 个有效 clips，过滤掉 2,587 个弱 target 或静音 clips。最终 best validation loss 降到 3.0279，validation accuracy 提升到 0.572，相比 550 run，validation loss 大约下降 12.3%。这说明更大的 track 覆盖、更少的重叠、静音过滤和 token cache 对这个小规模复现都很关键。

右侧这张完整的 mel-spectrogram 是定性证据。target bass 里面可以看到清晰的块状谐波结构；codec reconstruction 保留了主要结构；partial-mask reconstruction 也跟随了不少类似的时频块。所以我们认为这个结果不只是 loss 下降，而是模型确实学到了一些有效的 audio-token 结构。当然，100% mask 的 full generation 仍然明显不稳定，这也说明 full context-to-stem generation 还是当前主要瓶颈。

## Slide 4

从定性结果看，模型已经不只是输出纯噪声了。尤其是在 partial-mask reconstruction 下，听感和频谱都能看到一些合理结构，说明模型确实学到了 target token 的条件重建能力。但是 100% mask，也就是完全从 context 生成 bass 的时候，结果仍然不稳定，有时会退化成噪声。所以我们的结论是：当前小模型已经能学习 audio-token reconstruction，但完整的 context-to-stem generation 还是主要瓶颈。下一步我们会重点改进 full-mask decoding，比如尝试更好的 mask schedule、classifier-free guidance 或 condition dropout，并用统一的音频样本和 spectrogram 来做最终报告里的定性评估。
