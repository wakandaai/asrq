# LayerNorm, RMSNorm, rmsnorm
- [x] Class to convert a LayerNorm to an RMSNorm while folding the mean subtraction matrix and shift parameter into the previous and next linear layers around the LayerNorm, respectively.
- [x] A class that performs the rmsnorm without applying the scaling parameter. This converts an RMSNorm layer to one with no scaling, just pure rmsnorm. The scaling parameter is folded into the next linear layer.


# Rotation Folding into Linear layers
- [x] Functions that fold the rotations R1 and R2 into the linear layers. Specifically for the Q, K, V, O, FC1 and FC2 layers. We also recognize that pointwise Conv1d layers are involved and treat them as linear layers.
- [x] Functions that patch the forward method of the linear layers to apply the rotations during the forward pass. This enables Cayley SGD.
- [x] Online hadamard rotations for the down projection layers. The rotation is introduced in an equivalent transformation and the transpose is absorbed into the linear layer and the hadamard rotation remains in the forward pass and applied to the input of the down projection layer.
- [x] At inference, the online hadamard is a separate module placed after the activation that feeds the down projection (activation = Sequential(act, OnlineHadamard)), instead of patching the down projection's forward. The down projection then stays a plain linear layer, and a forward hook on it sees X @ H, so the GPTQ Hessian is in the right basis. During rotation learning, the dense X @ H is still applied inside the patched forward, since humming is not differentiable.
- [x] Randomized online hadamard for the down projection: random +-1 signs before the hadamard (X @ D @ H), with D @ H absorbed into the layer. A plain hadamard maps the mean of the mostly non-negative GELU output onto one coordinate per block, which made fc2 quantize worse with the rotation than without. The signs are saved with the rotations, one vector per fc2 input width.
- [x] Option to learn and apply only R1, with no R2 and no online hadamard. The inputs of the attention output projection and fc2 then stay unrotated and are quantized group-wise.
- [x] Option for a block-diagonal R1, with blocks aligned with the activation quantization groups.
- [x] Remove the forward patches that apply a hadamard inside the down projection. They are no longer used and give GPTQ the Hessian in the wrong basis.
- [x] Check that the hadamard block size equals the activation group size when fc2 is quantized group-wise.


# Cayley SGD
- [x] Learn a shared R1 and one head-wise R2 per attention module with Cayley SGD on the Stiefel manifold, with the rotations patched into the forward passes.
- [x] Check that the model output is unchanged after norm folding and after rotation patching, comparing logits with TF32 disabled.
- [x] Activation fake quantization with a straight-through estimator during the search. Each layer is quantized per-token or group-wise according to its role (q, k, v, attn_out, fc1, fc2, block_out), set in the config and shared with evaluation.
- [x] Choice of objective: KL divergence to the full-precision model (the same model with quantization switched off), or cross-entropy on the calibration labels.
- [x] Log the loss, learning rate and R1 orthogonality every few steps.
- [x] Fold the learned rotations into the weights for inference.
- [x] Whisper and Parakeet CTC: norm mapping, layers to rotate, the hook that centers and rotates the residual stream, and the hook that rotates it back before the final norm.
- [x] Learning rate: 0.5 is stable. Since the fc2 fix the loss barely moves from the random hadamard initialisation.
- [x] Compare R1 only against R1 + R2 + online hadamard: the full rotation is better, mainly on test-other.
- [x] Whisper end to end (rotation, GPTQ W4, WER): W4A4 matches full precision, both all per-token and with attn_out/fc2 group-wise, on 256-utterance runs.
- [x] Residual-stream rotations of an applied rotation are modules (the rotation is a buffer, the hook is the module's own method), so the rotated model saves and loads with torch.save / torch.load, and its state_dict holds R1. The search still uses plain hooks on the live R1.
- [x] attach_rotation_hooks is a required argument of learn_rotations and apply_rotations; leaving it out silently gave a wrong model for Whisper and Parakeet. Pass None explicitly only for a model whose residual stream needs no transform.
- [x] Whisper at full scale (learned rotation on 2048 calibration utterances, GPTQ W4 on 2048, fake A4 with attn_out/fc2 in groups of 128, bfloat16, batch 64): all of LibriSpeech test-clean 2.11 and test-other 4.44, against 1.94 and 3.88 unquantized (2620 and 2939 utterances). Rotation search 512 steps, average KL 0.0197 from 0.0274 at the random Hadamard start.
- [ ] Parakeet and Canary-Qwen at full scale; the other leaderboard splits for all three (ami, earnings22, gigaspeech, spgispeech, voxpopuli; ~19.5 GB to download).
- [x] ModelQ.quantize collects reference cycles and frees the CUDA cache when it finishes: the block-wise quantizers keep one captured input per calibration sample alive through their hooks, 33 GB for whisper-large-v3 on 2048 samples, which left evaluation without memory. Whisper needs eval_batch_size 64, not the model config's 192.
- [x] exp.py saves the quantized model (`quantized_path: auto` -> outputs/quantized/<model>-<settings hash>.pt) and loads it on a re-run instead of quantizing. Saved at the tensors' own dtype: a float16 save changed the WER of the reloaded bfloat16 model (Parakeet 8 utt 2.61 -> 2.44).
- [ ] Learned rotations are not yet shown to beat a random hadamard (Whisper, Parakeet and Canary-Qwen, 256 utterances). Try starting from the identity, longer searches, and compare on more data.
- [ ] The residual-stream hooks still break torch.compile graphs (humming itself also fails under compile).


# Scaling (SmoothQuant-style)
- [x] Scaling targets exactly the layers activation quantization uses (the same roles as rotation). The search scores s=1 and an alpha grid per entry (q/k/v jointly) by output error under the evaluated quantization: group-wise weights (GPTQ group size) and role-wise activations (group-wise for attn_out/fc2). At W4A8 Whisper keeps s=1 for 219 of 321 entries. The reciprocal is folded into the previous norm, linear or pointwise conv.
- [x] Where an activation feeds the layer (fc2, feed_forward linear2, conv.pointwise_conv2) the reciprocal is an InputScale module after the activation (activation = Sequential(act, InputScale)), like the online hadamard, so the activation quantizer, the GPTQ Hessian and humming ASRQLinear see the scaled input. Qwen down_proj's reciprocal is folded into up_proj's output rows (the SwiGLU product is elementwise).
- [x] Canary-Qwen scaling uses the merged LLM's names and a teacher-forced calibration forward.
- [x] exp.py with transform=scaling for Whisper, Parakeet and Canary-Qwen, fake and humming (W4A8, 64 utterances): humming matches fake within 0.12 WER. tests/test_scaling.py (exactness on tiny models, hooks see the scaled input, same layers as rotation) and scaling cases in the model integration tests.
- [ ] Compare no transform, scaling and rotation at W4A4 (256 utterances); scales searched at A4.

# Scale recovery
- [x] Closed-form norm tweaking in the GPTQ block loops (`quantizer.norm_tweak`, asrq/quantizers/norm_tweak.py): after each block's GPTQ, every norm feeding its quantized layers is scaled per channel by s solving (H * sum Q^T Q + lambda I) s = diag(H sum W^T Q) + lambda, from the Hessian GPTQ collected; weight error only.
- [x] W2 asymmetric g128 GPTQ, 256 utterances test-clean / test-other, without -> with norm tweaking:
  - Whisper, random Hadamard: 2.44 / 7.62 -> 3.03 / 5.59; no rotation: 3.36 / 9.36 -> 3.22 / 7.30.
  - Parakeet, random Hadamard: 2.96 / 4.80 -> 2.54 / 4.42; no rotation: 4.38 / 6.32 -> 3.47 / 5.40.
  - Canary-Qwen, random Hadamard: 52.6 / 36.0 -> 5.55 / 7.74 (the looping collapse is gone); no rotation: 103 / 109 -> 102 / 103.
- [x] W4A4 (learned rotation, GPTQ W4, A4 mixed 128), 256 utterances test-clean / test-other, none -> closed form: Whisper 2.39 / 4.15 -> 2.24 / 4.41; Parakeet 2.08 / 3.12 -> 1.84 / 3.00; Canary-Qwen 1.71 / 2.72 -> 1.56 / 2.65 (full precision 1.47 / 2.46).
- [x] A gradient variant (Adam on the norm scales with fake-quantized activations, from the closed form) was tried and removed: Canary-Qwen 1.54 / 2.57, Whisper and Parakeet within noise of the closed form, and hard to tune. Only the closed form is kept.
- [ ] Scale recovery at W4A4.
- [x] Compared fitting methods: closed, posthoc (fit s Q ~ the updated weight W~ GPTQ rounded, after GPTQ), inloop (s_j fitted as each column is rounded, the scaled error compensated; a norm's layers stacked into one GPTQ), inloop_average (per layer, then column-energy weighted average). W2 random Hadamard, 256 utt, closed / inloop / inloop_average / posthoc: Parakeet 2.54/4.42, 2.65/5.20, 2.75/5.32, 6.97/13.35; Canary-Qwen 5.55/7.74, 4.08/6.73, 3.99/6.34, 8.87/16.04; Whisper 3.03/5.59, 3.14/5.74, 3.32/5.93, 34.1/70.0. In-loop lowers every Parakeet layer's held-out output error below closed (0.84 vs 0.93 of GPTQ) yet raises its WER.
- [x] Only the in-loop fit is kept (closed form and posthoc removed); `quantizer.norm_tweak_method`: lockstep (default) or average.
- [x] Renamed to scale recovery (`quantizer.scale_recovery`, asrq/quantizers/scale_recovery.py): the fit is a per-input-channel scale on the quantized grid, and a norm is only one of the things that can absorb it. Folding into a preceding layer's rows (out_proj into v_proj, tied per value channel under Qwen's GQA; down_proj into up_proj) was implemented and removed at the user's request: the folds are exact (3e-16 on a real attention module) and free, but W2 256 utt measured flat to worse (Whisper 3.10/6.31 against 3.14/5.74, Canary-Qwen 3.32/5.40 against 3.21/5.20).
- [ ] Compare lockstep and average at full scale.

# Block output refitting
- [x] The inserted Conformer block-output Linears are refit and then quantized when `model.quantize_block_output_linear` is set (their Hessian comes from the refit pass, on the quantized block's inputs), and `quantizer.block_output_linear_bits` gives them their own width. Parakeet W2 256 utt, 41 layers of 86 MB in fp16 against ~275 MB of W2 encoder weights: refit-then-quantize / quantize-only: fp16 2.65/4.47 / 2.96/4.80, W8 2.54/4.68 / 2.87/4.65 (43 MB), W4 2.91/4.96 / 3.43/5.20 (21.5 MB), W2 4.08/6.40 / 3.49/5.51 (11 MB). W8 is the useful trade; the refit helps down to W4 and hurts at W2, where it moves the weights off the near-identity map a 2-bit grid can hold.
- [x] `quantizer.block_output_refit` (asrq/quantizers/output_refit.py), independent of norm tweaking: after each block's GPTQ, the block's output Linear (the identity a rotation inserts after a Conformer block's norm_out) is refit, weight and bias, in closed form to the full-precision block output, ridge toward the current weights. Output layers that are quantized are skipped.
- [x] 256 utterances test-clean / test-other, none / norm tweak / refit:
  - Parakeet W2 (random Hadamard): 2.96 / 4.80, 2.54 / 4.42, 2.65 / 4.47 (block error x0.65).
  - Parakeet W4A4 (learned rotation): 2.08 / 3.12, 1.84 / 3.00, 1.86 / 2.97 (x0.70).
  - Canary-Qwen W2: 52.6 / 36.0, 5.55 / 7.74, 55.7 / 41.6 (only the encoder has output layers; the LLM is where W2 breaks).
  - Canary-Qwen W4A4: 1.71 / 2.72, 1.56 / 2.65, 1.63 / 2.66.
- [x] `quantizer.block_output_refit_insert`: an identity `output_linear` (forward hook on the block, not quantized) is inserted into every transformer block (Whisper encoder/decoder, the Qwen LLM) and refit the same way. None / norm tweak / refit + inserted, 256 utterances:
  - Whisper W2: 2.44 / 7.62, 3.03 / 5.59, 3.10 / 5.10 (block error x0.48).
  - Whisper W4A4: 2.39 / 4.15, 2.24 / 4.41, 2.26 / 4.35 (x0.52).
  - Canary-Qwen W2: 52.6 / 36.0, 5.55 / 7.74, 5.17 / 5.83 (LLM x0.56, encoder x0.63).
  - Canary-Qwen W4A4: 1.71 / 2.72, 1.56 / 2.65, 1.56 / 2.62.
- [ ] Measure the inference cost of the inserted fp16 d x d layer per block (Whisper, Canary-Qwen), and whether it can run quantized.
- [x] `quantizer.block_output_refit_rank`: the inserted transformer-block layers as identity plus a rank-r correction (x + (x @ A) @ B + b, asrq/quantizers/output_refit.py LowRankOutput), fitted by reduced-rank regression on the block error (A = M V_r, B = V_r.T with V_r the top eigenvectors of M.T S M, M the unconstrained fit). Conformer norm_out.1 stays a full weight and bias refit. W2 random Hadamard, 256 utt, dense / d4 / d8 / d16: Whisper 3.10/5.10, 2.33/5.49, 2.85/5.76, 2.72/6.00; Canary-Qwen 5.17/5.83, 5.34/8.08, 13.0/13.2, 15.2/14.3. Dense stays the default: Canary-Qwen's LLM needs the full rank, Whisper trades test-clean for test-other.
- [ ] Optional: the eigenvalue spectrum of M.T S M per block, to see how many directions carry the error reduction.
- [x] The recovery recipe: norm tweaking for transformer blocks, block output refitting for Conformer blocks, both enabled (the inserted-layer refit for transformer blocks is removed). W2 256 utt: Parakeet 2.65/4.47 (refit only), Whisper 3.14/5.74 (norm tweak only), Canary-Qwen 3.21/5.20 (encoder refit + LLM norm tweak), its best single correction being 4.08/6.73.
- [ ] The recipe at full scale and at W4A4.

# Hadamard Rotation Search
- [x] `transform.search: evolution`: R1 = diag(s1) @ H @ diag(s2) searched by a (1 + lambda) evolutionary algorithm over the signs (2 flips per child), three-stage selection (8/16/64, 8/64/256 or 16/64/N samples, random subsets per generation), KL fitness with cached full-precision logits; R2 stays a random Hadamard. Shares learn_rotations' verification and checkpoint format; the checkpoint records the signs and per-generation history.
- [x] Evolution settings raised: 32 offspring (default), children deduplicated within a generation and against an archive of every candidate scored so far (sha1 of the signs), stages 16/32/N for N <= 128. 128 search samples cost 896 scored samples per generation against 320 at 64 samples with 16 offspring.
- [x] Search cost: the evolutionary search's batches are grouped by audio length (`length_sorted_batches`, evolution only), which cuts Parakeet's padded audio from 1.21-1.29x to 1.01-1.07x (16-19% less compute); Whisper pads to 30 s and gains nothing. Whisper fitness is 244/220/211 ms per sample at batch 4/8/16 in float32.
- [x] `transform.fitness_dtype` (autocast of the search's teacher and student passes, default float32): bfloat16 is 2.7x faster on Whisper but unusable. Across 16 children of one parent the fp32 KL spread is 0.0029 while bfloat16's deviation is 0.0041 (1.4x the spread), Spearman rank correlation 0.11, top-4 overlap 0 of 4.
- [x] Racing children against the survivor cut was implemented and reverted: KL accumulates positively, so a partial mean is a weak lower bound and a child within ~8% of the cut only aborts after ~93% of its batches. Measured no saving on a toy problem.
- [x] A rotation checkpoint records the model and the quantization it was searched against; apply_rotations refuses another model (by the recorded name, or by R2 coverage and R1 width for checkpoints without it) and check_rotation_settings refuses different activation settings or weight grid (`transform.check_settings`).
- [x] With symmetric activation quantization s2 does not change the KL (it flips already-rotated coordinates); with asymmetric it does. `evolution.mutate: auto` (default) flips s1 only for symmetric and both for asymmetric; `s1` and `s1_s2` force either.
- [x] Whisper smoke run (64 samples, A4): KL 0.0400 -> 0.0311 in 3 generations, ~150 s per generation (320 scored samples, float32 forwards at ~450 ms/sample; TF32 ~250 ms/sample).
- [x] Budget: Cayley SGD takes 5.8 s per step (batch 4, float32), 200 steps ~19 min; the evolutionary search matches it with 8 generations at 64 samples (default `generations: 8`).
- [x] Whisper at equal search time (symmetric A4 mixed; evolution 64 samples, 8 generations, 21.6 min; Cayley SGD 800 samples, lr 0.5, 20.8 min). Held-out KL (256 calibration samples): random 0.0297, Cayley 0.0170, evolution 0.0290 (search-set KL -15%, so it mostly fits its 64 samples). WER with GPTQ W4, 256 utterances test-clean / test-other: random 2.43 / 4.78, Cayley 2.27 / 4.91, evolution 2.25 / 4.53 -- within noise of each other.
- [ ] Parakeet and Canary-Qwen comparisons; a larger evolution budget or more search samples, since 8 generations change at most 16 of 1280 signs.
- [x] Weight-only R1 search (`transform.weight_only: true`, evolution only): each candidate is folded into the norm-folded weights with the random R2s and online Hadamards, the quantized layers are rounded with RTN (quantizer bits, group size, symmetry), activations stay full precision; KL fitness. `mutate: auto` follows the weight symmetry (s2 flips columns on W @ R1, which only asymmetric rounding sees, and whole rows on R1.T @ W, which neither does). The saved fitness matches an independent apply_rotations + RTN recomputation.
- [x] Whisper weight-only smoke run (W3 g128 symmetric, 64 samples): RTN KL 0.1008 -> 0.0728 in 3 generations, ~72 s per generation (folded forwards, no patching).
- [x] Weight-only W2 asymmetric g128, WER on 256 utterances (test-clean / test-other), RTN-fitness search (64 samples, 8 generations):
  - Whisper GPTQ: none 3.36 / 9.36, random Hadamard 2.44 / 7.62, evolution 3.10 / 6.08; RTN broken (~100) for all.
  - Parakeet GPTQ: none 4.38 / 6.32, random 2.96 / 4.80, evolution 2.98 / 4.85; RTN: none 100, random 54.4 / 52.6, evolution 35.3 / 37.2.
  - Canary-Qwen GPTQ: none 103.4 / 108.7, random 52.6 / 36.0, evolution 52.7 / 35.7; RTN broken for all.
- [x] GPTQ fitness for the weight-only search (`transform.weight_only_quantizer: gptq`, or null to follow the experiment's quantizer): Hessians collected once with R1 = I; residual-stream layers use R1.T @ H @ R1 per candidate, the others reuse fixed factors; layers of one shape are GPTQ-quantized together (asrq/quantizers/gptq_solver.py, shared with the GPTQ quantizer). Whisper W2 asymmetric: ~6 s per candidate (was ~120 s per layer loop), 199 s per generation; KL 0.0698 -> 0.0575 in 2 generations. The Hessians come from full-precision inputs, not sequential GPTQ.
- [x] GPTQ-fitness search (64 samples, 8 generations), W2 asymmetric g128, GPTQ WER test-clean / test-other: Whisper 2.84 / 5.64 (KL 0.0698 -> 0.0519, 28 min); Parakeet 2.85 / 4.71 (KL 0.446 -> 0.388, 14 min); Canary-Qwen 54.9 / 54.2 (KL 2.55 -> 1.18, 41 min; random Hadamard 52.6 / 36.0). Original weights are kept in pinned CPU memory and GPTQ groups capped at 5e8 entries so Canary-Qwen fits in 48 GB.
- [x] Why Canary-Qwen fails at W2 (random Hadamard, GPTQ W2 asymmetric g128): speech encoder only 1.90 / 3.42 (full precision 1.47 / 2.46); Qwen3 LLM only 68.7 / 51.4, with transcripts that start correctly and collapse into repetition (13-20 of 256 over twice the reference length). Relative W2 rounding error per layer is the same in all three models (~0.5), so the LLM is more sensitive to the same weight error, not harder to round.
- [x] Canary-Qwen LLM sensitivity (random Hadamard, GPTQ asymmetric from full-precision Hessians, KL on 64 held-out calibration utterances): whole LLM W4 g128 0.0007, W3 g128 0.0044, W2 g32 0.042, W2 g64 0.205, W2 g128 1.157. One layer type at W2: 0.0010-0.0044; one block at W2: 0.0003-0.0024 (block 0 and the last two highest), summing to 0.0245 -- 47x less than all blocks together, so the W2 collapse is error compounding through depth, not a few sensitive layers.
- [x] Canary-Qwen WER, encoder W2 g128 + LLM (GPTQ, 256 utterances test-clean / test-other): W2 g64 8.67 / 10.24, W2 g32 3.18 / 5.20, W3 g128 2.13 / 3.64, W4 g128 1.84 / 3.51 (encoder W2 alone 1.90 / 3.42, full precision 1.47 / 2.46).
- [ ] Canary-Qwen at ~W2 LLM: recovery fine-tuning (LoRA on the quantized LLM with the full-precision model as teacher, or block-wise scale/clipping tuning).


# Extreme low-bit quantization (2-bit)
- [x] W2A16 speed (batch 1, CUDA graphs): decoding step 1.14x fp16, encoder unchanged; linear weights 373 MiB vs 2801 MiB.
- [ ] W2 accuracy: RTN 2-bit breaks Whisper (WER 100). Try GPTQ.


# Calibration Data
- [x] Save 2048 LibriSpeech train.clean.360 utterances (30 s or shorter) to disk, and transcribe them with each model. The model's own transcripts are the calibration labels for both rotation learning and GPTQ; looping, truncated and empty transcripts are dropped.
- [x] Teacher-forcing targets start from the same forced prefix as generation, and the prefix tokens are left out of the loss.
- [x] Whisper, Parakeet and Canary-Qwen transcripts for the calibration set.
- [ ] The calibration audio is read audiobook speech only; consider other domains.


# Evaluation and experiments
- [x] Activation quantization at evaluation follows the config (bits, group size, group-wise roles), fc2 included, using the same quantizer as the search.
- [x] Evaluation datasets, the number of batches, the dtype (`eval_dtype`) and the transcription function (`model.generate_fn`, eager or CUDA graphs) are set in the config.
- [x] Evaluation loads the leaderboard's data utils from the submodule without it being on PYTHONPATH; asrq is installed with pip install -e .
- [x] exp.py and rot-exp.py run through asrq/experiment.py (prepare_experiment_config, run_experiment, learn_rotation_experiment). Verified end to end: exp.py for Whisper and Parakeet (fake W4A4; humming + CUDA graphs + float16) and Canary-Qwen (fake W4A4), rot-exp.py for Whisper and Parakeet.
- [x] tests/test_whisper.py and tests/test_parakeet_ctc.py (marker `integration`): learn a rotation, then rotate + GPTQ + evaluate, fake and humming, at 8-utterance scale. Whisper is held to the unquantized WER on the same utterances (the first 8 are over 30 s and truncated). tests/test_canary_qwen.py does the same for Canary-Qwen. The old shrinking-transform scripts are replaced.
- [ ] Check WER in float16 at scale before using eval_dtype: float16 for accuracy numbers.
- [ ] Graphed transcripts differ from eager on some utterances (Whisper W4A4 ~8%, from static-cache or padding rounding); check the WER effect at scale.
- [ ] The Open ASR Leaderboard runs Whisper and Parakeet-CTC eagerly in bfloat16, without CUDA graphs; report which setup each speed number uses.


# Inference Speedup
- [x] Real low-bit inference: ASRQLinear runs humming's quantized matmul (packed weights, per-token or group-wise activation quantization from the config, fused fc2 hadamard + signs, pointwise Conv1d layout); `inference: humming` converts the model after quantization. WER matches fake quantization for Whisper and Parakeet.
- [x] ASRQLinear's humming call selects the same kernels as humming's bench_humming.py; the benchmark only excludes input quantization from its timing.
- [x] W4A4 run-to-run nondeterminism: humming's Stream-K schedules. Layers with quantized activations use the batch-invariant mode (bit-identical, slightly faster); fp16-activation layers keep the default, which is faster for them.
- [x] `eval_dtype` config: in bfloat16 every ASRQLinear casts to fp16 and back, which cancelled the speedup. Use float16 for speed.
- [x] Whisper: CUDA-graph generation (`generate_whisper_cuda_graphs`), matching generate's transcripts. W4A4 is 1.22x fp16 at batch 1 and 1.15x at batch 64 (both graphed, float16); eager generate is slower than fp16 (humming's per-call Python overhead).
- [x] Profiles: quantized layers are ~56% of Whisper's encoder time and 9-47% of a decoder step (cross-attention dominates at large batch); ~51% of Parakeet's acoustic model (relative-position attention is 24%).
- [x] Prototype (outputs/humming_seeded, a patched copy of humming): process_input takes hadamard_sign_seed and multiplies each loaded value by a lowbias32-hashed sign of its column before the Hadamard. Bitwise equal to multiplying by the same signs first (Hadamard alone, W4A16, W4A4 g128); Whisper fc2 1.14-1.27x faster at A16 and 1.14-1.48x at A4 (up to 1.45x at 96k tokens) than the separate multiply. A constant input's crest factor is 3.5 with the seeded signs against 11.3 with the plain Hadamard.
- [x] humming fork (github.com/ldfrancis/humming) as submodule third_party/humming, branch hadamard-sign-seed from e5829681 (the commit previously installed) with the seeded-sign patch; installed editable. Seeded kernel bitwise equal to the signs multiply; fast suite passes.
- [ ] Commit and push the patch on the fork's branch, and record that commit in the superproject (git add third_party/humming).
- [x] Seeded signs adopted: rotation checkpoints save `hadamard_sign_seed` (one per model, drawn from torch's generator) instead of sign vectors; the fold uses seeded_hadamard_signs; OnlineHadamard and ASRQLinear pass the seed to humming instead of multiplying (bit-identical at A4; within the A16 GEMM's own run-to-run ulp). Checkpoints with stored sign vectors still apply through the multiply path.
- [ ] Optional: quantize the shared q/k/v input once; quantize the cross-attention KV cache, which dominates Whisper decoding at large batch.
- [ ] Saving and loading a model converted to ASRQLinear is untested.


# Parakeet-CTC (Conformer)
- [x] An identity linear layer after the norm_out of every conformer block except the last, so the next block's centering has a linear layer to fold into. R1 is folded into it on both sides (checked numerically on the real model).
- [x] Rotation of both feed-forwards and the conv module's pointwise convs; online hadamard with random signs at every feed_forward linear2 and conv.pointwise_conv2 (channels-first). GPTQ quantizes the pointwise convs.
- [x] Learned rotation + GPTQ W4A4 is within ~0.2 WER of full precision; without rotation W4A4 is unusable (>90% WER). A random hadamard does as well as the learned rotation (256 utterances).
- [x] Option `quantize_block_output_linear` (model config) weight- and activation-quantizes the inserted linear (role block_out) in the search, GPTQ/RTN, evaluation and ASRQLinear. 0.7% faster acoustic model, WER unchanged within noise. Default off. Removing the layer altogether (speed test only) would gain 1.5-2.5%.
- [x] Speed (humming W4A4, float16): acoustic model 1.16-1.19x fp16 at batch 32-128, 1.29x at batch 1 with CUDA graphs.
- [x] CUDA-graph CTC transcription (`generate_parakeet_cuda_graphs`): 5 s length buckets, one graph per batch size and bucket. W4A4 1.31x fp16 at batch 1 and 1.14x at batch 128 (both graphed, warm). NeMo transcribe leaves the encoder in training mode; the graph path resets eval mode. NeMo models cannot be deep-copied.


# Canary-Qwen
- [x] LoRA adapters (q_proj, v_proj) merged into the Qwen3 LLM on load (CanaryQwenQ.load_model, canary_qwen_utils.merge_lora).
- [x] Rotation with one R1 per residual stream: learn_rotations / apply_rotations accept `hidden_size` and R1 as `{stream: ...}`, with the stream named in each layers_to_rotate entry. Encoder mapped as for Parakeet under `perception.`; LLM q/k/v/o, gate/up, down with R2 per layer (GQA), Qwen RMSNorm scales folded (float32 normalization kept), rotate-only entry hook on llm layers[0], unrotate before the final norm (the output head is tied to the embedding).
- [x] Online hadamard at down_proj's own input (SwiGLU has no single activation to follow): an OnlineHadamard child run by a forward pre-hook; ASRQLinear fuses it.
- [x] Canary-Qwen transcripts for the calibration set; teacher-forced calibration on "Transcribe the following: <audio>" + transcript, targets on the transcript and <|im_end|>.
- [x] GPTQ of the encoder's Conformer blocks (linears and pointwise convs) and the LLM's decoder layers on the rotated, merged model.
- [x] Results (256 utterances, full precision 1.47 / 2.46): without rotation A4 is unusable (>108% WER, also after GPTQ W4); learned rotation + GPTQ W4A4 mixed 1.71 / 2.72, per-token 1.60 / 2.93; random hadamard + GPTQ W4A4 mixed 1.68 / 2.54. Learned and random are within noise.
- [x] tests/test_canary_qwen_rotation.py (tiny model with the same structure: exactness, tied head, fp16, fused down_proj hadamard) and tests/test_canary_qwen.py (integration).
- [x] CUDA-graph generation (`generate_fn: generate_canaryqwen_cuda_graphs`): encoder graph per batch size and 5 s bucket, prompt graph assembling the left-padded prompt on the GPU into a static KV cache, decode-step graph; matches generate's transcripts within 2% word disagreement. float16 is safe for Canary-Qwen (WER 1.49 vs 1.47 in bfloat16).
- [x] Speed (float16, humming W4A4, learned rotation, RTN weights): 1.45x fp16 at batch 1 with CUDA graphs (RTFx 49.5 vs 34.1), 1.07x at batch 128 (234.4 vs 218.9); eager W4A4 at batch 1 is 0.42x fp16.
- [x] GPTQ weights through humming match fake quantization (same model, 256 utterances, fp16): fake W4A4 mixed 1.63 / 2.63, humming eager 1.56 / 2.64, humming CUDA graphs 1.63 / 2.59 (RTFx 236 at batch 128, as with RTN weights).
- [ ] Profile why batch 128 gains little (LLM attention over ~400 audio tokens and the static cache's slots are unquantized).
- [ ] Confirm at scale and on other datasets.
