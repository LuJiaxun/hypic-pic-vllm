# HYPIC PIC integration for vLLM 0.22.1rc1.dev502

The canonical staged migration plan is maintained in
`docs/hypic_pic_vllm_0221_stage_plan.md`. It is the source of truth for
stage boundaries, validation status, and incremental patch baselines.

This branch is based on the Docker image's exact vLLM build commit
`0decac0d96c42b49572498019f0a0e3600f50398`, used by
`vllm/vllm-openai:v0.22.1-cu129`. The CUDA suffix is a build variant; the
Git commit is the source identity used for this port.

The port is being developed in isolated stages. The current branch contains
the following layers:

* Stage 0: token segmentation, segment hashing, metadata lookup, and a
  request-level execution plan.
* Stage 1: PIC handle lifetime management, scheduler-to-worker metadata
  propagation, worker-to-scheduler materialization metadata, and optional
  PIC/non-PIC batch isolation.
* Stage 2: Mamba/GDN state layout discovery for recurrent state and conv tail.
* Stage 3: range/seam/transition planning for non-prefix reuse.
* Stage 4: an independent device byte pool and non-contiguous slot mapping
  primitive.
* Stage 5: a worker safety gate that falls back to ordinary vLLM execution
  until the active attention backend supports range execution.

The normal V1 prefix-cache, block allocator, and model forward path are still
unchanged. The current code does not yet copy real attention KV blocks into
the independent pool, modify vLLM's block table to consume PIC slots, or skip
reused ranges inside the model forward. Those are the remaining integration
steps that require GPU/container validation and model-specific attention
kernel work.

Enable the capability with `--pic-enable`. Request overrides use
`SamplingParams.extra_args`: `pic_enabled`, `pic_mode`, `pic_separator`, and
`pic_seam_sink`. Token-only prompts are classified automatically only when the
configured separator is present; embedding and mixed prompts remain on the
normal path. If physical handles or range execution are unavailable,
`--pic-allow-fallback` keeps the request on the ordinary path.
