# Evaluating an alternative model

Use this procedure for a concrete request such as clearer Finnish names, a more natural mouth during pauses, a smaller download, a different language or a device the baseline cannot use. Define that success criterion first. A new release date alone does not make a candidate better.

## Research before downloading

Read a prior dated evaluation if one exists under the chosen job's ignored working directory. Reuse relevant results while checking facts that can change. Browse the official project repository and releases, model card/checkpoint repository, papers and hardware instructions. Separate release date, source revision and the date checked. Marketing videos and third-party rankings may suggest leads; they do not establish local performance.

Compare the current pins with a small relevant shortlist. Record:

- Capability: the actual target language, reference conditioning, pronunciation controls, motion versus lip-sync scope and input limitations.
- Assets: exact checkpoint/revision, tokenizer/codec/detector dependencies, expected download bytes and total additional installed/cache space.
- Execution: supported OS/device, precision, RAM/VRAM claims with their source and scope, CPU fallback if documented, and whether an online API is required.
- Integration: loader/API, sample rate, frame rate, inference-window behavior, metadata/caption interfaces and any code needed to make it work here.
- Terms: code and weight licences separately, relevant redistribution/use limits, and watermark/disclosure behavior. Preserve required notices; do not assume the project's MIT licence covers a model.
- Evidence: demonstrated target-language quality, maintenance state, reproducible benchmark settings and unresolved claims. Label manufacturer claims, measured results and your inference separately.

If browsing is unavailable, offer the stable baseline and say that the alternative-model comparison is not current. Do not substitute guessed sizes or claimed language support.

Write the findings in `projects/<job-id>/work/model-evaluation.md`, keeping personal references and test utterances private. Include links beside claims and a date. Explain the recommendation to the user in terms of the desired improvement, additional resources and uncertainty. Check the machine again using candidate-specific resource needs; the baseline resource report cannot certify another model.

## Run a bounded comparison

Use an isolated environment and new output directories for a selected trial. Retain the working baseline, pins, successful outputs and provenance. Do not replace its packages/checkpoints during exploration. Respect the user's download, device and time constraints; when a trial would exceed existing authorization, present its concrete scope and size before running it.

Use the same intended short text, displayed names, reference source and source-video interval where the models permit it. Match meaningful settings and document unavoidable differences. A matching seed across different model architectures does not ensure a controlled or deterministic performance.

Record exact revisions, commands, output hashes, elapsed wall time and audio/video duration. If measuring peak RAM/VRAM, specify how it was sampled and its limits; a single `nvidia-smi` snapshot is not a peak measurement. Run decode, timing, clipping, caption and provenance checks on the resulting files.

Have the user compare the short takes for the stated criterion: accent, name intelligibility, omitted words, prosody, mouth detail, pause behavior, cuts and overall naturalness as relevant. Use ASR to locate possible errors, not to declare a winner. If the candidate changes the acting/motion model, do not attribute its result solely to lip synchronization.

## Decide and preserve the evidence

Summarize what improved, what regressed, actual resource use and any untested assumptions. An attractive upstream demo or a successful import is insufficient for adoption. A technically valid clip can still be rejected by the listener.

If adoption is wanted, prepare a separate implementation change with pinned dependencies/assets, provenance and export compatibility, model-specific resource assumptions, updated setup documentation and a tested path back to the baseline. Keep the previous model selectable until the new path is verified. A trial is not automatic authorization to publish changes or personal comparison media.
