# Make a greeting with an AI agent

You can guide an agent using ordinary language: give it a recording or video link, the passage to use, and the greeting you want. The bundled [Dubiously Yours skill](../skills/dubiously-yours/SKILL.md) tells the agent how to operate this project, check the computer, prepare an audition and finish the chosen take.

The agent needs permission to read files and run local commands. This is guidance for an agent, not a browser app or a hosted generation service. Model setup still needs disk space and an appropriate machine; the skill helps establish that before a long download or render.

## Start a conversation

Open this project folder in your agent and give it this prompt, replacing the example details:

> Read skills/dubiously-yours/SKILL.md and use it to help me make a ten-second Finnish birthday greeting. Use this video: [paste a link], around 1:00–2:00. The recipient is Alex. Make it gently sarcastic. Check whether my computer has enough space and resources first, handle the setup and files, and let me hear a short sample before doing the video.

Reading the skill file explicitly works without a global installation. If your agent supports installing Agent Skills, install the `skills/dubiously-yours` folder using that agent's installer. Keep the skill's `references/` and `agents/` subfolders together. A separately installed skill still needs the project checkout; model environments and recordings belong in the checkout, not in the installed skill folder. In agents that support named skill invocation, use `$dubiously-yours` after installation.

The agent should handle the configuration, commands and bookkeeping. You supply creative direction and judge whether the voice and video work. Useful follow-ups include:

- “Use the next minute instead; it has a clearer close-up.”
- “The name sounds wrong. Compare two short versions before continuing.”
- “Make only the audio for now.”
- “Would another model improve the mouth during pauses? Compare the evidence before downloading anything.”

## What the resource check means

The agent runs this read-only helper using the intended greeting length:

```bash
python3 scripts/check_resources.py --mode video --duration 10 --json
```

It reports available storage on the relevant filesystems, RAM, CPU/platform and accessible GPU memory. It estimates the remaining pinned model downloads, environment/cache space and temporary media workspace. Existing model sizes reduce the estimated missing download; this does not establish that those files are intact. Separate setup/model checks verify assets before inference.

The report separates current measurements from estimates. `ready_to_try` permits a short trial; it does not promise that every model or long video fits. A confirmed shortage should stop the affected operation. Unavailable GPU diagnostics require investigation rather than an assertion that the machine has no GPU. Audio can run on CPU; the current video path requires NVIDIA CUDA. Other applications retain control of their resources—the helper does not stop or unload them.

The [performance guide](PERFORMANCE.md) contains dated observations from one computer. New models need their own estimates and tests. Source downloads and the number of retained takes can add substantial space beyond the planned greeting itself.

## Auditions and newer models

Start with a short voice sample. The user hears the name, accent and delivery before the agent spends time rendering a full greeting. After video export, watch the mouth, captions and ending. Decoding and transcript checks are useful but cannot decide whether the result is convincing.

The skill keeps the working model versions for routine use. It can research newer models when asked or when a specific limitation warrants it, using dated primary sources and a small comparison. Research is separate from installation: candidate trials should use an isolated environment, preserve the baseline and respect the agreed download/time budget.

The current job format has no separate phonetic-text override. Source footage supplies blinks and head motion; the skill does not create support for a new feature simply by requesting it. It should explain a limitation and propose a concrete next step.

Personal source recordings, names, job files and outputs stay in ignored local directories. Video exports retain the visible AI-parody label, and generation retains the speech model's built-in watermark. Sharing a finished greeting is a separate decision from generating it.
