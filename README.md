# Vera runbook: ThinkingCap 27B FP8 + vLLM + Qwen Code

**Tested on Vera, 13 October 2026.** This is a from-scratch recipe for a Chalmers/C3SE user, based on a successful single-NVIDIA-A40 deployment of `bottlecapai/ThinkingCap-Qwen3.8-27B-FP8`. The tested server used vLLM **0.29.0**, **252,000** maximum context, **FP8 KV cache**, three-token **MTP speculative decoding**, and `xhigh` reasoning. The observed A40 serving rate in one coding-style generation was roughly **37–47 generated tokens/s**; actual performance varies with prompts.

Three downloadable companion files are provided with this guide:

- `start-thinkingcap.sh`: reusable server launcher, inside a Slurm allocation.
- `thinkingcap-3day.sbatch`: optional three-day batch-service job.
- `qwen-settings-workstation.json`: Qwen Code settings for a workstation tunnel on localhost port 8001.

**Replace `YOUR_CID` and your Vera project ID everywhere.** Do not reuse another researcher's project without being a member of it. Do not place access tokens in scripts, Git repositories, or chat logs.

## 1. Access Vera

1. In [SUPR](https://supr.naiss.se), join an approved project, accept the user agreement, and request a Vera account. Your Chalmers CID by itself does not automatically confer cluster access.
2. Connect from a Chalmers/GU network, or first use the Chalmers VPN from elsewhere.
3. Log in from your local computer:

```bash
ssh YOUR_CID@vera2.c3se.chalmers.se
```

On `vera2` (a login node), inspect your project, storage, and available GPUs:

```bash
projinfo
C3SE_quota
jobinfo -p vera
sinfo -N -p vera -O 'NodeList:22,StateCompact:12,Gres:26,GresUsed:30' | grep -E 'NODELIST|gpu:A40'
```

- `idle~`: idle but in a power-saving state; the node may need to wake up.
- `mix`: partially allocated; it can still have free GPUs.
- `GRES` describes installed GPUs; `GRES_USED` describes allocations. Slurm decides what resources are actually available.

**Never run the model on `vera2`**. Login nodes are for access, editing, job submission, and light checks.

## 2. Request an A40 (interactive, for first-time testing)

To make the interactive session less fragile, start **tmux on `vera2`**, then launch `srun` inside it:

```bash
tmux new -s vera-gpu
# Inside tmux on vera2:
srun -A YOUR_VERA_PROJECT \
  -p vera \
  --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=128G --gpus-per-node=A40:1 \
  --time=08:00:00 --pty bash
```

Our first attempt with eight CPUs was rejected: Vera tied an A40 to **exactly 16 allocated CPU cores** for this request. Leave `--cpus-per-task=16`, even though the serving process can use `OMP_NUM_THREADS=1` internally.

Once `srun` starts, you are on a **compute node**. Verify:

```bash
hostname
nvidia-smi
echo "$SLURM_JOB_ID"
echo "$TMPDIR"
df -h "$TMPDIR"
```

Our job received approximately **210 GB of node-local `$TMPDIR`**; check the amount for **your own job**. Do not infer the space from the total size of `/local` or `/cephyr` shown by `df -h`.

**tmux basics:** `Ctrl+b`, then `d` detaches; `tmux attach -t vera-gpu` reattaches after logging back into `vera2`. In a second tmux window, `Ctrl+b`, then `c` creates a window and `Ctrl+b`, then `n` switches windows. If tmux is unavailable, check `command -v screen`; the alternative is `screen -S vera-gpu`, detach with `Ctrl+a`, then `d`, and reattach with `screen -r vera-gpu`.

**Limitation:** tmux/screen protects against ordinary terminal disconnects, not Slurm time limits or reboots of the login/compute node. C3SE recommends batch jobs rather than long-lived interactive `srun`. Use section 8 for a three-day server.

## 3. Storage and the container

A Vera user's persistent home quota is generally **30 GiB**; the ThinkingCap checkpoint alone was around **29 GiB**, and the container is several more GiB. Check your own quota with `C3SE_quota`.

For an initial run, keep big downloads in **job-local scratch** (`$TMPDIR`). This data is **automatically erased when the Slurm job ends**, even if the job is cancelled. For repeated service, arrange persistent **Cephyr/Mimer project storage** and set `HF_HUB_CACHE` and `SIF` to that approved directory. A large `df -h /cephyr` result does *not* override your personal quota.

On the allocated **compute node**, prepare the persistent Hugging Face *token* path and the temporary caches:

```bash
mkdir -p "$HOME/.config/vera-thinkingcap"
chmod 700 "$HOME/.config/vera-thinkingcap"
export HF_TOKEN_PATH="$HOME/.config/vera-thinkingcap/hf-token"
export HF_HOME="$TMPDIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
export APPTAINER_TMPDIR="$TMPDIR/apptainer-tmp"
mkdir -p "$HF_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_TOKEN_PATH="$HF_TOKEN_PATH"
```

Check the [ThinkingCap model's access conditions](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-FP8) and accept them through your own Hugging Face account. Also review its license for your intended research or shared-service use.

Vera had older shared vLLM containers. This exact model's publisher specifies **vLLM 0.29**, so the tested method was to pull its official container into `$TMPDIR`:

```bash
apptainer pull "$TMPDIR/vllm-0.29.0.sif" docker://vllm/vllm-openai:v0.29.0
apptainer exec --nv --bind "$TMPDIR:$TMPDIR" \
  "$TMPDIR/vllm-0.29.0.sif" vllm --version
# Expected: 0.29.0
```

Authenticate **once** (interactive prompt; do not post your token anywhere):

```bash
apptainer exec --nv --bind "$TMPDIR:$TMPDIR" \
  "$TMPDIR/vllm-0.29.0.sif" hf auth login
chmod 600 "$HF_TOKEN_PATH"
apptainer exec --nv --bind "$TMPDIR:$TMPDIR" \
  "$TMPDIR/vllm-0.29.0.sif" hf auth whoami
```

The large model files should go to the temporary `HF_HUB_CACHE`, while your small authentication file persists in your home directory. Confirm where the token ended up (`test -s "$HF_TOKEN_PATH"`); do not cancel the job before establishing persistent authorization if you plan to use a batch service.

You can optionally pre-download model weights, or let vLLM download them at first startup:

```bash
apptainer exec --nv --bind "$TMPDIR:$TMPDIR" \
  "$TMPDIR/vllm-0.29.0.sif" hf download \
  bottlecapai/ThinkingCap-Qwen3.8-27B-FP8
```

## 4. Start the tested vLLM server

Place the supplied `start-thinkingcap.sh` into `$HOME/vera-thinkingcap/` and make it executable:

```bash
mkdir -p "$HOME/vera-thinkingcap"
chmod +x "$HOME/vera-thinkingcap/start-thinkingcap.sh"
```

For an interactive experiment inside your `srun` compute-node shell (in a tmux session on `vera2`):

```bash
"$HOME/vera-thinkingcap/start-thinkingcap.sh" > "$TMPDIR/vllm.log" 2>&1 &
echo "Server launcher PID: $!"
tail -f "$TMPDIR/vllm.log"
```

`Ctrl+C` stops **only tail**; it does not stop the background server. Use tmux to keep the enclosing interactive Slurm shell alive. Do not `exit` the original `srun` shell until finished.

The launcher does all of the following:

- Uses vLLM 0.29.0 and `bottlecapai/ThinkingCap-Qwen3.8-27B-FP8`.
- Binds to compute-node **`127.0.0.1:8000`**, not a public interface.
- Requests `--max-model-len 252000`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.94`, and `--kv-cache-dtype fp8`.
- Enables prefix caching, chunked prefill, text-only mode, Qwen XML tool parsing, Qwen3 reasoning extraction, and **MTP with three draft tokens**.
- Sets default chat-template reasoning effort to **`xhigh`**, enables request/KV metrics, and sets `OMP_NUM_THREADS=1`.
- Stores model downloads and compile caches in scratch, while retaining a tiny persistent HF token file.

We first tried 0.90 GPU utilization, but at 252k context **plus MTP**, vLLM reported insufficient KV-cache memory. **0.94 worked on our A40** and yielded an estimated cache capacity of 346,682 tokens. Another allocation may profile differently; monitor startup logs.

## 5. Verify the API on the compute node

Run these on the allocated compute node (use a second shell, or stop `tail -f` first):

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

Look for `"max_model_len":252000` and a successful `Application startup complete` in the logs. Then try:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"bottlecapai/ThinkingCap-Qwen3.8-27B-FP8",
    "messages":[{"role":"user","content":"What is 17 times 23? Explain briefly."}],
    "max_tokens":512
  }'
```

For MTP and queue stats:

```bash
curl -s http://127.0.0.1:8000/metrics | grep -Ei 'spec_decode|speculative' | head -35
curl -s http://127.0.0.1:8000/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions_total)'
nvidia-smi
```

Other useful commands on `vera2`: `squeue -u "$USER"`, `jobinfo -u "$USER"`, `scontrol show job JOBID`, `job_stats.py JOBID`, `projinfo`, and `scancel JOBID`.

## 6. Connect a laptop/workstation over SSH (working two-hop route)

We initially tried `ssh -J` directly from a workstation to the compute node, but got `Too many authentication failures` or failed compute-node password authentication. The **working solution** used one SSH connection *originating from the Vera login node* and one connection from the workstation.

**A. On `vera2`, start the inner tunnel.** In another terminal, identify the current Slurm compute node (it may differ for each new job):

```bash
squeue -u "$USER" -o '%.12i %.25j %.10T %.25N'
NODE=vera-m07-04  # REPLACE using the NODELIST shown above
ssh "$NODE" hostname  # test that login-node -> your allocated node works
```

Start the inner tunnel in **a separate tmux session on `vera2`** so it survives disconnects from your laptop:

```bash
tmux new -s vera-tunnel
NODE=vera-m07-04  # REPLACE with your current allocated node inside this new tmux shell
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18000:127.0.0.1:8000 "$NODE"
# Detach with Ctrl+b, then d; don't exit the SSH process.
```

**B. On your own workstation**, connect its localhost port **8001** to Vera's login-node port 18000 (in a new terminal or workstation tmux session):

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8001:127.0.0.1:18000 \
  YOUR_CID@vera2.c3se.chalmers.se
```

**C. In a second workstation terminal**, test the complete path:

```bash
curl -sS http://127.0.0.1:8001/v1/models
```

Your computer's `localhost:8001` now goes to the vLLM service on the allocated GPU. Choose different *local* port numbers if 8001 is occupied. Recreate the inner tunnel whenever Slurm assigns a different compute node.

**Important security caveat:** `127.0.0.1:18000` on `vera2` is loopback-only from the network's perspective, but **other users with processes running on that same shared login node may be able to use the port**. This is **not authenticated per user** and is suitable only for a controlled test, not a confidential or institutional multi-user service. Do not bind unprotected vLLM to `0.0.0.0`. Consult C3SE about authenticated services when sharing beyond your own account.

## 7. Configure Qwen Code

**Recommended:** Run Qwen Code on your own workstation, where your source checkout and developer tools already live. Qwen Code communicates with the GPU via the SSH tunnel and uses the GPU allocation only for inference. You may also install Qwen Code on the compute node to work on a repository there; in that case use port **8000**, not 8001, in the settings.

On your workstation, install Qwen Code using the [official instructions](https://qwenlm.github.io/qwen-code-docs/en/users/quickstart/). For Linux/macOS standalone installation, you can download and inspect the official script before running it:

```bash
curl -fsSL \
  https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen-standalone.sh \
  -o /tmp/install-qwen-standalone.sh
bash /tmp/install-qwen-standalone.sh
qwen --version
```

Back up `~/.qwen/settings.json`, then use the supplied `qwen-settings-workstation.json` (it includes the full `modelProviders.openai` configuration, `qwen-chat-template` reasoning profile, 252k window, `xhigh` effort, and base URL `http://127.0.0.1:8001/v1`):

```bash
mkdir -p ~/.qwen
cp ~/.qwen/settings.json ~/.qwen/settings.json.bak 2>/dev/null || true
cp /path/to/qwen-settings-workstation.json ~/.qwen/settings.json
printf '%s\n' 'VLLM_API_KEY=unused-local-placeholder' >> ~/.qwen/.env
chmod 600 ~/.qwen/.env ~/.qwen/settings.json
```

`VLLM_API_KEY` here is merely a placeholder for the client's OpenAI-compatible API path; **the vLLM server is not protected by that string**. Also, Qwen Code's `timeout: 0` and `streamIdleTimeoutMs: 0` do not remove the separate total stream lifetime default. For extremely long generations, consider `QWEN_STREAM_MAX_LIFETIME_MS=0` in the *workstation's* environment, if appropriate.

Then run from a working repository **on the workstation**:

```bash
cd /path/to/your/repository
qwen
```

Use Qwen Code's `/model` and `/effort` commands to verify selection. **Do not run `qwen` on the login node to perform compute-heavy local tools.** Use your workstation or your allocated compute node.

## 8. Switch from interactive testing to a three-day Slurm service

A **batch job** is the preferred way to serve across SSH disconnects. It does not depend on the survival of a tmux session on `vera2`.

1. Ensure the Hugging Face token file was saved persistently in `$HOME/.config/vera-thinkingcap/hf-token`.
2. Put both supplied scripts into `$HOME/vera-thinkingcap/` and `chmod +x` the `.sh` file.
3. Replace `REPLACE_WITH_YOUR_VERA_PROJECT` in `thinkingcap-3day.sbatch` with your actual project ID (from `projinfo`).
4. From **`vera2`**, submit:

```bash
cd "$HOME/vera-thinkingcap"
sbatch thinkingcap-3day.sbatch
squeue -u "$USER" -o '%.12i %.20j %.10T %.25N'
```

Once it runs, `thinkingcap-JOBID.log` in the submission directory records its startup. The job uses `srun start-thinkingcap.sh` with the same **16 CPU cores, 1 A40, 128 GB system RAM, 252k/MTP** configuration. It will pull the container and model into its new `$TMPDIR` unless you configure shared persistent storage. Use `tail -f thinkingcap-JOBID.log` to watch it and update the tunnel's `NODE` to the new `squeue` node list.

The batch job keeps running even if you log out. The inner SSH tunnel on the login node does **not** have the same guarantee: if the login node is rebooted, recreate the tunnel. To stop the service:

```bash
scancel JOBID
```

**Accounting:** On Vera, an A40 contributes about 16 equivalent CPU core-hours per GPU-hour in addition to the 16 physically allocated CPUs. This setup costs approximately **32 core-hours per hour** while it runs, or **2,304 equivalent core-hours if it runs the full three days**. Cancelling stops further consumption; there is no charge for the unused portion of the requested wall time. The regular job limit is seven days.

## 9. How it behaves with multiple Qwen Code users

This example deliberately uses **`--max-num-seqs 1`**. Multiple clients can connect, but only one inference sequence runs at once; additional work waits in vLLM's scheduler. You don't need to create a separate manual queue per colleague.

`--max-model-len 252000` is a **per-request ceiling**, *not* 252k tokens reserved for every connected user. KV cache is allocated from a **shared GPU pool** and reusable prefixes can remain cached after requests finish until the space is needed. In the tested A40 startup, the estimated total KV cache capacity was ~346k tokens. Thus two independent ~120k-token prefixes might both remain cached between turns, though **cache hits must be measured** and cannot be assumed (especially with the model's hybrid recurrent/attention architecture). Two simultaneous full-length 252k requests cannot both fit in that pool.

MTP is separate from KV caching: it uses the model's speculative head to draft up to three tokens per verification step. `xhigh` is a reasoning-effort setting; it is not itself a 252k-context reservation. Actual time-to-first-token and decode throughput depend heavily on active context and the accepted speculative-token rate.

## 10. Troubleshooting quick reference

| Symptom | Likely explanation / action |
|---|---|
| `8 * 1 must match 16 * 1` | On these A40 Vera requests, use 16 allocated CPUs per GPU. `OMP_NUM_THREADS=1` does not change Slurm's CPU allocation. |
| Home is full | `C3SE_quota`; move **model and SIF**, not only code, to `$TMPDIR` or approved project storage. `--mem=128G` is system RAM, not GPU VRAM. |
| vLLM says insufficient KV cache for 252k | The tested 0.90 GPU memory utilization failed; **0.94 succeeded** with MTP. Check GPU availability/logs if another node behaves differently. |
| API does not start | `tail -n 80 "$TMPDIR/vllm.log"` for interactive use, or `tail -n 80 thinkingcap-JOBID.log` for batch. |
| Direct `ssh -J` authentication fails | Use the **two-hop forwarding** pattern above. It worked when direct compute-node authentication from a workstation did not. |
| `/v1/models` fails locally | Check `squeue`, vLLM `/health` **on compute node**, inner `vera2` tunnel, then outer workstation tunnel. |
| MTP reports FlashInfer fallbacks | This was expected on the A40; MTP still worked. Disabling FlashInfer isn't an automatic speed improvement. |
| Huge decoding latency | Inspect TTFT, actual prompt length, prefix-cache hits, MTP acceptance, GPU utilization; a 252k limit alone doesn't imply 252k tokens are sent. |
| Job or terminal disappears | `srun` and its `$TMPDIR` are tied to job lifetime; use `sbatch` for a durable service. |

## Official references

- [C3SE: Introduction to Vera](https://www.c3se.chalmers.se/documentation/first_time_users/intro-vera/slides/)
- [C3SE: Slurm batch/GPU jobs](https://www.c3se.chalmers.se/documentation/submitting_jobs/running_jobs/)
- [C3SE: SSH / tunneling](https://www.c3se.chalmers.se/documentation/connecting/ssh/)
- [C3SE: vLLM](https://www.c3se.chalmers.se/documentation/software/machine_learning/vllm/)
- [ThinkingCap FP8 model and license](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-FP8)
- [Qwen Code: model providers](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/)
- [Qwen Code: installation](https://qwenlm.github.io/qwen-code-docs/en/users/quickstart/)
