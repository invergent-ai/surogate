# Ultra Pool Selection Report

## Evidence Sources

- Joined 9-model direct slice: 80 tasks; domains {'math': 20, 'code': 20, 'science': 20, 'general': 20}.
- Full open direct bank: 8900 tasks; domains {'math': 2000, 'code': 3300, 'science': 1800, 'general': 1800}.
- Open agentic bank: 80 tau tasks; domains {'tau_airline': 40, 'tau_retail': 40}.
- Live frontier tau shard: 4 tasks; domains {'tau_airline': 2, 'tau_retail': 2}.
- Live coding-agent shard: 3 tasks; domains {'coding': 3}.
- Frontier slice binarization: commercial/frontier task success is `r_bar >= 0.5`, matching the derived table.
- Pool score is task-level coverage: a task is covered when any selected worker solves it.

## OpenRouter Catalog

Legacy OpenRouter prices are USD per million tokens. Current Yunwu runs do not report cost here; external cost monitoring is authoritative.

model         slug                           input / MTok  output / MTok
------------  -----------------------------  ------------  -------------
flash         deepseek/deepseek-v4-flash     $0.090        $0.180       
deepseek-pro  deepseek/deepseek-v4-pro       $0.435        $0.870       
glm           z-ai/glm-5.2                   $0.950        $3.000       
kimi-code     moonshotai/kimi-k2.7-code      $0.740        $3.500       
mimo          xiaomi/mimo-v2.5-pro           $0.435        $0.870       
minimax       minimax/minimax-m3             $0.300        $1.200       
opus          anthropic/claude-opus-4.8      $5.000        $25.000      
gemini        google/gemini-3.1-pro-preview  $2.000        $12.000      
gpt           openai/gpt-5.5                 $5.000        $30.000      

## Joined 9-Model Direct Accuracy

model         overall  math    code    science  general
------------  -------  ------  ------  -------  -------
flash          53.8%    40.0%   50.0%   60.0%    65.0% 
deepseek-pro   46.2%    45.0%   30.0%   35.0%    75.0% 
glm            47.5%    45.0%   25.0%   45.0%    75.0% 
kimi-code      38.8%    20.0%   25.0%   40.0%    70.0% 
mimo           33.8%    15.0%   20.0%   35.0%    65.0% 
minimax        31.2%    15.0%   10.0%   35.0%    65.0% 
opus           68.8%    40.0%   80.0%   75.0%    80.0% 
gemini         67.5%    55.0%   60.0%   70.0%    85.0% 
gpt            63.7%    20.0%   80.0%   75.0%    80.0% 

## Direct Coverage On Joined Slice

pool               members                                                                coverage
-----------------  ---------------------------------------------------------------------  --------
commercial-only    opus, gemini, gpt                                                       76.2%  
open-only          flash, deepseek-pro, glm, kimi-code, mimo, minimax                      67.5%  
proposed-six       opus, gemini, gpt, glm, flash, mimo                                     78.8%  
all-nine           flash, deepseek-pro, glm, kimi-code, mimo, minimax, opus, gemini, gpt   78.8%  
minimal-best-four  flash, glm, opus, gemini                                                78.8%  

Best subsets by direct coverage on the joined slice:

size  coverage  best subset(s)                                                                                                                                                                     
----  --------  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1      68.8%    opus                                                                                                                                                                               
2      75.0%    opus+gemini                                                                                                                                                                        
3      77.5%    flash+opus+gemini                                                                                                                                                                  
4      78.8%    flash+glm+opus+gemini                                                                                                                                                              
5      78.8%    flash+deepseek-pro+glm+opus+gemini; flash+glm+kimi-code+opus+gemini; flash+glm+mimo+opus+gemini; flash+glm+minimax+opus+gemini (+1 ties)                                           
6      78.8%    flash+deepseek-pro+glm+kimi-code+opus+gemini; flash+deepseek-pro+glm+mimo+opus+gemini; flash+deepseek-pro+glm+minimax+opus+gemini; flash+deepseek-pro+glm+opus+gemini+gpt (+6 ties)

Leave-one-out direct coverage for the proposed six:

removed  coverage without  delta kept  bootstrap 95% CI
-------  ----------------  ----------  ----------------
opus      76.2%              2.5%      [  0.0%,   6.2%]
gemini    76.2%              2.5%      [  0.0%,   6.2%]
gpt       78.8%              0.0%      [  0.0%,   0.0%]
glm       77.5%              1.2%      [  0.0%,   3.8%]
flash     77.5%              1.2%      [  0.0%,   3.8%]
mimo      78.8%              0.0%      [  0.0%,   0.0%]

Challenger direct swaps/additions against the proposed six:

comparison               coverage  delta vs proposed  bootstrap 95% CI
-----------------------  --------  -----------------  ----------------
add deepseek-pro          78.8%      0.0%             [  0.0%,   0.0%]
deepseek-pro for opus     76.2%     -2.5%             [ -6.2%,   0.0%]
deepseek-pro for gemini   76.2%     -2.5%             [ -6.2%,   0.0%]
deepseek-pro for gpt      78.8%      0.0%             [  0.0%,   0.0%]
deepseek-pro for glm      77.5%     -1.2%             [ -3.8%,   0.0%]
deepseek-pro for flash    77.5%     -1.2%             [ -3.8%,   0.0%]
deepseek-pro for mimo     78.8%      0.0%             [  0.0%,   0.0%]
add kimi-code             78.8%      0.0%             [  0.0%,   0.0%]
kimi-code for opus        77.5%     -1.2%             [ -3.8%,   0.0%]
kimi-code for gemini      77.5%     -1.2%             [ -3.8%,   0.0%]
kimi-code for gpt         78.8%      0.0%             [  0.0%,   0.0%]
kimi-code for glm         77.5%     -1.2%             [ -3.8%,   0.0%]
kimi-code for flash       77.5%     -1.2%             [ -3.8%,   0.0%]
kimi-code for mimo        78.8%      0.0%             [  0.0%,   0.0%]
add minimax               78.8%      0.0%             [  0.0%,   0.0%]
minimax for opus          76.2%     -2.5%             [ -6.2%,   0.0%]
minimax for gemini        76.2%     -2.5%             [ -6.2%,   0.0%]
minimax for gpt           78.8%      0.0%             [  0.0%,   0.0%]
minimax for glm           77.5%     -1.2%             [ -3.8%,   0.0%]
minimax for flash         77.5%     -1.2%             [ -3.8%,   0.0%]
minimax for mimo          78.8%      0.0%             [  0.0%,   0.0%]

## Full Open Direct Bank

model         overall
------------  -------
flash          59.9% 
deepseek-pro   45.3% 
glm            46.8% 
kimi-code      43.6% 
mimo           33.0% 
minimax        34.4% 

Best open-only direct subset of size 1: flash at  59.9%.
Best open-only direct subset of size 4: flash+glm+kimi-code+minimax at  69.2%.

## Open Agentic Bank

model         overall
------------  -------
flash          42.5% 
deepseek-pro   32.5% 
glm            47.5% 
kimi-code      40.0% 
mimo           56.2% 
minimax        35.0% 

Agentic core `glm+flash+mimo` coverage:  72.5%.

removed  coverage without  delta kept  bootstrap 95% CI
-------  ----------------  ----------  ----------------
glm       63.7%              8.8%      [  2.5%,  15.0%]
flash     68.8%              3.8%      [  0.0%,   8.8%]
mimo      56.2%             16.2%      [  8.8%,  25.0%]

Best open-only agentic subset of size 1: mimo at  56.2%.
Best open-only agentic subset of size 3: flash+glm+mimo at  72.5%.

## Live Tau Frontier Shard

Rows: 36 expected worker-task cells across 4 tasks.

model         successes  success rate  reported cost/task  reported total
------------  ---------  ------------  ------------------  --------------
flash         1/4         25.0%        $0.005142           $0.020570     
deepseek-pro  2/4         50.0%        $0.031571           $0.126284     
glm           1/4         25.0%        $0.032181           $0.128725     
kimi-code     3/4         75.0%        $0.017752           $0.071010     
mimo          2/4         50.0%        $0.036013           $0.144052     
minimax       2/4         50.0%        $0.007979           $0.031915     
opus          3/4         75.0%        $0.228554           $0.914214     
gemini        2/4         50.0%        $0.124297           $0.497188     
gpt           2/4         50.0%        $0.160232           $0.640928     

Task-level solvers:

domain       task            solvers                                                  
-----------  --------------  ---------------------------------------------------------
tau_airline  tau-airline-18  opus                                                     
tau_airline  tau-airline-3   kimi-code, mimo, opus                                    
tau_retail   tau-retail-3    flash, deepseek-pro, glm, kimi-code, minimax, gemini, gpt
tau_retail   tau-retail-7    deepseek-pro, kimi-code, mimo, minimax, opus, gemini, gpt

## Live Coding-Agent Shard

Rows: 27 expected worker-task cells across 3 SWE-smith tasks.

model         successes  success rate  reported cost/task  reported total
------------  ---------  ------------  ------------------  --------------
flash         0/3          0.0%        $0.011891           $0.035674     
deepseek-pro  1/3         33.3%        $0.016834           $0.050503     
glm           1/3         33.3%        $0.027689           $0.083068     
kimi-code     3/3        100.0%        $0.283495           $0.850484     
mimo          2/3         66.7%        $0.014630           $0.043889     
minimax       1/3         33.3%        $0.014898           $0.044693     
opus          0/3          0.0%        $0.357901           $1.073704     
gemini        0/3          0.0%        $0.542093           $1.626278     
gpt           1/3         33.3%        $0.692997           $2.078992     

Task-level solvers:

domain  task                                                          solvers                      
------  ------------------------------------------------------------  -----------------------------
coding  getnikola__nikola.0f4c230e.func_pm_remove_loop__dt3xvmca      glm, kimi-code, mimo, gpt    
coding  pydicom__pydicom.7d361b3d.func_pm_remove_cond__nh9m18q5       kimi-code, minimax           
coding  pygments__pygments.27649ebb.func_pm_ctrl_invert_if__x2f92u9g  deepseek-pro, kimi-code, mimo

Coding-primary ranking:

model         solved  reported cost/task  reported total
------------  ------  ------------------  --------------
kimi-code     3/3     $0.283495           $0.850484     
mimo          2/3     $0.014630           $0.043889     
minimax       1/3     $0.014898           $0.044693     
deepseek-pro  1/3     $0.016834           $0.050503     
glm           1/3     $0.027689           $0.083068     
gpt           1/3     $0.692997           $2.078992     
flash         0/3     $0.011891           $0.035674     
opus          0/3     $0.357901           $1.073704     
gemini        0/3     $0.542093           $1.626278     

Saved-rollout audit for commercial failures versus Kimi-Code:

model      task       reward  status  diff_len  elapsed  reported cost
---------  ---------  ------  ------  --------  -------  -------------
opus       pydicom    0       ok      1051      88s      $0.34        
gemini     pydicom    0       ok      483       116s     $0.02        
kimi-code  pydicom    1       ok      807       534s     $0.22        
gemini     pygments   0       ok      855       120s     $0.34        
opus       pygments   0       ok      855       136s     $0.43        
kimi-code  pygments   1       ok      496       307s     $0.09        
opus       getnikola  0       ok      1711      447s     $0.30        
gemini     getnikola  0       ok      1711      799s     $1.26        
kimi-code  getnikola  1       ok      1711      1347s    $0.54        

Diff-length audit notes:

- getnikola has diff_len=1711 for gemini, glm, kimi-code, mimo, opus.
- pygments has diff_len=855 for gemini, mimo, minimax, opus.

## Quality-First Ultra Decision

The product target is a general agentic model, not a coding-only model.
The main pool should be frontier triad plus empirically useful open/specialist workers.
The objective is to train workflows that beat each individual frontier/specialist baseline, not to select the cheapest single-axis worker set.

- Quality-first core: `opus+gemini+gpt+kimi-code+mimo+glm+flash`.
- Optional expanded pool: `opus+gemini+gpt+kimi-code+mimo+glm+flash+minimax+deepseek-pro`.
- Use the seven-worker core for the first serious GRPO pilot unless rollout budget supports all nine.
- Treat the open-only coding pool as a systems-integration / low-cost ablation, not the main Ultra implementation.

Role intent:

- Opus: debugger, verifier, security/code reviewer, hard agentic escalation, final repair.
- Gemini: science/factual specialist, long-context reasoner, knowledge-heavy aggregator, final synthesizer.
- GPT: planner, math/physics specialist, algorithm designer, alternate-perspective reviewer.
- Kimi-Code: primary OpenCode builder, implementation specialist, repair worker.
- MiMo: tool-dialogue worker, agentic executor, cheap independent attempt, procedural repair.
- GLM: open generalist, structured coding/debugging worker, secondary builder.
- Flash: strong fast open direct worker for easy subtasks, first-pass answers, and low-risk branches.
- MiniMax and DeepSeek-Pro: optional expanded-pool challengers until held-out workflow evidence proves core value.

## Scaffold-Aware Coding Layer

OpenCode is one coding harness adapter, not the full coding data distribution.
Claude Code and Codex traces should be first-class trace sources, and Claude Code/Codex should be first-class scaffold backends where available.
A worker identity is `model + scaffold + settings`, not only a model name.

Trace source mix for the coding portion:

source                 share   purpose                                        
---------------------  ------  -----------------------------------------------
OpenCode traces        25-35%  Open and controllable baseline harness         
Claude Code traces     25-35%  Opus-style debugging and long-horizon repo work
Codex traces           25-35%  GPT/Codex-style building, repair, and skills   
Fresh benchmark tasks  10-20%  Avoid overfitting to existing agent traces     

Scaffold-aware coding worker candidates:

- `codex:gpt-5-codex`
- `claude-code:opus-4.8`
- `opencode:kimi-code`
- `opencode:mimo`
- `opencode:glm`
- `direct:gemini-3.1-pro`
- `direct:gpt-5.5`
- `direct:opus-4.8`
- `opencode:flash`

Fair scaffold-aware baselines:

- Claude Code solo
- Codex solo
- OpenCode/Kimi solo
- best direct frontier model solo
- best-of-N single scaffold
- single-scaffold self-reflection
- fixed multi-agent workflow
- trained Fugu-Ultra Conductor

Implementation milestones:

- Trace ingestion: AgentTrace schema plus OpenCode, Claude Code, and Codex adapters.
- Harness parity canary: run the same toy repo tasks through OpenCode/Kimi, OpenCode/MiMo, Claude Code/Opus, and Codex/GPT.
- Scaffold-aware pool tournament: test Codex builder, Claude Code debugger, OpenCode/Kimi builder, OpenCode/MiMo repair, and OpenCode/GLM alternate builder roles.
- GRPO training: train over scaffold-aware worker IDs; the workflow JSON stays the same while worker_id resolution changes.

## Diagnostic Role-Weighted Table

The table below is diagnostic only. It helps expose tradeoffs, but it is not allowed to drop the frontier triad from the quality-first candidate pool by optimizing a tiny shard.
Current diagnostic weights: direct80=0.10, hist_tau80_open=0.20, live_tau4=0.35, coding3=0.35.

candidate                           size  members                                                                weighted  direct80  hist_tau80_open  live_tau4  coding3
----------------------------------  ----  ---------------------------------------------------------------------  --------  --------  ---------------  ---------  -------
quality-first/expanded-nine         9     opus, gemini, gpt, kimi-code, mimo, glm, flash, minimax, deepseek-pro  0.9313    0.7875    0.7625           1.0000     1.0000 
deployment-constrained/agentic-six  6     opus, glm, flash, mimo, kimi-code, minimax                             0.9300    0.7750    0.7625           1.0000     1.0000 
quality-first/core-seven            7     opus, gemini, gpt, kimi-code, mimo, glm, flash                         0.9287    0.7875    0.7500           1.0000     1.0000 
drop-gemini-add-kimi                6     opus, gpt, glm, flash, mimo, kimi-code                                 0.9275    0.7750    0.7500           1.0000     1.0000 
drop-flash-add-kimi                 6     opus, gemini, gpt, glm, mimo, kimi-code                                0.9225    0.7750    0.7250           1.0000     1.0000 
drop-glm-add-kimi                   6     opus, gemini, gpt, flash, mimo, kimi-code                              0.9175    0.7750    0.7000           1.0000     1.0000 
drop-mimo-add-kimi                  6     opus, gemini, gpt, glm, flash, kimi-code                               0.9137    0.7875    0.6750           1.0000     1.0000 
drop-opus-add-kimi                  6     gemini, gpt, glm, flash, mimo, kimi-code                               0.8400    0.7750    0.7500           0.7500     1.0000 
coding-ablation/open-six            6     kimi-code, mimo, glm, deepseek-pro, minimax, flash                     0.8325    0.6750    0.7625           0.7500     1.0000 
coding-ablation/positive-five       5     kimi-code, mimo, glm, deepseek-pro, minimax                            0.8175    0.5500    0.7500           0.7500     1.0000 
original-six                        6     opus, gemini, gpt, glm, flash, mimo                                    0.8071    0.7875    0.7250           1.0000     0.6667 

Top strict-six subsets by diagnostic weighted score:

subset                                      weighted  direct80  hist_tau80_open  live_tau4  coding3
------------------------------------------  --------  --------  ---------------  ---------  -------
flash+glm+kimi-code+mimo+minimax+opus       0.9300    0.7750    0.7625           1.0000     1.0000 
flash+glm+mimo+minimax+opus+gemini          0.9287    0.7875    0.7500           1.0000     1.0000 
flash+glm+kimi-code+mimo+opus+gemini        0.9287    0.7875    0.7500           1.0000     1.0000 
flash+glm+kimi-code+mimo+opus+gpt           0.9275    0.7750    0.7500           1.0000     1.0000 
flash+deepseek-pro+glm+kimi-code+mimo+opus  0.9275    0.7750    0.7500           1.0000     1.0000 
glm+kimi-code+mimo+minimax+opus+gpt         0.9263    0.7625    0.7500           1.0000     1.0000 
glm+kimi-code+mimo+minimax+opus+gemini      0.9263    0.7625    0.7500           1.0000     1.0000 
flash+glm+mimo+minimax+opus+gpt             0.9263    0.7625    0.7500           1.0000     1.0000 
flash+deepseek-pro+glm+mimo+minimax+opus    0.9263    0.7625    0.7500           1.0000     1.0000 
flash+kimi-code+mimo+minimax+opus+gemini    0.9250    0.7750    0.7375           1.0000     1.0000 

- If a deployment constraint forces six workers today, the diagnostic six is `opus+glm+flash+mimo+kimi-code+minimax`.
- That six-worker compression is not the scientific default for Fugu-Ultra.

## Diagnostic Equal-Stratum Table

This table reports each stratum separately and gives an equal-weight average for audit only.
It is not the decision rule for the general-agentic pool; it is kept to expose dilution and sensitivity.
Historical tau only contains open-worker measurements, so commercial workers receive no credit in that stratum.

candidate                           size  members                                                                direct80  hist_tau80_open  live_tau4  coding3  equal avg
----------------------------------  ----  ---------------------------------------------------------------------  --------  ---------------  ---------  -------  ---------
original-six                        6     opus, gemini, gpt, glm, flash, mimo                                    0.7875    0.7250           1.0000     0.6667   0.7948   
quality-first/core-seven            7     opus, gemini, gpt, kimi-code, mimo, glm, flash                         0.7875    0.7500           1.0000     1.0000   0.8844   
quality-first/expanded-nine         9     opus, gemini, gpt, kimi-code, mimo, glm, flash, minimax, deepseek-pro  0.7875    0.7625           1.0000     1.0000   0.8875   
deployment-constrained/agentic-six  6     opus, glm, flash, mimo, kimi-code, minimax                             0.7750    0.7625           1.0000     1.0000   0.8844   
coding-ablation/open-six            6     kimi-code, mimo, glm, deepseek-pro, minimax, flash                     0.6750    0.7625           0.7500     1.0000   0.7969   
coding-ablation/positive-five       5     kimi-code, mimo, glm, deepseek-pro, minimax                            0.5500    0.7500           0.7500     1.0000   0.7625   
drop-opus-add-kimi                  6     gemini, gpt, glm, flash, mimo, kimi-code                               0.7750    0.7500           0.7500     1.0000   0.8187   
drop-gemini-add-kimi                6     opus, gpt, glm, flash, mimo, kimi-code                                 0.7750    0.7500           1.0000     1.0000   0.8812   
drop-flash-add-kimi                 6     opus, gemini, gpt, glm, mimo, kimi-code                                0.7750    0.7250           1.0000     1.0000   0.8750   
drop-mimo-add-kimi                  6     opus, gemini, gpt, glm, flash, kimi-code                               0.7875    0.6750           1.0000     1.0000   0.8656   
drop-glm-add-kimi                   6     opus, gemini, gpt, flash, mimo, kimi-code                              0.7750    0.7000           1.0000     1.0000   0.8688   

Top strict-six subsets by equal-stratum average:

subset                                      equal avg  direct80  hist_tau80_open  live_tau4  coding3
------------------------------------------  ---------  --------  ---------------  ---------  -------
flash+glm+mimo+minimax+opus+gemini          0.8844     0.7875    0.7500           1.0000     1.0000 
flash+glm+kimi-code+mimo+opus+gemini        0.8844     0.7875    0.7500           1.0000     1.0000 
flash+glm+kimi-code+mimo+minimax+opus       0.8844     0.7750    0.7625           1.0000     1.0000 
flash+glm+kimi-code+mimo+opus+gpt           0.8812     0.7750    0.7500           1.0000     1.0000 
flash+deepseek-pro+glm+kimi-code+mimo+opus  0.8812     0.7750    0.7500           1.0000     1.0000 
flash+kimi-code+mimo+minimax+opus+gemini    0.8781     0.7750    0.7375           1.0000     1.0000 
glm+kimi-code+mimo+minimax+opus+gpt         0.8781     0.7625    0.7500           1.0000     1.0000 
glm+kimi-code+mimo+minimax+opus+gemini      0.8781     0.7625    0.7500           1.0000     1.0000 
flash+glm+mimo+minimax+opus+gpt             0.8781     0.7625    0.7500           1.0000     1.0000 
flash+deepseek-pro+glm+mimo+minimax+opus    0.8781     0.7625    0.7500           1.0000     1.0000 

## Coding-Focused Ablation

If the product target were coding-primary, the live coding shard would be decisive until contradicted by a larger coding rerun.
That is not the final Ultra decision here, but it defines how the Conductor should route coding-heavy work.
Reported costs below are legacy provider telemetry where available; do not treat zero/missing Yunwu cost as free inference.

- Provisional low-cost coding pool, fixed six: `kimi-code+mimo+glm+deepseek-pro+minimax+flash`.
- Coding-positive five, no filler: `kimi-code+mimo+glm+deepseek-pro+minimax`.
- `kimi-code` is the anchor: 3/3 solved on the live SWE-smith/OpenCode shard.
- `mimo` is the second coding worker: 2/3 solved with the lowest successful legacy reported cost per task.
- `glm`, `deepseek-pro`, and `minimax` each solved 1/3 and are cheap enough to keep as challengers/coverage workers.
- `flash` solved 0/3; include it only as a cheap fixed-six filler or direct-QA worker, not as a coding-positive result.
- `gpt` solved 1/3 but had much higher legacy reported cost than the open workers and added no task coverage beyond Kimi/MiMo/GLM on this shard.
- `opus` and `gemini` solved 0/3 while having much higher legacy reported cost than MiMo; they are not justified as coding-core workers from current evidence.
- Saved rollouts show Opus/Gemini were not empty-diff or tool-calling failures: status was `ok`, errors were null, and nonzero source diffs were produced.
- Caveat: n=3 is too small. The remaining uncertainty is agent behavior/prompt sensitivity, especially patch-once-and-stop behavior versus Kimi-Code's longer iteration.
- Nikola has repeated diff_len values across several independent agents, so inspect actual patches before overinterpreting diff length on that task.

## Current Scientific Conclusion

- Direct-only evidence was insufficient: `opus+gemini+gpt+glm+flash+mimo` matches all-nine direct coverage on the joined slice at  78.8%, but it misses coding coverage.
- Final proposed quality-first core: `opus+gemini+gpt+kimi-code+mimo+glm+flash`.
- Optional expanded candidate universe: `opus+gemini+gpt+kimi-code+mimo+glm+flash+minimax+deepseek-pro`.
- Coding implementation should be scaffold-aware: include OpenCode, Claude Code, and Codex as trace sources and harness backends.
- Opus, Gemini, and GPT stay because Fugu-Ultra is quality-first and must beat those same frontier models as individual baselines.
- Kimi-Code and MiMo are mandatory because they carry the coding-agent signal and also contribute live tau coverage.
- GLM remains as the strongest open generalist; Flash remains as the strong fast open direct worker.
- MiniMax and DeepSeek-Pro are optional expanded-pool workers; include them in the fixed-workflow tournament if rollout budget can support the larger action space.
- The next scientific spend should run a performance-first role tournament over the quality-first pool, then prune by held-out workflow contribution.

## Preregistered Low-Spend Paid Test

Budget cap: $200.00. With Yunwu, enforce this via the external spend monitor because provider-reported cost may be absent.

Stage 1: saved-rollout and prompt-behavior audit before more spend.

- Inspect saved OpenCode transcripts and actual patches for Opus/Gemini/Kimi on the three coding tasks.
- Confirm whether Opus/Gemini stopped after one plausible patch while Kimi-Code persisted through longer test/repair loops.
- Inspect Nikola actual patches because repeated diff_len=1711 appears across multiple agents and may be a diff-capture or task-specific quirk.
- Do not route Opus/Gemini as default first-pass OpenCode builders from direct/tau performance alone.

Stage 2: broaden coding-agent evidence.

- Expand from 3 SWE-smith tasks to 12-20 tasks before spending on more direct QA.
- Prioritize the quality-first core plus MiniMax/DeepSeek-Pro challengers if budget permits.
- Score by task coverage and marginal contribution, not standalone average accuracy.

Stage 3: mixed workflow-role tournament.

- Test GPT specifically as planner/math/alternate-reasoning worker, not just another direct worker.
- Test Opus on debugging, verification, security/review, and hard tool-use/airline-style tasks where live tau found unique coverage.
- Test Gemini on science/factual/long-context synthesis and aggregator roles.
- Test Kimi/MiMo/GLM fixed workflows for coding repair and synthesis.

Stage 4: tau/tool-dialogue expansion.

- Add hard tau airline tasks because Opus and Kimi/MiMo separated there.
- Keep task selection discriminative: avoid all-solved retail tasks and all-failed dead zones.

Decision rule:

- Keep a worker if leave-one-out removal lowers paired held-out workflow success or moves the cost-quality frontier.
- Reject an excluded challenger if no swap improves the proposed pool by at least 1 point and its paired CI is not positive.
- If a worker only helps one capability family, keep it only when that family has declared product weight.
