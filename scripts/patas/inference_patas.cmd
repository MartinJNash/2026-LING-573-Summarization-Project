executable = scripts/inference_patas.sh
arguments  = --model $(model) --output results/outputs/$(name).json
error      = logs/$(name).inf.err
output     = logs/$(name).inf.out
log        = logs/$(name).inf.log
getenv              = true
notification        = never
transfer_executable = false
request_memory      = 8192
request_GPUs        = 1
Requirements        = (Machine == "patas-gn3.ling.washington.edu")

queue model, name from (
  facebook/bart-base,                               bart-base-baseline
  mjnash-uw/bart-base-lora,                         bart-base-lora
  GanjinZero/biobart-v2-base,                       biobart-base-baseline
  mjnash-uw/biobart-base-lora,                      biobart-base-lora
  Pika4028/biobart-v2-large-multiclinsum-lora,      biobart-large-lora
)