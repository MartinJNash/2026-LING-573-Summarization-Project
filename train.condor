executable = run_on_patas.sh
arguments  = --base-model $(model) --output-dir results/$(name)/best $(peft)
error      = $(name).err.txt
output     = $(name).out.txt
log        = $(name).log.txt

getenv     = true
notification = never
transfer_executable = false
request_memory = 8192
request_GPUs = 1                                                                                     
Requirements = (Machine == "patas-gn3.ling.washington.edu")

queue model, name, peft from (
  GanjinZero/biobart-v2-base,    gpu-biobart-base-lora,   --use-peft
  GanjinZero/biobart-v2-large,   gpu-biobart-large-lora,  --use-peft
  facebook/bart-base,            gpu-bart-base-lora,      --use-peft
  facebook/bart-large,           gpu-bart-large-lora,     --use-peft
)
